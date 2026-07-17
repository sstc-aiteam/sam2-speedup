"""Minimal TensorRT engine runner that binds torch CUDA tensors directly.

TensorRT 10's execute_async_v3 takes raw device pointers per tensor name, so
we can hand it torch.Tensor.data_ptr() straight from CUDA tensors -- no
pycuda, no separate host<->device copy management, no extra dependency.
Output tensors are pre-allocated once (shapes are static for both the
encoder and decoder graphs here) and reused across calls.

Uses the system-wide TensorRT python bindings (apt package, not in this
venv) via a sys.path insert -- see trt/README or the module docstrings in
export_encoder.py / export_decoder.py for why: the wheel would duplicate
~1GB+ already installed system-wide, and disk on this device is tight.
"""
import sys

sys.path.insert(0, "/usr/lib/python3.10/dist-packages")
import tensorrt as trt  # noqa: E402
import torch  # noqa: E402

TRT_DTYPE_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT8: torch.int8,
    trt.DataType.BOOL: torch.bool,
}

_LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTModule:
    """Loads a serialized TRT engine and runs it against fixed-shape torch
    CUDA tensors, keyed by the ONNX graph's input/output tensor names.
    """

    def __init__(self, engine_path: str, device: str = "cuda"):
        self.device = device
        with open(engine_path, "rb") as f, trt.Runtime(_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_names = []
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        # Pre-allocate fixed-shape output buffers (all shapes in this
        # project's encoder/decoder graphs are static once inputs are set).
        self._outputs = {}
        self.stream = torch.cuda.Stream(device=device)

    def _ensure_output_buffers(self):
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = TRT_DTYPE_TO_TORCH[self.engine.get_tensor_dtype(name)]
            buf = self._outputs.get(name)
            if buf is None or tuple(buf.shape) != shape or buf.dtype != dtype:
                self._outputs[name] = torch.empty(shape, dtype=dtype, device=self.device)

    def __call__(self, **inputs: torch.Tensor) -> dict:
        for name, tensor in inputs.items():
            tensor = tensor.contiguous()
            inputs[name] = tensor
            self.context.set_input_shape(name, tuple(tensor.shape))
            self.context.set_tensor_address(name, tensor.data_ptr())

        self._ensure_output_buffers()
        for name, buf in self._outputs.items():
            self.context.set_tensor_address(name, buf.data_ptr())

        with torch.cuda.stream(self.stream):
            ok = self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")
        return dict(self._outputs)
