# SAM2 (sam2_hiera_small) Inference Speedup — Jetson AGX Orin

Device: Jetson AGX Orin 64GB, JetPack 6 / L4T 36.4.7, CUDA 12.6, GPU compute
capability 8.7 (Ampere), TensorRT 10.3.

Environment: `.venv` (Python 3.10), PyTorch 2.9.1 + torchvision 0.24.1 from
`pypi.jetson-ai-lab.io/jp6/cu126`.

## Image predictor results (1800x1200 input, `notebooks/images/truck.jpg`)

| Config | Encoder (ms) | Full predict (ms) | FPS | Speedup |
|---|---|---|---|---|
| fp32 (no TF32) | 196.5 | 223.2 | 4.48 | 1.0x |
| TF32 | 135.4 | 154.2 | 6.49 | 1.4x |
| bf16 autocast | 85.8 | 114.5 | 8.74 | 2.0x |
| fp16 autocast | 86.9 | 105.1 | 9.51 | 2.1x |
| bf16 + `torch.compile` (encoder) | 57.4 | 92.9 | 10.77 | 2.4x |
| **fp16 + `torch.compile` (encoder)** | **53.7** | **77.1** | **12.97** | **2.9x** |

Accuracy check (`bench/check_accuracy.py`): fp16+compile vs fp32 eager on the
same point prompt — best-mask IoU 0.9996, confidence-score delta 0.0002.
Effectively lossless.

Run: `python bench/bench_image_predictor.py --dtype fp16 --compile`

## Video predictor (`notebooks/videos/bedroom`, 200 frames)

| Config | FPS | ms/frame |
|---|---|---|
| bf16 autocast + TF32, eager | 9.18 | 108.9 |
| `vos_optimized=True` (whole-model `torch.compile`) | fails | — |

The upstream `vos_optimized=True` path (`SAM2VideoPredictorVOS`, full-model
compile) hits a PyTorch 2.9 inductor/CUDA-graphs bug: it raises
`RuntimeError: accessing tensor output of CUDAGraphs that has been
overwritten by a subsequent run` inside `memory_attention.py`'s
`normed_output.transpose(0, 1)`. This is an upstream SAM2/PyTorch inductor
compatibility issue (cudagraph-tree buffer reuse aliasing a transposed
view), not something specific to this device — not pursued further here.
The `torch.compile`-the-image-encoder-only technique from the image-predictor
results above is not affected by this bug and is the recommended path.

## Recommended production setup

Use `torch.autocast("cuda", dtype=torch.float16)` (or bf16, near-identical
speed) plus `torch.compile` on `image_encoder.forward` — the Hiera backbone
dominates cost. This is a **2.9x** speedup over naive fp32 eager with no
measurable accuracy loss, taking sam2_hiera_small from ~4.5 FPS to ~13 FPS on
a single 1800x1200 image on Jetson AGX Orin.

## Jetson-specific setup gotchas (fixed in `.venv/bin/activate`)

The Jetson wheels from `pypi.jetson-ai-lab.io/jp6/cu126` are missing a few
pieces that stock x86 PyTorch wheels normally bundle or auto-resolve:

1. **`libcudss.so.0` not found** — torch 2.9's `libtorch_cuda.so` links
   directly against cuDSS, which isn't part of JetPack. Fix: `pip install
   nvidia-cudss-cu12` (only that package — installing it naively also pulls
   a redundant `cuda-toolkit[cublas]` meta-package that duplicates JetPack's
   own cuBLAS; uninstall `cuda-toolkit`/`nvidia-cublas-cu12`/
   `nvidia-cuda-nvrtc-cu12` afterwards) and add its lib dir to
   `LD_LIBRARY_PATH` (torch's own preload glue doesn't search
   `nvidia/cu12/lib` for cudss).
2. **`torch.compile` / Triton fails with `TritonMissing`** — no default
   Triton wheel for aarch64; install `triton==3.6.0` from the same Jetson
   index.
3. **Triton JIT compile fails, missing `cuda.h`** — the Jetson triton wheel
   ships no `triton/backends/nvidia/include/` directory at all. Fix: set
   `TRITON_CUDACRT_PATH` to the system CUDA toolkit's include dir
   (`/usr/local/cuda-12.6/targets/aarch64-linux/include`).
4. **`RuntimeError: Cannot find ptxas`** — the Jetson triton wheel's bundled
   `backends/nvidia/bin/` only ships `ptxas-blackwell`, not plain `ptxas`.
   Fix: set `TRITON_PTXAS_PATH=/usr/local/cuda-12.6/bin/ptxas` (from the
   system CUDA toolkit) and add `/usr/local/cuda-12.6/bin` to `PATH`.

All four are already wired up in `.venv/bin/activate` — just `source` it.

Disk space was tight throughout (device had ~5.7G free at the start,
57G total, 90% used before any of this work); the redundant cuBLAS pull in
(1) and pip's HTTP cache both needed cleaning up (`pip cache purge`) to stay
afloat. Final state after all installs + checkpoint: ~4G free.
