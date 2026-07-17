"""Export the SAM2 prompt encoder + mask decoder to ONNX for TensorRT.

Scoped to exactly the path SAM2AutomaticMaskGenerator._process_batch() uses
(see sam2/sam2/automatic_mask_generator.py): point-only prompts (no boxes,
no mask_input refinement -- i.e. use_m2m=False, the default), one point per
query, multimask_output=True, repeat_image=True (the image embedding is
broadcast across the batch of point queries). That's a narrower scope than
the full SAM2ImagePredictor._predict() surface (which also supports boxes,
mask-refinement input, and single-image batching) -- this engine is built
for the segment-everything use case in examples/fast_image_segmentation.py,
not as a drop-in replacement for arbitrary predict() calls.

Unlike the encoder (fixed 1024x1024 input, see export_encoder.py), the
point-batch dimension here varies with --points-per-batch, so it's exported
with a dynamic axis and the TensorRT engine is built with an optimization
profile spanning the range actually used (see build_decoder_engine.sh).

Usage:
  python trt/export_decoder.py --out trt/decoder.onnx
"""
import argparse
import contextlib
from pathlib import Path

import torch
import torch.nn as nn

from sam2.build_sam import build_sam2


@contextlib.contextmanager
def _patch_repeat_interleave_for_tracing():
    """mask_decoder.predict_masks() calls
    torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0) to
    broadcast the (always batch-size-1) image embedding across the point
    batch. That hits a CUDA/CPU device-mismatch bug in aten's
    repeat_interleave specifically under torch.jit tracing (works fine
    eager) -- likely an artifact of how the tracer's dispatch mode handles
    repeat_interleave's internal index_select. Since dim 0 always has size 1
    here, repeat_interleave(x, n, dim=0) is exactly equivalent to
    x.expand(n, ...), which traces to a plain ONNX Expand op with no such
    issue. Patched only for the duration of the trace/export call.

    Note: under tracing with dynamic_axes requested, `repeats` (e.g.
    tokens.shape[0]) arrives as a traced 0-dim tensor rather than a plain
    Python int -- Tensor.expand() accepts either directly and traces to a
    correctly dynamic ONNX Expand node either way, so no int() coercion here.
    """
    orig = torch.repeat_interleave

    def patched(input, repeats, dim=None, **kwargs):
        if dim == 0 and input.shape[0] == 1:
            return input.expand(repeats, *input.shape[1:]).contiguous()
        return orig(input, repeats, dim=dim, **kwargs)

    torch.repeat_interleave = patched
    try:
        yield
    finally:
        torch.repeat_interleave = orig

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = "configs/sam2/sam2_hiera_s.yaml"
DEFAULT_CKPT = REPO_ROOT / "sam2" / "checkpoints" / "sam2_hiera_small.pt"

# Must match export_encoder.py's BB_FEAT_SIZES-derived output shapes.
IMAGE_EMBED_SHAPE = (1, 256, 64, 64)
HIGH_RES_FEAT0_SHAPE = (1, 32, 256, 256)
HIGH_RES_FEAT1_SHAPE = (1, 64, 128, 128)


class DecoderExportWrapper(nn.Module):
    """Wraps sam_prompt_encoder + sam_mask_decoder for the point-only,
    multimask_output=True, repeat_image=True path used by
    SAM2AutomaticMaskGenerator (see module docstring for scope).
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        image_embed: torch.Tensor,
        high_res_feat0: torch.Tensor,
        high_res_feat1: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
    ):
        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=(point_coords, point_labels), boxes=None, masks=None
        )
        low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=True,
            high_res_features=[high_res_feat0, high_res_feat1],
        )
        return low_res_masks, iou_predictions


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    ap.add_argument("--out", default=str(REPO_ROOT / "trt" / "decoder.onnx"))
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument(
        "--export-points-per-batch",
        type=int,
        default=128,
        help="Point-batch size to trace with. The exported graph's point-batch "
        "axis is marked dynamic regardless, so this only affects the dummy "
        "input used during tracing.",
    )
    args = ap.parse_args()

    model = build_sam2(args.config, args.checkpoint, device="cuda")
    model.eval()
    wrapper = DecoderExportWrapper(model).eval().cuda()

    n = args.export_points_per_batch
    dummy_image_embed = torch.randn(*IMAGE_EMBED_SHAPE, device="cuda")
    dummy_hr0 = torch.randn(*HIGH_RES_FEAT0_SHAPE, device="cuda")
    dummy_hr1 = torch.randn(*HIGH_RES_FEAT1_SHAPE, device="cuda")
    dummy_coords = torch.rand(n, 1, 2, device="cuda") * model.image_size
    dummy_labels = torch.ones(n, 1, dtype=torch.int32, device="cuda")

    with torch.no_grad():
        ref_masks, ref_iou = wrapper(
            dummy_image_embed, dummy_hr0, dummy_hr1, dummy_coords, dummy_labels
        )
    print("reference (eager fp32) output shapes:", ref_masks.shape, ref_iou.shape)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with _patch_repeat_interleave_for_tracing():
        torch.onnx.export(
            wrapper,
            (dummy_image_embed, dummy_hr0, dummy_hr1, dummy_coords, dummy_labels),
            str(out_path),
            input_names=[
                "image_embed",
                "high_res_feat0",
                "high_res_feat1",
                "point_coords",
                "point_labels",
            ],
            output_names=["low_res_masks", "iou_predictions"],
            dynamic_axes={
                "point_coords": {0: "num_points"},
                "point_labels": {0: "num_points"},
                "low_res_masks": {0: "num_points"},
                "iou_predictions": {0: "num_points"},
            },
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"exported to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
