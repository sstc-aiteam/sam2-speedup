"""Export the SAM2 image encoder to ONNX for TensorRT.

Exports exactly what SAM2ImagePredictor.set_image() computes on the encoder
side: model.forward_image() (Hiera trunk + FPN neck + the mask decoder's
conv_s0/conv_s1 high-res convs, which sam2_base.forward_image applies inline)
followed by _prepare_backbone_features() + the no_mem_embed add + the
seq->spatial reshape -- i.e. everything up to (and including) the
image_embed/high_res_feats the predictor caches and later feeds to the mask
decoder.

Fixed shapes throughout: SAM2Transforms resizes every input to a fixed
1024x1024 square (see sam2/utils/transforms.py) regardless of source image
size, and batch size is always 1 for single-image inference, so there is no
dynamic-shape concern here -- unlike the mask decoder, whose point-prompt
count varies per call.

Exported in fp32; the fp16 TensorRT engine is built from this graph with
`trtexec --fp16` (see build_encoder_engine.sh), matching the standard
ONNX->TRT workflow of exporting once in fp32 and letting TRT's builder do
the precision reduction + kernel selection.

Usage:
  python trt/export_encoder.py --out trt/encoder.onnx
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn

from sam2.build_sam import build_sam2

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = "configs/sam2/sam2_hiera_s.yaml"
DEFAULT_CKPT = REPO_ROOT / "sam2" / "checkpoints" / "sam2_hiera_small.pt"

# Fixed spatial size of the 3 FPN feature levels for a 1024x1024 input
# (strides 4, 8, 16) -- see SAM2ImagePredictor._bb_feat_sizes.
BB_FEAT_SIZES = [(256, 256), (128, 128), (64, 64)]


class EncoderExportWrapper(nn.Module):
    """Wraps SAM2Base.forward_image() + _prepare_backbone_features() into a
    single (image) -> (image_embed, high_res_feat0, high_res_feat1) module,
    matching what SAM2ImagePredictor.set_image() caches in self._features.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, img_batch: torch.Tensor):
        backbone_out = self.model.forward_image(img_batch)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], BB_FEAT_SIZES[::-1])
        ][::-1]
        # image_embed, high_res_feat0, high_res_feat1
        return feats[-1], feats[0], feats[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    ap.add_argument("--out", default=str(REPO_ROOT / "trt" / "encoder.onnx"))
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    model = build_sam2(args.config, args.checkpoint, device="cuda")
    model.eval()
    wrapper = EncoderExportWrapper(model).eval().cuda()

    dummy = torch.randn(1, 3, model.image_size, model.image_size, device="cuda")

    with torch.no_grad():
        ref_out = wrapper(dummy)
    print("reference (eager fp32) output shapes:", [t.shape for t in ref_out])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out_path),
        input_names=["image"],
        output_names=["image_embed", "high_res_feat0", "high_res_feat1"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"exported to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
