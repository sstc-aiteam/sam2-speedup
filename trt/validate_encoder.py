"""Numerically validate trt/encoder_fp16.engine against the PyTorch eager
fp32 reference on a real image, and benchmark it head-to-head against the
fp16+torch.compile encoder from examples/fast_image_segmentation.py.

Usage:
  python trt/validate_encoder.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "examples"))
sys.path.insert(0, str(REPO_ROOT / "trt"))

from export_encoder import EncoderExportWrapper  # noqa: E402
from trt_runner import TRTModule  # noqa: E402
from sam2.build_sam import build_sam2  # noqa: E402
from sam2.utils.transforms import SAM2Transforms  # noqa: E402

CONFIG = "configs/sam2/sam2_hiera_s.yaml"
CKPT = str(REPO_ROOT / "sam2/checkpoints/sam2_hiera_small.pt")
ENGINE = str(REPO_ROOT / "trt/encoder_fp16.engine")
IMAGE = REPO_ROOT / "examples/images/truck.jpg"


def main():
    model = build_sam2(CONFIG, CKPT, device="cuda")
    model.eval()
    wrapper = EncoderExportWrapper(model).eval().cuda()
    transforms = SAM2Transforms(resolution=model.image_size, mask_threshold=0.0)

    image = np.array(Image.open(IMAGE).convert("RGB"))
    input_image = transforms(image)[None, ...].cuda()

    with torch.no_grad():
        ref_embed, ref_hr0, ref_hr1 = wrapper(input_image)

    trt_encoder = TRTModule(ENGINE)
    trt_out = trt_encoder(image=input_image)
    trt_embed = trt_out["image_embed"]
    trt_hr0 = trt_out["high_res_feat0"]
    trt_hr1 = trt_out["high_res_feat1"]

    for name, ref, trt_t in [
        ("image_embed", ref_embed, trt_embed),
        ("high_res_feat0", ref_hr0, trt_hr0),
        ("high_res_feat1", ref_hr1, trt_hr1),
    ]:
        ref32 = ref.float()
        trt32 = trt_t.float()
        abs_diff = (ref32 - trt32).abs()
        cos = torch.nn.functional.cosine_similarity(
            ref32.flatten(), trt32.flatten(), dim=0
        )
        print(
            f"{name:16s} shape={tuple(trt_t.shape)!s:22s} "
            f"max_abs_diff={abs_diff.max().item():.4f} "
            f"mean_abs_diff={abs_diff.mean().item():.4f} "
            f"cos_sim={cos.item():.6f}"
        )

    # Timing: TRT engine vs. fp16+compiled encoder (examples path)
    N = 30
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        trt_encoder(image=input_image)
    torch.cuda.synchronize()
    trt_ms = (time.perf_counter() - t0) / N * 1000

    model.image_encoder.forward = torch.compile(model.image_encoder.forward)
    compiled_wrapper = EncoderExportWrapper(model).eval().cuda()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(8):  # warmup / trigger compile
            compiled_wrapper(input_image)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N):
            compiled_wrapper(input_image)
        torch.cuda.synchronize()
    compiled_ms = (time.perf_counter() - t0) / N * 1000

    print(f"\nfp16+torch.compile encoder: {compiled_ms:.2f} ms/call")
    print(f"TensorRT fp16 encoder:      {trt_ms:.2f} ms/call")
    print(f"speedup: {compiled_ms/trt_ms:.2f}x")


if __name__ == "__main__":
    main()
