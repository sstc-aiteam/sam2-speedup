"""Benchmark SAM2 (sam2_hiera_small) image-predictor inference on Jetson AGX Orin.

Measures:
  - image encoder forward pass (set_image)
  - full predict() call (encoder + prompt encoder + mask decoder)

Usage:
  python bench/bench_image_predictor.py [--dtype fp32|tf32|bf16|fp16] [--compile]
"""
import argparse
import time

import numpy as np
import torch
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

CONFIG = "configs/sam2/sam2_hiera_s.yaml"
CKPT = "checkpoints/sam2_hiera_small.pt"
IMAGE_PATH = "notebooks/images/truck.jpg"
POINT = np.array([[500, 375]])
LABEL = np.array([1])

WARMUP = 5
ITERS = 30


def sync():
    torch.cuda.synchronize()


def bench(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter() - t0)
    times = np.array(times) * 1000  # ms
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["fp32", "tf32", "bf16", "fp16"], default="fp32")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--compile-mode", default="default")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    if args.dtype == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    sam2_model = build_sam2(CONFIG, CKPT, device="cuda")
    if args.compile:
        sam2_model.image_encoder.forward = torch.compile(
            sam2_model.image_encoder.forward, mode=args.compile_mode
        )
    predictor = SAM2ImagePredictor(sam2_model)

    image = np.array(Image.open(IMAGE_PATH).convert("RGB"))

    autocast_dtype = None
    if args.dtype == "bf16":
        autocast_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        autocast_dtype = torch.float16

    def run_ctx():
        if autocast_dtype is not None:
            return torch.autocast("cuda", dtype=autocast_dtype)
        return torch.autocast("cuda", enabled=False)

    def encode():
        with torch.inference_mode(), run_ctx():
            predictor.set_image(image)

    def full_predict():
        with torch.inference_mode(), run_ctx():
            predictor.set_image(image)
            predictor.predict(point_coords=POINT, point_labels=LABEL, multimask_output=True)

    print(f"=== dtype={args.dtype} compile={args.compile} ===")

    enc_times = bench(encode)
    print(
        f"image encoder (set_image): mean={enc_times.mean():.2f}ms "
        f"median={np.median(enc_times):.2f}ms p95={np.percentile(enc_times, 95):.2f}ms "
        f"fps={1000/enc_times.mean():.2f}"
    )

    full_times = bench(full_predict)
    print(
        f"full predict (encoder+decoder): mean={full_times.mean():.2f}ms "
        f"median={np.median(full_times):.2f}ms p95={np.percentile(full_times, 95):.2f}ms "
        f"fps={1000/full_times.mean():.2f}"
    )

    print(f"peak_mem_MB={torch.cuda.max_memory_allocated()/1e6:.1f}")


if __name__ == "__main__":
    main()
