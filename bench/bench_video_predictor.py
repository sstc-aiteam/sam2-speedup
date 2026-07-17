"""Benchmark SAM2 (sam2_hiera_small) VIDEO predictor inference on Jetson AGX Orin.

Adapted from sam2/benchmark.py (upstream) for the small model + optional
vos_optimized (whole-model torch.compile) comparison.

Usage:
  python bench/bench_video_predictor.py [--vos-optimized]
"""
import argparse
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from sam2.build_sam import build_sam2_video_predictor

CONFIG = "configs/sam2/sam2_hiera_s.yaml"
CKPT = "checkpoints/sam2_hiera_small.pt"
VIDEO_DIR = "notebooks/videos/bedroom"

WARMUP = 3
RUNS = 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vos-optimized", action="store_true")
    args = ap.parse_args()

    assert torch.cuda.is_available()
    device = torch.device("cuda")

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    predictor = build_sam2_video_predictor(
        CONFIG, CKPT, device=device, vos_optimized=args.vos_optimized
    )

    frame_names = [
        p
        for p in os.listdir(VIDEO_DIR)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    num_frames = len(frame_names)
    print(f"video: {num_frames} frames from {VIDEO_DIR}")

    inference_state = predictor.init_state(video_path=VIDEO_DIR)

    ann_frame_idx, ann_obj_id = 0, 1
    points = np.array([[210, 350]], dtype=np.float32)
    labels = np.array([1], np.int32)

    predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    total, count = 0.0, 0
    per_run_fps = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with torch.autocast("cuda", torch.bfloat16), torch.inference_mode():
        for i in tqdm(range(RUNS), desc=f"Benchmarking (vos_optimized={args.vos_optimized})"):
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in predictor.propagate_in_video(inference_state):
                pass
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            per_run_fps.append(num_frames / elapsed)

            if i == WARMUP - 1:
                print(f"warmup mean FPS: {np.mean(per_run_fps):.2f}")
                total, count = 0.0, 0
                per_run_fps = []
            total += elapsed
            count += 1

    fps_arr = np.array(per_run_fps)
    print(f"=== vos_optimized={args.vos_optimized} ===")
    print(f"mean FPS: {fps_arr.mean():.2f}  median FPS: {np.median(fps_arr):.2f}")
    print(f"mean per-frame latency: {1000/fps_arr.mean():.2f} ms")
    print(f"peak_mem_MB={torch.cuda.max_memory_allocated()/1e6:.1f}")


if __name__ == "__main__":
    main()
