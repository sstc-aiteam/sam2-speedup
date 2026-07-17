"""Compare baseline_image_segmentation.py (fp32 eager) against the
TensorRT-backed pipeline (trt/segment_all_trt.py) over examples/images.

Same pattern as examples/compare_segmentation.py (baseline vs fp16+compile)
and trt/compare_trt.py (fp16+compile vs TensorRT) -- this fills in the third
leg of the triangle: baseline vs TensorRT directly, so the TRT engines' win
over the *unoptimized* starting point is visible in one table (rather than
having to chain the other two comparisons' numbers together).

Reports per-image latency/speedup, mask counts, and "coverage IoU" (IoU
between the union of all masks each run found -- see
examples/compare_segmentation.py's docstring for why that's the right check
for segment-everything instead of a mask-for-mask comparison).

Usage:
  python trt/compare_baseline_trt.py
"""
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "trt"))
sys.path.insert(0, str(REPO_ROOT / "examples"))

import baseline_image_segmentation as baseline  # noqa: E402
import segment_all_trt as trt_pipeline  # noqa: E402
from baseline_image_segmentation import find_images  # noqa: E402

IMAGES_DIR = REPO_ROOT / "examples" / "images"


def union_mask(anns: list[dict], shape: tuple[int, ...]) -> np.ndarray:
    union = np.zeros(shape[:2], dtype=bool)
    for ann in anns:
        union |= ann["segmentation"]
    return union


def run_all(build_fn, warmup_fn, segment_fn, image_paths: list[Path], label: str) -> tuple[dict, float]:
    print(f"[{label}]")
    mask_generator = build_fn()

    t0 = time.perf_counter()
    warmup_fn(mask_generator)
    torch.cuda.synchronize()
    warmup_s = time.perf_counter() - t0
    print(f"  warmup: {warmup_s:.1f}s")

    results = {}
    for path in image_paths:
        image = np.array(Image.open(path).convert("RGB"))
        t0 = time.perf_counter()
        anns = segment_fn(mask_generator, image)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        results[path.name] = {"anns": anns, "ms": elapsed_ms, "shape": image.shape}
        print(f"  {path.name:35s} masks={len(anns):3d}  {elapsed_ms:8.1f}ms")

    del mask_generator
    gc.collect()
    torch.cuda.empty_cache()
    return results, warmup_s


def main():
    assert torch.cuda.is_available(), "CUDA GPU required"

    image_paths = find_images(IMAGES_DIR)
    if not image_paths:
        raise SystemExit(f"no images found in {IMAGES_DIR}")

    base_results, base_warmup_s = run_all(
        baseline.build_plain_mask_generator,
        baseline.warmup,
        baseline.segment_all,
        image_paths,
        "baseline: fp32 eager",
    )
    print()
    trt_results, trt_warmup_s = run_all(
        trt_pipeline.build_trt_mask_generator,
        trt_pipeline.warmup,
        trt_pipeline.segment_all,
        image_paths,
        "trt: fp16 TensorRT encoder + decoder engines",
    )

    header = (
        f"\n{'image':35s} {'masks base/trt':>15s}  {'ms base/trt':>18s}  "
        f"{'speedup':>8s}  {'coverage IoU':>12s}"
    )
    print(header)
    total_base_ms = total_trt_ms = 0.0
    for path in image_paths:
        b = base_results[path.name]
        t = trt_results[path.name]
        total_base_ms += b["ms"]
        total_trt_ms += t["ms"]

        b_union = union_mask(b["anns"], b["shape"])
        t_union = union_mask(t["anns"], t["shape"])
        intersection = np.logical_and(b_union, t_union).sum()
        union_px = np.logical_or(b_union, t_union).sum()
        iou = intersection / union_px if union_px else 1.0
        speedup = b["ms"] / t["ms"]

        print(
            f"{path.name:35s} {len(b['anns']):>6d} / {len(t['anns']):<6d}  "
            f"{b['ms']:>8.1f} / {t['ms']:<7.1f}  {speedup:>7.2f}x  {iou:>11.4f}"
        )

    print(f"\nwarmup:       baseline {base_warmup_s:6.1f}s   trt {trt_warmup_s:6.1f}s")
    print(
        f"segmentation: baseline {total_base_ms/1000:6.1f}s   trt {total_trt_ms/1000:6.1f}s  "
        f"({total_base_ms/total_trt_ms:.2f}x overall)"
    )


if __name__ == "__main__":
    main()
