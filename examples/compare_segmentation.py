"""Compare baseline_image_segmentation.py (fp32 eager) against
fast_image_segmentation.py (fp16 + compiled encoder/decoder) over the same
images in examples/images.

Runs each mask generator over every image, then reports per-image:
  - latency and the resulting speedup
  - mask count from each run
  - "coverage IoU": the IoU between the union of all masks each run found.
    Individual object boundaries can shift a pixel or two between fp32 and
    fp16, and a threshold flip can occasionally add/drop a small mask (see
    fast_image_segmentation.py's docstring), so a mask-for-mask comparison
    isn't meaningful -- this instead checks that both runs agree on *which
    regions of the image are foreground* overall, the same spirit as
    bench/check_accuracy.py's best-mask IoU check but for segment-everything.

The two mask generators are built and run one at a time (baseline fully
finishes and is freed before the fast one is built) rather than held
resident together, since Jetson's GPU memory is shared with the whole
system.

Usage:
  python examples/compare_segmentation.py
"""
import gc
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import baseline_image_segmentation as baseline
import fast_image_segmentation as fast

REPO_ROOT = Path(__file__).resolve().parent.parent
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

    image_paths = baseline.find_images(IMAGES_DIR)
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
    fast_results, fast_warmup_s = run_all(
        fast.build_fast_mask_generator,
        fast.warmup,
        fast.segment_all,
        image_paths,
        "fast: fp16 + compiled encoder/decoder",
    )

    header = (
        f"\n{'image':35s} {'masks base/fast':>16s}  {'ms base/fast':>18s}  "
        f"{'speedup':>8s}  {'coverage IoU':>12s}"
    )
    print(header)
    total_base_ms = total_fast_ms = 0.0
    for path in image_paths:
        b = base_results[path.name]
        f = fast_results[path.name]
        total_base_ms += b["ms"]
        total_fast_ms += f["ms"]

        b_union = union_mask(b["anns"], b["shape"])
        f_union = union_mask(f["anns"], f["shape"])
        intersection = np.logical_and(b_union, f_union).sum()
        union_px = np.logical_or(b_union, f_union).sum()
        iou = intersection / union_px if union_px else 1.0
        speedup = b["ms"] / f["ms"]

        print(
            f"{path.name:35s} {len(b['anns']):>7d} / {len(f['anns']):<6d}  "
            f"{b['ms']:>8.1f} / {f['ms']:<7.1f}  {speedup:>7.2f}x  {iou:>11.4f}"
        )

    print(f"\nwarmup:       baseline {base_warmup_s:6.1f}s   fast {fast_warmup_s:6.1f}s")
    print(
        f"segmentation: baseline {total_base_ms/1000:6.1f}s   fast {total_fast_ms/1000:6.1f}s  "
        f"({total_base_ms/total_fast_ms:.2f}x overall)"
    )


if __name__ == "__main__":
    main()
