"""Baseline (unoptimized) SAM2 automatic image segmentation.

Runs SAM2AutomaticMaskGenerator in plain fp32 eager mode -- no autocast, no
torch.compile on the encoder or decoder, no TF32, library-default
points_per_batch. This is the "just call the API" starting point that
examples/fast_image_segmentation.py's fp16 + compiled-encoder + compiled-
decoder optimizations are measured against; see compare_segmentation.py for
a side-by-side run of both on the same images.

Usage:
  python examples/baseline_image_segmentation.py \\
      --images-dir examples/images --out-dir examples/segmented_baseline
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = "configs/sam2/sam2_hiera_s.yaml"
DEFAULT_CKPT = REPO_ROOT / "sam2" / "checkpoints" / "sam2_hiera_small.pt"
DEFAULT_IMAGES_DIR = REPO_ROOT / "examples" / "images"
DEFAULT_OUT_DIR = REPO_ROOT / "examples" / "segmented_baseline"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def build_plain_mask_generator(
    config_file: str = DEFAULT_CONFIG,
    ckpt_path: str | Path = DEFAULT_CKPT,
    device: str = "cuda",
    **mask_generator_kwargs,
) -> SAM2AutomaticMaskGenerator:
    """Build a SAM2AutomaticMaskGenerator with no speedups applied: fp32
    eager, no torch.compile, PyTorch's out-of-the-box TF32 settings.
    """
    model = build_sam2(config_file, str(ckpt_path), device=device)
    return SAM2AutomaticMaskGenerator(model, **mask_generator_kwargs)


def warmup(
    mask_generator: SAM2AutomaticMaskGenerator,
    image_hw: tuple[int, int] = (1024, 1024),
    iters: int = 8,
) -> None:
    """Run a short burst before timed calls.

    There's no torch.compile trace to pay for here, but Jetson's GPU clocks
    still idle down between sparse calls and only ramp to max frequency
    under sustained load, so a burst is still worth running for a fair
    steady-state comparison against fast_image_segmentation.py's warmup().
    """
    dummy = np.zeros((*image_hw, 3), dtype=np.uint8)
    with torch.inference_mode():
        for _ in range(iters):
            mask_generator.generate(dummy)
    torch.cuda.synchronize()


def segment_all(
    mask_generator: SAM2AutomaticMaskGenerator,
    image: np.ndarray,
) -> list[dict]:
    """Run one fp32 eager automatic mask generation call over the whole image.

    Returns mask records (see SAM2AutomaticMaskGenerator.generate), sorted
    by descending area.
    """
    with torch.inference_mode():
        anns = mask_generator.generate(image)
    return sorted(anns, key=lambda ann: ann["area"], reverse=True)


def save_all_overlay(image: np.ndarray, anns: list[dict], out_path: str) -> None:
    """Blend a distinct color over every mask.

    Draws largest-area masks first so smaller masks nested inside a bigger
    one (e.g. a wheel inside a car) get painted on top and stay visible.
    """
    overlay = image.copy()
    rng = np.random.default_rng(0)
    for ann in anns:
        color = rng.integers(0, 256, size=3, dtype=np.uint8)
        mask = ann["segmentation"]
        overlay[mask] = (0.5 * overlay[mask] + 0.5 * color).astype(np.uint8)
    Image.fromarray(overlay).save(out_path)


def find_images(images_dir: Path) -> list[Path]:
    return sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    ap.add_argument("--images-dir", default=str(DEFAULT_IMAGES_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument(
        "--points-per-side",
        type=int,
        default=32,
        help="Density of the point-prompt grid used to find objects; "
        "total prompts = points_per_side**2. Higher = more masks, slower.",
    )
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU required"

    images_dir = Path(args.images_dir)
    image_paths = find_images(images_dir)
    if not image_paths:
        raise SystemExit(f"no images ({sorted(IMAGE_EXTENSIONS)}) found in {images_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_generator = build_plain_mask_generator(
        args.config, args.checkpoint, points_per_side=args.points_per_side
    )

    t0 = time.perf_counter()
    warmup(mask_generator)
    torch.cuda.synchronize()
    warmup_s = time.perf_counter() - t0
    print(f"warmup: {warmup_s:.1f}s\n")

    seg_times_ms = []
    for path in image_paths:
        image = np.array(Image.open(path).convert("RGB"))

        t0 = time.perf_counter()
        anns = segment_all(mask_generator, image)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        seg_times_ms.append(elapsed_ms)

        out_path = out_dir / f"{path.stem}_segmented.png"
        save_all_overlay(image, anns, str(out_path))

        w, h = image.shape[1], image.shape[0]
        print(
            f"{path.name:35s} {w:5d}x{h:<5d}  masks={len(anns):3d}  "
            f"{elapsed_ms:8.1f}ms  -> {out_path}"
        )

    total_seg_s = sum(seg_times_ms) / 1000
    mean_ms = sum(seg_times_ms) / len(seg_times_ms)
    print(f"\n{len(image_paths)} images: warmup {warmup_s:.1f}s, "
          f"segmentation total {total_seg_s:.1f}s, mean {mean_ms:.1f}ms/image")
    print(f"wall total (warmup + segmentation): {warmup_s + total_seg_s:.1f}s")
    print(f"outputs written to {out_dir}")


if __name__ == "__main__":
    main()
