"""Fast SAM2 automatic (segment-everything) image segmentation on Jetson AGX Orin.

Applies the fp16 + compiled-encoder optimizations benchmarked in
bench/RESULTS.md (2.1x fp16 vs fp32 eager, 2.9x combined with a compiled
encoder, no accuracy loss) -- but those numbers were measured for a single
point-prompted predict() call (one encode + one decode). This script instead
runs SAM2AutomaticMaskGenerator, which reuses the same fp16 image embedding
but decodes a whole grid of point prompts (points_per_side**2 of them,
batched) to find every object in the image, so total latency is dominated by
many repeated decode passes rather than the one-shot encode.

bench/RESULTS.md's advice not to compile the (small, control-flow-heavy)
mask decoder holds for a single decode call, where the extra compile time
isn't worth it -- but here the decoder runs points_per_side**2 / points_per_batch
times *per image*, so the amortized win flips: compiling it too cuts
steady-state segment-everything latency by ~35-45% (measured: 7.5s -> 4.9s on
truck.jpg, 40.6s -> 25.0s total across the 7 example images), with mask
counts verified identical to the uncompiled decoder on every example image.
points_per_batch was also bumped 64 -> 128 (fewer, bigger decode batches);
pushing to 256 shaved a bit more time but changed one image's mask count by
1 (fp16 kernel-fusion rounding flipping a near-threshold mask) and 1024
triggered Jetson unified-memory allocator errors -- 128 was the largest batch
size that stayed both fast and exact in this testing.

The first call to predict() after building the model pays a one-time
torch.compile warmup cost (tens of seconds, or a few seconds if Inductor's
on-disk kernel cache from a prior run is warm). Always warm up at startup
with dummy images before serving real requests -- see `warmup()` below.

Note on latency variance: the ~77ms full-predict number in bench/RESULTS.md
was measured in a tight back-to-back loop, which keeps the GPU busy enough
for Jetson's DVFS governor to sustain max clocks. A single isolated call
(e.g. one request in an interactive/low-QPS server) can idle the GPU down
between requests, so its latency will land higher (100-300+ms) until the
governor ramps back up -- warmup() running a burst instead of one call
mitigates but does not eliminate this. For consistent low-latency serving,
run `sudo jetson_clocks` once on the device to lock GPU/CPU/EMC clocks to
their maximum and disable DVFS ramp-up lag entirely.

Note on input size: SAM2Transforms resizes every image to a fixed square
resolution before it reaches the encoder (see sam2/utils/transforms.py), so
the compiled encoder sees the same input shape regardless of source image
resolution -- one warmup() covers a whole directory of mixed-resolution
images with no per-shape recompile.

This script walks a directory of images, runs segment-everything on each,
and reports per-image and aggregate timing (warmup separate from steady-state
segmentation).

Usage:
  python examples/fast_image_segmentation.py \\
      --images-dir examples/images --out-dir examples/segmented
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
DEFAULT_OUT_DIR = REPO_ROOT / "examples" / "segmented"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def build_fast_mask_generator(
    config_file: str = DEFAULT_CONFIG,
    ckpt_path: str | Path = DEFAULT_CKPT,
    device: str = "cuda",
    points_per_batch: int = 128,
    **mask_generator_kwargs,
) -> SAM2AutomaticMaskGenerator:
    """Build a SAM2AutomaticMaskGenerator with the Jetson speedups applied.

    Compiles both the image encoder (run once per image) and the mask
    decoder (run points_per_side**2 / points_per_batch times per image) --
    see the module docstring for why compiling the decoder is worth it here
    even though bench/RESULTS.md found it wasn't for a single predict() call.

    Call warmup() once before real inference so both torch.compile traces
    happen up front rather than on the first real request.
    """
    # TF32 matmuls are a free win for any fp32 remainder ops that fall
    # outside the fp16 autocast region.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    model = build_sam2(config_file, str(ckpt_path), device=device)
    model.image_encoder.forward = torch.compile(model.image_encoder.forward)
    model.sam_mask_decoder.forward = torch.compile(model.sam_mask_decoder.forward)
    return SAM2AutomaticMaskGenerator(
        model, points_per_batch=points_per_batch, **mask_generator_kwargs
    )


def warmup(
    mask_generator: SAM2AutomaticMaskGenerator,
    image_hw: tuple[int, int] = (1024, 1024),
    iters: int = 8,
) -> None:
    """Trigger torch.compile's first-call trace, then run a short burst.

    Do this once at process startup (e.g. before opening a server socket),
    not on the first real user request. A single dummy call is enough to
    pay the one-time torch.compile trace cost (tens of seconds), but Jetson's
    GPU clocks idle down between short/sparse calls and only ramp to max
    frequency under a few hundred ms of sustained load -- so a lone warmup
    call leaves the *next* real call running at a partially-clocked-down GPU
    (seen here as ~130ms instead of the ~80ms steady-state number). Running
    a short burst avoids surprising latency on the first real request.
    """
    dummy = np.zeros((*image_hw, 3), dtype=np.uint8)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(iters):
            mask_generator.generate(dummy)
    torch.cuda.synchronize()


def segment_all(
    mask_generator: SAM2AutomaticMaskGenerator,
    image: np.ndarray,
) -> list[dict]:
    """Run one fp16 automatic mask generation call over the whole image.

    Returns mask records (see SAM2AutomaticMaskGenerator.generate), sorted
    by descending area.
    """
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
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
    ap.add_argument(
        "--points-per-batch",
        type=int,
        default=128,
        help="Point prompts decoded per batch. Higher = fewer, bigger decode "
        "calls (faster, more GPU memory). 128 was the largest value that "
        "stayed both faster and mask-exact vs. the uncompiled decoder in "
        "testing on this device -- 256 was a bit faster but changed one "
        "mask's pass/fail by fp16 rounding, and 1024 hit Jetson unified-"
        "memory allocator errors.",
    )
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU required"

    images_dir = Path(args.images_dir)
    image_paths = find_images(images_dir)
    if not image_paths:
        raise SystemExit(f"no images ({sorted(IMAGE_EXTENSIONS)}) found in {images_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_generator = build_fast_mask_generator(
        args.config,
        args.checkpoint,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
    )

    t0 = time.perf_counter()
    warmup(mask_generator)
    torch.cuda.synchronize()
    warmup_s = time.perf_counter() - t0
    print(f"warmup (includes torch.compile trace): {warmup_s:.1f}s\n")

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
