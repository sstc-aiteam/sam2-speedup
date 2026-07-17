"""Segment-everything over examples/images using TensorRT engines for both
the image encoder and mask decoder, instead of PyTorch (eager, fp16
autocast, or torch.compile).

Builds a normal SAM2AutomaticMaskGenerator (so all of its crop/batch/NMS/
stability-score/RLE postprocessing logic runs unchanged) and swaps its
internal predictor for TRTSAM2ImagePredictor, which routes just the encoder
and decoder forward passes through trt/encoder_fp16.engine and
trt/decoder_fp16.engine. See trt/export_encoder.py, trt/export_decoder.py,
and trt/trt_image_predictor.py for what those engines cover and their scope
limits (point-only prompts, multimask_output=True -- exactly what
SAM2AutomaticMaskGenerator uses, and no more).

Mirrors examples/fast_image_segmentation.py's CLI/output shape so results
are directly comparable; see trt/compare_trt.py for a run-both-and-diff
driver analogous to examples/compare_segmentation.py.

Usage:
  python trt/segment_all_trt.py --images-dir examples/images --out-dir trt/segmented_trt
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "trt"))
sys.path.insert(0, str(REPO_ROOT / "examples"))

from trt_image_predictor import TRTSAM2ImagePredictor  # noqa: E402
from baseline_image_segmentation import find_images, save_all_overlay  # noqa: E402

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # noqa: E402
from sam2.build_sam import build_sam2  # noqa: E402

DEFAULT_CONFIG = "configs/sam2/sam2_hiera_s.yaml"
DEFAULT_CKPT = REPO_ROOT / "sam2" / "checkpoints" / "sam2_hiera_small.pt"
DEFAULT_IMAGES_DIR = REPO_ROOT / "examples" / "images"
DEFAULT_OUT_DIR = REPO_ROOT / "trt" / "segmented_trt"
DEFAULT_ENCODER_ENGINE = REPO_ROOT / "trt" / "encoder_fp16.engine"
DEFAULT_DECODER_ENGINE = REPO_ROOT / "trt" / "decoder_fp16.engine"


def build_trt_mask_generator(
    config_file: str = DEFAULT_CONFIG,
    ckpt_path: str = str(DEFAULT_CKPT),
    encoder_engine: str = str(DEFAULT_ENCODER_ENGINE),
    decoder_engine: str = str(DEFAULT_DECODER_ENGINE),
    points_per_side: int = 32,
    points_per_batch: int = 128,
) -> SAM2AutomaticMaskGenerator:
    model = build_sam2(config_file, ckpt_path, device="cuda")
    mask_generator = SAM2AutomaticMaskGenerator(
        model, points_per_side=points_per_side, points_per_batch=points_per_batch
    )
    # Swap in the TRT-backed predictor; matches the max_hole_area/
    # max_sprinkle_area SAM2AutomaticMaskGenerator's own __init__ used
    # (min_mask_region_area defaults to 0 in both).
    mask_generator.predictor = TRTSAM2ImagePredictor(
        model, encoder_engine, decoder_engine
    )
    return mask_generator


def warmup(mask_generator, image_hw=(1024, 1024), iters=8) -> None:
    dummy = np.zeros((*image_hw, 3), dtype=np.uint8)
    for _ in range(iters):
        mask_generator.generate(dummy)
    torch.cuda.synchronize()


def segment_all(mask_generator, image: np.ndarray) -> list[dict]:
    anns = mask_generator.generate(image)
    return sorted(anns, key=lambda ann: ann["area"], reverse=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    ap.add_argument("--encoder-engine", default=str(DEFAULT_ENCODER_ENGINE))
    ap.add_argument("--decoder-engine", default=str(DEFAULT_DECODER_ENGINE))
    ap.add_argument("--images-dir", default=str(DEFAULT_IMAGES_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--points-per-side", type=int, default=32)
    ap.add_argument("--points-per-batch", type=int, default=128)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU required"

    images_dir = Path(args.images_dir)
    image_paths = find_images(images_dir)
    if not image_paths:
        raise SystemExit(f"no images found in {images_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_generator = build_trt_mask_generator(
        args.config,
        args.checkpoint,
        args.encoder_engine,
        args.decoder_engine,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
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
    print(f"outputs written to {out_dir}")


if __name__ == "__main__":
    main()
