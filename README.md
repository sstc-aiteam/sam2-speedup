# sam2-speedup

Inference speedup work for SAM2 (`sam2_hiera_small`) on Jetson AGX Orin:
fp16 + `torch.compile` tuning for both single-point prediction and
segment-everything (`SAM2AutomaticMaskGenerator`), plus a full ONNX export +
TensorRT engine pipeline evaluated against it.

Full write-up, methodology, and every result table lives in
[`NOTES.md`](NOTES.md); this README is the condensed quick start + summary.

## Quick start

Target device: Jetson AGX Orin, JetPack 6 / L4T 36.4.7, CUDA 12.6, TensorRT
10.3. (Should also run on other CUDA 12.6 + Ampere-or-newer GPUs, but the
Jetson-wheel-specific fixes below won't apply.)

```bash
# 1. Clone this repo, then clone SAM2 into sam2/ (vendored, not committed here)
git clone https://github.com/facebookresearch/sam2.git sam2

# 2. Create the venv and install PyTorch from the Jetson wheel index
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools   # stock setuptools can't do PEP 660 editable installs
pip install torch torchvision --index-url https://pypi.jetson-ai-lab.io/jp6/cu126

# 3. Install SAM2 editable (skip the optional CUDA post-processing extension)
cd sam2 && SAM2_BUILD_CUDA=0 pip install -e . --no-build-isolation && cd ..

# 4. Download the checkpoint
cd sam2/checkpoints && ./download_ckpts.sh sam2_hiera_small.pt && cd ../..
```

Jetson's PyTorch/Triton wheels are missing a few pieces `torch.compile`
needs (`libcudss`, Triton's CUDA headers, `ptxas`). All fixes for this are
already baked into `.venv/bin/activate` once created per the steps above —
see `NOTES.md` §1 if you hit `libcudss.so.0`, `TritonMissing`, or `Cannot
find ptxas` errors.

### Run the examples

```bash
source .venv/bin/activate

# Segment every object in each image under examples/images/, fp16+compile
python examples/fast_image_segmentation.py

# Same, but fp32 eager (no optimizations) — comparison baseline
python examples/baseline_image_segmentation.py

# Run both and print a timing/mask-count/coverage-IoU comparison table
python examples/compare_segmentation.py
```

Outputs (per-mask overlay images) are written to `examples/segmented/` and
`examples/segmented_baseline/` respectively (gitignored — regenerate them
by running the scripts).

### TensorRT pipeline (optional)

Requires the system TensorRT 10.3 apt package (`python3-libnvinfer` etc.)
and `trtexec` at `/usr/src/tensorrt/bin/trtexec`.

```bash
source .venv/bin/activate

python trt/export_encoder.py && ./trt/build_encoder_engine.sh   # ~4 min
python trt/export_decoder.py && ./trt/build_decoder_engine.sh   # ~10 min

python trt/segment_all_trt.py        # TRT-backed segment-everything
python trt/compare_trt.py            # vs. fast_image_segmentation.py (fp16+compile)
python trt/compare_baseline_trt.py   # vs. baseline_image_segmentation.py (fp32 eager)
```

## Testing report

All numbers below are from the Jetson AGX Orin device this work was done
on; see `NOTES.md` for full methodology, accuracy checks, and additional
findings (e.g. video predictor benchmarks, Jetson DVFS clock-ramp latency
caveat).

### Single point-prompt predict (`bench/bench_image_predictor.py`)

| Config | Full predict | FPS | Speedup |
|---|---|---|---|
| fp32 baseline | 223ms | 4.5 | 1.0x |
| fp16 + `torch.compile` (encoder) | 77ms | 13.0 | **2.9x** |

Accuracy: fp16+compile vs. fp32 eager masks have 99.96% IoU, 0.0002 score
delta (`bench/check_accuracy.py`) — effectively lossless.

### Segment-everything, baseline vs. fp16+`torch.compile` (`examples/compare_segmentation.py`)

`SAM2AutomaticMaskGenerator` over all 7 images in `examples/images/`, with
both the encoder **and** mask decoder compiled (the decoder is called
16+ times per image here, unlike the single-predict case, so compiling it
now pays off) and `points_per_batch` tuned to 128:

| Image | Masks (base/fast) | Time (base/fast) | Speedup | Coverage IoU |
|---|---|---|---|---|
| cars.jpg | 32/32 | 10.31s / 6.73s | 1.53x | 0.9992 |
| groceries.jpg | 57/57 | 6.56s / 2.11s | 3.11x | 0.9991 |
| rgb_20260701_123720_139762.png | 19/19 | 6.53s / 2.13s | 3.07x | 0.9999 |
| rgb_20260707_192013_959061.png | 25/25 | 7.67s / 3.06s | 2.51x | 0.9999 |
| rgb_20260707_220449_123550.png | 28/28 | 7.51s / 2.95s | 2.54x | 0.9999 |
| rgb_20260708_010855_674519.png | 25/25 | 7.50s / 2.98s | 2.52x | 0.9999 |
| truck.jpg | 36/36 | 9.67s / 4.83s | 2.00x | 1.0000 |

**2.25x overall** (55.8s → 24.8s total), identical mask counts everywhere,
coverage IoU ≥0.999. ("Coverage IoU" = IoU between the union of all masks
each run found — the right check when mask *count* itself can vary.)

### ONNX export + TensorRT

Encoder and decoder were exported to ONNX and built into fp16 TensorRT
engines (`trt/`). Two comparisons:

**TensorRT vs. baseline (fp32 eager)** — `trt/compare_baseline_trt.py`:

| Image | Masks (base/trt) | Time (base/trt) | Speedup | Coverage IoU |
|---|---|---|---|---|
| cars.jpg | 32/33 | 10.33s / 7.21s | 1.43x | 0.9998 |
| groceries.jpg | 57/57 | 6.46s / 1.93s | 3.36x | 0.9995 |
| rgb_20260701_123720_139762.png | 19/19 | 6.44s / 1.90s | 3.40x | 0.9991 |
| rgb_20260707_192013_959061.png | 25/25 | 7.46s / 2.75s | 2.71x | 0.9995 |
| rgb_20260707_220449_123550.png | 28/27 | 7.35s / 2.65s | 2.77x | 0.9997 |
| rgb_20260708_010855_674519.png | 25/25 | 7.35s / 2.67s | 2.75x | 0.9999 |
| truck.jpg | 36/36 | 9.32s / 4.48s | 2.08x | 0.9983 |

**2.32x overall** (54.7s → 23.6s) — TensorRT is a clear win over unoptimized
fp32 eager.

**TensorRT vs. fp16+`torch.compile`** — `trt/compare_trt.py`: only **1.03x
overall** (26.1s → 25.3s), a wash, despite the TRT decoder engine being a
genuine 2-3x faster per-kernel than the compiled PyTorch decoder (139.7ms
vs. an estimated 340-530ms for a 128-point batch). Root cause: once both
neural-net passes are fast, `SAM2AutomaticMaskGenerator`'s Python-side
postprocessing (stability score, box NMS, RLE encoding — unchanged between
both pipelines) dominates wall time and caps the achievable end-to-end win.

**Conclusion**: TensorRT is not currently worth the extra engineering
(ONNX export patches, dynamic-shape engine builds, a custom TRT runtime)
over the already-shipped fp16+`torch.compile` pipeline — it would only pay
off after also optimizing the postprocessing pipeline. Full detail,
including the encoder finding (TRT's default build is actually *slower*
than the compiled PyTorch encoder on this device) and the export bugs
worked around, in `NOTES.md` §6.

## Repository layout

- `bench/` — dtype/compile sweep benchmarks and accuracy checks for the
  single point-prompt image predictor, plus video predictor FPS.
- `examples/` — segment-everything examples: fp16+compile
  (`fast_image_segmentation.py`), fp32 eager baseline
  (`baseline_image_segmentation.py`), and their comparison
  (`compare_segmentation.py`).
- `trt/` — ONNX export, TensorRT engine builds, a `SAM2ImagePredictor`
  subclass routing through the TRT engines, and comparison scripts against
  both baseline and the fp16+compile pipeline.
- `NOTES.md` — full session notes: environment setup gotchas, every
  benchmark, and the reasoning behind each optimization decision.
