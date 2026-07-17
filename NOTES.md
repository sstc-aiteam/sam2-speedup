# SAM2 (sam2_hiera_small) Speedup on Jetson AGX Orin — Session Notes

Device: Jetson AGX Orin 64GB, JetPack 6 / L4T 36.4.7, CUDA 12.6, cuDNN 9.3,
TensorRT 10.3, GPU compute capability 8.7 (Ampere). Disk was tight
throughout: 57G total, only ~5.7G free at the start.

## 1. Environment setup

- Created `.venv` (Python 3.10) for the project.
- Installed **PyTorch 2.9.1 + torchvision 0.24.1** from the Jetson wheel
  index (`pypi.jetson-ai-lab.io/jp6/cu126`).
- Upgraded `pip`/`setuptools` first (the stock 59.6.0 setuptools can't do
  editable (PEP 660) installs, which SAM2's `pip install -e .` needs).

### Jetson wheel packaging gaps found and fixed

These wheels aren't fully self-contained the way the standard x86 PyPI
`torch`/`triton` wheels are. All fixes are baked into `.venv/bin/activate`
(just `source` it — nothing to remember):

1. `import torch` failed with `libcudss.so.0: cannot open shared object
   file`. Torch 2.9's `libtorch_cuda.so` links directly against cuDSS,
   which JetPack doesn't ship. Fixed by `pip install nvidia-cudss-cu12`
   and adding its lib dir to `LD_LIBRARY_PATH` (torch's own preload logic
   doesn't search that path for cudss specifically).
   - Installing it naively also drags in a redundant `cuda-toolkit[cublas]`
     meta-package (~620MB of cuBLAS/nvrtc that duplicate what JetPack
     already provides at `/usr/local/cuda-12.6`) — uninstalled those
     afterwards to save disk.
2. `torch.compile` failed with `TritonMissing` — no Triton ships with the
   torch wheel on aarch64. Fixed by installing `triton==3.6.0` from the
   same Jetson index.
3. Triton's JIT compile then failed on a missing `cuda.h` — the Jetson
   triton wheel ships no `triton/backends/nvidia/include/` directory at
   all. Fixed via the `TRITON_CUDACRT_PATH` env var pointing at the system
   CUDA toolkit's include dir.
4. Then failed with `Cannot find ptxas` — the wheel's bundled
   `backends/nvidia/bin/` only has `ptxas-blackwell`, not plain `ptxas`.
   Fixed via `TRITON_PTXAS_PATH` pointing at the system CUDA toolkit's
   `ptxas`, plus adding `/usr/local/cuda-12.6/bin` to `PATH`.

Disk management: pip's HTTP cache and the redundant cuBLAS pull both
needed cleaning (`pip cache purge`, uninstalling unneeded packages) to
avoid running out of space mid-session. Final state: ~4G free.

## 2. SAM2 repo setup

- Cloned `facebookresearch/sam2` into `sam2/`, installed editable
  (`pip install -e . --no-build-isolation`, `SAM2_BUILD_CUDA=0` — skipped
  the optional CUDA post-processing extension to save build time/disk;
  it's optional and doesn't affect segmentation results in most cases).
- Downloaded the `sam2_hiera_small.pt` checkpoint (176MB) — the original
  SAM2 "small" model as named in the request (config:
  `configs/sam2/sam2_hiera_s.yaml`), not the newer SAM2.1 variant.
- Verified end-to-end correctness with a real point-prompt segmentation on
  the bundled `truck.jpg` before doing any perf work.

## 3. Baseline → optimized benchmarking (image predictor)

Tried, in order: fp32 eager → TF32 → bf16 autocast → fp16 autocast →
bf16/fp16 + `torch.compile` on the image encoder (the Hiera backbone,
which dominates cost). Full numbers and methodology in
[`bench/RESULTS.md`](bench/RESULTS.md); headline result:

| Config | Full predict | FPS | Speedup |
|---|---|---|---|
| fp32 baseline | 223ms | 4.5 | 1.0x |
| fp16 + torch.compile (encoder) | 77ms | 13.0 | **2.9x** |

Verified this isn't a quality tradeoff: fp16+compile vs fp32-eager masks
have 99.96% IoU and a 0.0002 score delta (`bench/check_accuracy.py`).

Also benchmarked the **video predictor** (bf16 eager: 9.18 FPS on the
200-frame bedroom clip). The upstream `vos_optimized=True` path
(whole-model `torch.compile`, `SAM2VideoPredictorVOS`) hit a PyTorch
2.9 inductor/CUDA-graphs bug in SAM2's `memory_attention.py` (buffer
aliasing across a `.transpose()`) — an upstream compatibility issue,
not pursued further. The encoder-only compile technique isn't affected
and is what's recommended.

## 4. Real-world latency caveat (Jetson DVFS)

While writing the example script, isolated single calls measured 100-300ms+
even after warmup — much higher than the 77ms tight-loop benchmark number.
Root cause, confirmed via `/sys/class/devfreq/*/cur_freq`: Jetson's GPU
clocks (730-998MHz observed) idle down between sparse calls and only ramp
to their max (1300MHz) under sustained load; a tight back-to-back
benchmark loop sustains that load, a single request in an interactive
server doesn't. Documented in `examples/fast_image_segmentation.py` along
with the standard fix: run `sudo jetson_clocks` once on the device to lock
clocks to max for consistent low-latency serving (not run automatically
here since it's a root, device-wide, persistent change).

## 5. Segment-everything (automatic mask generation)

Extended `examples/fast_image_segmentation.py` from a single point-prompted
`predict()` call to `SAM2AutomaticMaskGenerator` (decodes a
`points_per_side**2` grid of point prompts to find every object in an
image), run in batch over all images in `examples/images/`.

This changed the compile tradeoff from §3: `bench/RESULTS.md` found
compiling the mask decoder wasn't worth it for a single decode call, but
automatic mask generation calls the decoder `points_per_side**2 /
points_per_batch` times per image (16+ calls, not 1), so the amortized win
flips. Compiling the decoder too cut steady-state segment-everything latency
by ~35-45% (7.5s → 4.9s on `truck.jpg`), with mask counts verified identical
to the uncompiled decoder on every example image. Also tuned
`points_per_batch` 64 → 128 (fewer, bigger decode batches): 128 was the
largest batch size that stayed both faster and mask-exact — 256 was a bit
faster but flipped one image's mask count by 1 (fp16 kernel-fusion rounding
tipping a near-threshold mask), and 1024 triggered Jetson unified-memory
allocator errors (`NvMapMemAllocInternalTagged`).

Added `examples/baseline_image_segmentation.py` (same
`SAM2AutomaticMaskGenerator` usage, but fp32 eager — no autocast, no
`torch.compile`, no TF32) and `examples/compare_segmentation.py` to validate
the optimized path end-to-end against it, since automatic mask generation
produces a variable-count, unordered set of masks that a single-mask IoU
check (§3's `check_accuracy.py` approach) can't directly compare. Instead it
checks "coverage IoU": the IoU between the union of all masks each run
found, i.e. do both runs agree on which regions are foreground overall.
Result over all 7 images in `examples/images/`:

| Image | Masks (base/fast) | Time (base/fast) | Speedup | Coverage IoU |
|---|---|---|---|---|
| cars.jpg | 32/32 | 10.31s / 6.73s | 1.53x | 0.9992 |
| groceries.jpg | 57/57 | 6.56s / 2.11s | 3.11x | 0.9991 |
| rgb_20260701_123720_139762.png | 19/19 | 6.53s / 2.13s | 3.07x | 0.9999 |
| rgb_20260707_192013_959061.png | 25/25 | 7.67s / 3.06s | 2.51x | 0.9999 |
| rgb_20260707_220449_123550.png | 28/28 | 7.51s / 2.95s | 2.54x | 0.9999 |
| rgb_20260708_010855_674519.png | 25/25 | 7.50s / 2.98s | 2.52x | 0.9999 |
| truck.jpg | 36/36 | 9.67s / 4.83s | 2.00x | 1.0000 |

**2.25x overall speedup** (55.8s → 24.8s total segmentation time), identical
mask counts on every image, coverage IoU ≥0.999 everywhere.

## 6. ONNX export + TensorRT

Explored whether TensorRT beats the fp16 + `torch.compile` pipeline from
§5. Reused the system-wide TensorRT 10.3 apt package (`python3-libnvinfer`
etc., under `/usr/lib/python3.10/dist-packages`) via a `sys.path` insert in
`trt/trt_runner.py` rather than `pip install tensorrt` in the venv, since
that would duplicate an already-installed ~1GB+ wheel and disk was down to
~8.6G free.

**Encoder** (`trt/export_encoder.py`): exports cleanly to ONNX — fixed
1024x1024 input (`SAM2Transforms` always resizes to this, so no dynamic
shape needed), just `TracerWarning`s about window-padding branches that are
constant at this fixed shape. `trt/build_encoder_engine.sh` builds an fp16
engine in ~4min. Result: **the default-build TRT engine (38-41ms) is
slower** than the already-compiled PyTorch encoder (33ms) — Inductor's
autotuned kernels already beat TensorRT's default tactic selection for this
model on Orin's 16 SMs. Accuracy is fine either way (cosine similarity
0.999992-0.999998 vs fp32 eager). Tried `--builderOptimizationLevel=5` for a
fairer shot — didn't finish in 20 minutes, abandoned as impractical.

**Decoder** (`trt/export_decoder.py`): scoped to exactly the path
`SAM2AutomaticMaskGenerator` uses (point-only prompts, `multimask_output=
True`, `repeat_image=True`, no box/mask-refinement input) with a dynamic
point-batch axis (profile 1-256, built via `trt/build_decoder_engine.sh`,
~10min — slower than the encoder's fixed-shape build since TRT searches
tactics across the whole shape range). Export hit a real
`torch.jit.trace`-only bug: `mask_decoder.predict_masks()`'s
`torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)` throws a
CUDA/CPU device-mismatch error under tracing (works fine eager) — since dim
0 is always size 1 there, patched it to the equivalent `.expand()` via a
context-scoped monkeypatch in the export script only, no changes to the
vendored SAM2 source.

Result: the decoder engine is a genuine, large **per-kernel** win — 139.7ms
GPU compute for a 128-point batch vs. an estimated ~340-530ms for the
compiled PyTorch decoder at the same batch size (2-3x). But wired end-to-end
(`trt/segment_all_trt.py`, `trt/trt_image_predictor.py` — a
`SAM2ImagePredictor` subclass overriding just `set_image()`/`_predict()` so
all of `SAM2AutomaticMaskGenerator`'s NMS/stability-score/RLE orchestration
is reused unchanged) that speedup **doesn't show up in wall time**. Ran
`trt/compare_trt.py` twice over the 7 images in `examples/images/` to check
this wasn't a fluke — both runs land in the same place (second run shown):

| Image | Masks (fast/trt) | Time (fast/trt) | Speedup | Coverage IoU |
|---|---|---|---|---|
| cars.jpg | 32/33 | 7.54s / 7.59s | 0.99x | 0.9993 |
| groceries.jpg | 57/57 | 2.22s / 2.04s | 1.09x | 0.9994 |
| rgb_20260701_123720_139762.png | 19/19 | 2.15s / 1.98s | 1.09x | 0.9990 |
| rgb_20260707_192013_959061.png | 25/25 | 3.17s / 3.03s | 1.04x | 0.9995 |
| rgb_20260707_220449_123550.png | 28/27 | 3.03s / 2.88s | 1.05x | 0.9998 |
| rgb_20260708_010855_674519.png | 25/25 | 3.01s / 2.90s | 1.04x | 0.9999 |
| truck.jpg | 36/36 | 4.99s / 4.86s | 1.03x | 0.9983 |

**Segmentation total**: 26.1s (fast/compile) vs 25.3s (TRT) — **1.03x
overall** (first run: 25.7s vs 25.0s, also 1.03x). **Warmup**: 42.2s
(compile) vs 31.3s (TRT) — TRT's engine load is faster and more predictable
since there's no JIT trace involved, the one clear operational plus it has.
Mask counts match within ±1 per image (same class of fp16-rounding
threshold flips as §5's `points_per_batch=256` finding) and coverage IoU is
0.9983-0.9999 both runs — the 1.03x wash is a stable result, not
measurement noise.

That 1.03x wash is TRT vs. the *already-optimized* fp16+compile pipeline,
which shares the same postprocessing bottleneck. Against the real starting
point it looks very different: `trt/compare_baseline_trt.py` (same
`union_mask`/`run_all` pattern, `baseline_image_segmentation.py` vs.
`trt/segment_all_trt.py`) over the same 7 images:

| Image | Masks (base/trt) | Time (base/trt) | Speedup | Coverage IoU |
|---|---|---|---|---|
| cars.jpg | 32/33 | 10.33s / 7.21s | 1.43x | 0.9998 |
| groceries.jpg | 57/57 | 6.46s / 1.93s | 3.36x | 0.9995 |
| rgb_20260701_123720_139762.png | 19/19 | 6.44s / 1.90s | 3.40x | 0.9991 |
| rgb_20260707_192013_959061.png | 25/25 | 7.46s / 2.75s | 2.71x | 0.9995 |
| rgb_20260707_220449_123550.png | 28/27 | 7.35s / 2.65s | 2.77x | 0.9997 |
| rgb_20260708_010855_674519.png | 25/25 | 7.35s / 2.67s | 2.75x | 0.9999 |
| truck.jpg | 36/36 | 9.32s / 4.48s | 2.08x | 0.9983 |

**Segmentation total**: 54.7s (baseline) vs 23.6s (TRT) — **2.32x overall**.
**Warmup**: 66.1s (baseline) vs 28.3s (TRT). This roughly matches chaining
§5's baseline-vs-fast (2.25x) with the ~1.03x fast-vs-TRT ratio above —
consistent, not a new independent effect. So TensorRT *is* a clear win over
unoptimized fp32 eager; it just isn't an *additional* win once fp16+compile
is already in place, because both post-compile pipelines are bottlenecked
on the same unchanged Python postprocessing.

Root cause: once the neural-net calls got fast enough, `automatic_mask_
generator.py`'s per-batch Python-side postprocessing (stability score, box
NMS, RLE encoding — CPU/GPU-mixed, unchanged between both pipelines) became
the dominant cost, capping the achievable end-to-end win regardless of how
much faster the decoder kernel itself got. **Conclusion: for this
pipeline, TensorRT is not currently a win over the already-shipped
fp16+torch.compile approach** — it requires substantially more engineering
(ONNX export patches, dynamic-shape engine builds, a custom TRT runtime) for
a wash in throughput, and would only pay off after also optimizing the
postprocessing pipeline (e.g. vectorized/batched NMS+RLE, fewer per-batch
Python round-trips) so it stops being the bottleneck. One minor operational
plus: TRT's engine-load "warmup" is faster and more predictable than
`torch.compile`'s JIT trace (~31s vs ~42s, no compile-time variance).

## 7. Deliverables

- `bench/bench_image_predictor.py` — sweeps dtype (fp32/tf32/bf16/fp16) x
  compile on/off, reports encoder-only and full-predict timing.
- `bench/bench_video_predictor.py` — video propagate-in-video FPS,
  eager vs `vos_optimized`.
- `bench/check_accuracy.py` — mask IoU / score-delta sanity check between
  fp32 eager and the optimized config (single point-prompt case).
- `bench/RESULTS.md` — full results table + setup gotchas writeup.
- `examples/fast_image_segmentation.py` — runnable, documented example
  applying fp16 + compiled encoder/decoder to segment-everything
  (`SAM2AutomaticMaskGenerator`) over a directory of images, with a proper
  `warmup()` and per-mask overlay visualization.
- `examples/baseline_image_segmentation.py` — same segment-everything flow,
  fp32 eager with no optimizations, as a comparison baseline.
- `examples/compare_segmentation.py` — runs both over `examples/images/` and
  reports the timing/mask-count/coverage-IoU table above.
- `trt/export_encoder.py`, `trt/export_decoder.py` — ONNX export of the
  image encoder and prompt-encoder+mask-decoder, scoped to
  `SAM2AutomaticMaskGenerator`'s usage.
- `trt/build_encoder_engine.sh`, `trt/build_decoder_engine.sh` — `trtexec`
  fp16 engine builds (fixed-shape and dynamic point-batch profile,
  respectively).
- `trt/trt_runner.py` — minimal TensorRT engine runner binding torch CUDA
  tensors directly (no pycuda).
- `trt/trt_image_predictor.py` — `SAM2ImagePredictor` subclass routing
  encode/decode through the TRT engines.
- `trt/segment_all_trt.py`, `trt/compare_trt.py` — TRT-backed
  segment-everything driver and its comparison against §5's pipeline
  (the §6 results table above).
- `trt/compare_baseline_trt.py` — TRT-backed pipeline compared directly
  against `examples/baseline_image_segmentation.py` (fp32 eager), the
  2.32x-overall table above.

## Open items / not done

- Video predictor whole-model compile (`vos_optimized=True`) is blocked on
  the upstream inductor/CUDA-graphs bug described in §3.
- `sudo jetson_clocks` not applied (root + device-wide; left for the user
  to opt into for production serving).
- §6's TensorRT postprocessing bottleneck (NMS/stability-score/RLE) not
  optimized — would need to be addressed for TensorRT to actually beat
  §5's fp16+torch.compile pipeline end-to-end.
