#!/usr/bin/env bash
# Builds trt/decoder_fp16.engine from trt/decoder.onnx (run export_decoder.py first).
# Point-batch dimension is dynamic (see export_decoder.py) -- profile spans
# the range actually used by examples/fast_image_segmentation.py
# (points_per_batch=128 default, up to the 256 tested in NOTES.md). Building
# this took ~10min on Jetson AGX Orin, notably slower than the encoder's
# ~4min fixed-shape build, since TensorRT searches tactics across the whole
# shape range rather than one fixed shape.
set -euo pipefail
cd "$(dirname "$0")/.."

/usr/src/tensorrt/bin/trtexec \
  --onnx=trt/decoder.onnx \
  --saveEngine=trt/decoder_fp16.engine \
  --fp16 \
  --memPoolSize=workspace:4096 \
  --minShapes=point_coords:1x1x2,point_labels:1x1 \
  --optShapes=point_coords:128x1x2,point_labels:128x1 \
  --maxShapes=point_coords:256x1x2,point_labels:256x1
