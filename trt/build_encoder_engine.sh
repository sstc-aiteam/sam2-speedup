#!/usr/bin/env bash
# Builds trt/encoder_fp16.engine from trt/encoder.onnx (run export_encoder.py first).
# Fixed 1024x1024 input -> no dynamic shape profile needed.
set -euo pipefail
cd "$(dirname "$0")/.."

/usr/src/tensorrt/bin/trtexec \
  --onnx=trt/encoder.onnx \
  --saveEngine=trt/encoder_fp16.engine \
  --fp16 \
  --memPoolSize=workspace:4096
