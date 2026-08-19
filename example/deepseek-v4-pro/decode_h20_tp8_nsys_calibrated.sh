#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/DeepSeek-V4-Pro/config.json" >&2
  exit 2
fi

config_path=$1

# These are complete decoder-layer GPU critical-path spans measured with
# Nsight Systems at TP8DP1, batch size 16, and 40K context on H20-141GB.
# They are deliberately separate from the standalone kernel lookup tables.
python3 main.py \
  --config-path "$config_path" \
  --device-type H20 \
  --gpu-memory-gb 141 \
  --world-size 8 \
  --tp-size 8 \
  --decode-bs 16 \
  --target-isl 40960 \
  --target-osl 1000 \
  --use-fp8-kv \
  --decode-scheduler-overhead-ms 0 \
  --dsv4-c4-layer-latency-us 636.574 \
  --dsv4-c128-layer-latency-us 573.050 \
  --decode-only
