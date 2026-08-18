#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/DeepSeek-V4-Pro/config.json" >&2
  exit 2
fi

config_path=$1
common_args=(
  --config-path "$config_path"
  --device-type H20
  --gpu-memory-gb 141
  --world-size 8
  --tp-size 8
  --decode-bs 16
  --target-isl 40960
  --target-osl 1000
  --use-fp8-kv
  --decode-only
)

python3 main.py "${common_args[@]}"
