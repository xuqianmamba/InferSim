# DeepSeek-V4 H20 decode kernel benchmark

This benchmark reproduces the decode shapes used by SGLang for
DeepSeek-V4-Pro. It is intended to provide measured kernel latency tables for
an InferSim optimization-upper-bound analysis.

## Shape mapping

- TP8 + DP4 attention DP gives an attention TP degree of `8 / 4 = 2`.
- DeepSeek-V4-Pro has 128 global Q heads, so each attention rank has 64 heads.
- The complete Q/K head dimension is 512: 448 NoPE + 64 RoPE. Do not add the
  RoPE dimension a second time.
- The value dimension is 512 and the KV cache is FP8.
- The C4 indexer is replicated and remains 64 heads x 128 dimensions.
- C4 attends to SWA128 plus top-k 1024 compressed tokens. C128 attends to
  SWA128 plus all available C128-compressed tokens.

The default matrix represents the observed TP8DP4 serving shapes:

- local batch size: `16,24,26,32`
- full logical KV length: `16384,32768,40960,65536`
- query length: 1 (decode)

## Outputs

- `attn-64-512-c4.csv`: combined SWA + C4 FlashMLA attention latency.
- `attn-64-512-c128.csv`: combined SWA + C128 FlashMLA attention latency.
- `logits-64-128.csv`: compatibility table for the DeepGEMM logits kernel.
- `indexer-64-128-topk1024.csv`: logits, top-k, and combined indexer latency.

The attention CSV `kv_len` is the original full-context length. The logits
compatibility CSV `s_kv` is the C4-compressed length actually seen by the
DeepGEMM kernel.

## Run on H20

Use the same Python environment that starts the tested SGLang 0.5.17 service;
it must provide matching `sglang`, `sglang-kernel`, `flashinfer`, and
`deep_gemm` packages.

```bash
PY=/home/logs/kaiying/pydeps/sglang-v0.5.17-pr31700/bin/python
REPO=/home/logs/kaiying/workspaces/InferSim-dsv4-h20-decode
MODEL=/home/logs/kaiying/models/DeepSeek-V4-Pro
OUT=/home/logs/kaiying/runs/dsv4_h20_decode_$(date +%y%m%d_%H%M%S)

"$PY" "$REPO/kernel_benchmark/run_dsv4_h20_decode_bench.py" \
  --config-path "$MODEL/config.json" \
  --output-dir "$OUT"
```

For a short smoke test, use one batch/context point and fewer repeats:

```bash
"$PY" "$REPO/kernel_benchmark/run_dsv4_h20_decode_bench.py" \
  --config-path "$MODEL/config.json" \
  --output-dir "$OUT" \
  --batch-sizes 24 \
  --kv-lens 40960 \
  --warmup 2 \
  --repeats 5
```

The runner exits nonzero if a kernel fails or an expected CSV/row is missing.
