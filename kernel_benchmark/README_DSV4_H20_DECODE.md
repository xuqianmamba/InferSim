# DeepSeek-V4 H20 decode kernel benchmark

This benchmark reproduces the TP8DP1 decode shapes used by SGLang for
DeepSeek-V4-Pro. It measures attention, indexer, and routed-MoE kernels for an
InferSim optimization-upper-bound analysis.

## Shape mapping

- TP8DP1 gives an attention TP degree of 8 and 16 useful local Q heads.
- SGLang pads 16 useful heads to FlashMLA's 64-head kernel shape. The measured
  latency therefore uses 64 heads; InferSim scales MFU back to 16 useful heads.
- The complete Q/K head dimension is 512: 448 NoPE + 64 RoPE. Do not add the
  RoPE dimension a second time.
- The value dimension is 512 and the KV cache is FP8.
- The C4 indexer is replicated and remains 64 heads x 128 dimensions.
- C4 attends to SWA128 plus top-k 1024 compressed tokens. C128 attends to
  SWA128 plus all available C128-compressed tokens.

The default matrix covers the observed TP8DP1 serving shapes:

- local batch size: `16,24,26,32`
- full logical KV length: `16384,32768,40960,65536`
- query length: 1 (decode)

## Outputs

- `attn-64-512-c4.csv`: combined SWA + C4 FlashMLA attention latency.
- `attn-64-512-c128.csv`: combined SWA + C128 FlashMLA attention latency.
- `logits-64-128.csv`: compatibility table for the DeepGEMM logits kernel.
- `indexer-64-128-topk1024.csv`: logits, top-k, and combined indexer latency.
- `groupedgemm-decode-dsv4-tp8dp1.csv`: the two production CUTLASS grouped
  GEMMs launched by `flashinfer_mxfp4` (384 experts, top-k 6, hidden 7168,
  TP-sharded intermediate size 384).

The MoE benchmark follows SGLang v0.5.17's H20/SM90 W4A16 production call:

- BF16 activation and output, MXFP4 E2M1 packed expert weights, and E8M0
  group scales;
- the actual TP/EP topology (`tp_size=8`, `ep_size=1` for TP8DP1);
- `tune_max_num_tokens=next_power_of_2(batch_size)` so FlashInfer selects the
  same token-bucket tactic as serving;
- eager launches, so the two grouped GEMM CUDA kernels remain independently
  visible to the profiler. CUDA Graph replay is deliberately not used for this
  single-kernel lookup table.

The benchmark profiles raw CUDA events and requires exactly two matching
CUTLASS `GroupProblemShape` kernels per eager call. It records the first as
`up_proj_us` (fused gate/up), the second as `down_proj_us`, and stores their sum
as `total_latency_us`. Routing, sorting, SwiGLU activation, finalization, and
kernel gaps are deliberately excluded. `total_mfu` is a diagnostic
useful-FLOP normalization, not a claim about the native MXFP4 hardware peak.

Routing IDs in this microbenchmark use a deterministic synthetic distribution.
The CSV records its active-expert and maximum-expert-load statistics. If an
Nsys production trace shows materially different routing skew, replaying
captured routing tensors is the next calibration step; do not compensate by
manually fitting the reported latency.

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
  --output-dir "$OUT" \
  --execution-mode eager \
  --install
```

For a short smoke test, use one batch/context point and fewer repeats:

```bash
"$PY" "$REPO/kernel_benchmark/run_dsv4_h20_decode_bench.py" \
  --config-path "$MODEL/config.json" \
  --output-dir "$OUT" \
  --batch-sizes 24 \
  --kv-lens 40960 \
  --warmup 2 \
  --repeats 5 \
  --execution-mode eager
```

The runner exits nonzero if a kernel fails or an expected CSV/row is missing.
With `--install`, validation finishes before any lookup data is replaced. The
runner overwrites H20 DSA files and replaces only matching DSV4 TP8DP1 rows in
`bench_data/grouped_gemm/decode/h20/data.csv`; unrelated rows are preserved.
Only eager single-kernel results may be installed. The benchmark still passes
the production TP/EP topology and tuning bucket to FlashInfer; graph launch
overhead and other fused-MoE kernels do not belong in this lookup table.

To refresh only the MoE grouped-GEMM table, without rerunning attention and
indexer benchmarks:

```bash
"$PY" "$REPO/kernel_benchmark/run_dsv4_h20_decode_bench.py" \
  --config-path "$MODEL/config.json" \
  --output-dir "$OUT" \
  --batch-sizes 16,24,26,32 \
  --warmup 10 \
  --repeats 50 \
  --execution-mode eager \
  --moe-only \
  --install
```

MoE installation keys include `batch_size_per_gpu`, so a partial BS16 refresh
does not delete BS24/26/32. Before replacement, the prior H20 grouped-GEMM
table is copied to `grouped_gemm-decode-h20-data.before.csv` in the output
directory.
