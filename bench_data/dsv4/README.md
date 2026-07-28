# DeepSeek-V4-Pro H20 TP8 measurements

These rows were collected for architecture fingerprint
`dc8a15770f7f8647` on one node with eight NVIDIA H20 GPUs and SGLang
0.5.15.

Each value is a per-base-decoder-layer CUDA-event median. For each timed
forward, latency is first reduced to the maximum across TP ranks, then the
median is taken across timed forwards. Communication is excluded and modeled
separately by InferSim.

The serving keys are:

- attention: `dsv4`, FP8 weights, `fp8_e4m3` KV cache and FP32 compression
  state;
- mHC: `fused_post_pre`;
- MoE: native packed-FP4 experts with `flashinfer_mxfp4`.

The exact measured shapes are prefill `B=1, Q=4096` and decode
`B=128, Q=1, past=5120`. InferSim intentionally rejects other fingerprints,
parallel configurations, precisions and shapes instead of silently
extrapolating these values.

Run the matching end-to-end simulation with:

```bash
python3 main.py \
  --config-path /path/to/DeepSeek-V4-Pro/config.json \
  --device-type H20 --world-size 8 --tp-size 8 --num-nodes 1 \
  --max-prefill-tokens 4096 --target-isl 4096 --target-osl 2048 \
  --decode-bs 128 --dsv4-decode-past-len 5120
```
