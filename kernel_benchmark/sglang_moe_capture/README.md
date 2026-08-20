# Real SGLang MoE capture

`sitecustomize.py` wraps the production FlashInfer MXFP4 fused-MoE call in an
eager SGLang debug server. It collects one decode step (61 layers for
DeepSeek-V4-Pro) at the requested running batch size, then replays those exact
calls back-to-back inside an Nsys CUDA profiler range.

The replay keeps the real hidden states, top-k IDs and weights, checkpoint
weights/scales, TP/EP arguments, and tuning bucket. The Nsys trace should be
filtered to the two CUTLASS `GroupProblemShape` kernels; routing, SwiGLU, and
finalization kernels are not part of the grouped-GEMM lookup value.

This hook is for a server started with `--disable-cuda-graph`. The isolated
replay itself is bounded by `cudaProfilerStart/Stop`, so Nsys does not record
model prefill or ordinary decode execution.
