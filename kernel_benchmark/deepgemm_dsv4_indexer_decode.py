"""Benchmark the DeepSeek-V4 C4 decode indexer (logits and top-k).

The indexer is replicated rather than attention-TP sharded: DSV4-Pro always
uses 64 index heads of width 128. Logical full-context lengths are converted
to C4 compressed lengths before invoking the same DeepGEMM and SGLang top-k
kernels used by serving.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from dsv4_bench_utils import (
    ceil_div,
    compressed_length,
    indexer_flops,
    load_dsv4_shape,
    parse_int_list,
    validate_matrix_rows,
    write_csv,
)


def kv_cache_cast_to_fp8(source: torch.Tensor) -> torch.Tensor:
    blocks, block_size, heads, dim = source.shape
    assert heads == 1
    amax = source.abs().float().amax(dim=3, keepdim=True).clamp_min_(1e-4)
    scale = amax / 448.0
    values = (source / scale).to(torch.float8_e4m3fn)
    packed = torch.empty(
        (blocks, block_size * (dim + 4)), dtype=torch.uint8, device=source.device
    )
    packed[:, : block_size * dim] = values.reshape(blocks, -1).view(torch.uint8)
    packed[:, block_size * dim :] = scale.reshape(blocks, -1).view(torch.uint8)
    return packed.view(blocks, block_size, 1, dim + 4)


def median_cuda_us(function, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(repeats):
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return sorted(samples)[len(samples) // 2]


def benchmark_point(
    batch: int,
    logical_kv_len: int,
    shape: dict[str, int],
    warmup: int,
    repeats: int,
    fp8_peak_tflops: float,
):
    import deep_gemm
    from sglang.kernels.ops.attention.dsv4 import topk_transform_512

    c4_len = compressed_length(logical_kv_len, 4)
    block_size = 64
    blocks_per_request = ceil_div(c4_len, block_size)
    total_blocks = batch * blocks_per_request
    max_c4_len = blocks_per_request * block_size

    torch.manual_seed(5678 + batch + logical_kv_len)
    q = torch.randn(
        batch,
        1,
        shape["index_heads"],
        shape["index_dim"],
        dtype=torch.bfloat16,
        device="cuda",
    ).to(torch.float8_e4m3fn)
    source = torch.randn(
        total_blocks,
        block_size,
        1,
        shape["index_dim"],
        dtype=torch.bfloat16,
        device="cuda",
    )
    cache = kv_cache_cast_to_fp8(source)
    weights = torch.randn(
        batch, shape["index_heads"], dtype=torch.float32, device="cuda"
    )
    # SGLang keeps the logical C4 lengths one-dimensional for the top-k
    # transform, but DeepGEMM's paged-MQA scheduler/kernel requires a
    # (num_queries, next_n) layout. Decode has next_n=1.
    lengths = torch.full((batch,), c4_len, dtype=torch.int32, device="cuda")
    lengths_2d = lengths.unsqueeze(-1)
    page_table = torch.arange(total_blocks, dtype=torch.int32, device="cuda").view(
        batch, blocks_per_request
    )
    metadata = deep_gemm.get_paged_mqa_logits_metadata(
        lengths_2d, block_size, deep_gemm.get_num_sms()
    )

    def run_logits():
        return deep_gemm.fp8_paged_mqa_logits(
            q,
            cache,
            weights,
            lengths_2d,
            page_table,
            metadata,
            max_c4_len,
            clean_logits=False,
        )

    logits = run_logits()
    topk = min(shape["index_topk"], c4_len)
    # SGLang pads this last dimension to 64. DSV4-Pro's topk=1024 already is.
    output = torch.full((batch, topk), -1, dtype=torch.int32, device="cuda")

    def run_topk():
        topk_transform_512(logits, lengths, page_table, output, block_size)

    def run_combined():
        current = run_logits()
        topk_transform_512(current, lengths, page_table, output, block_size)

    logits_us = median_cuda_us(run_logits, warmup, repeats)
    topk_us = median_cuda_us(run_topk, warmup, repeats)
    total_us = median_cuda_us(run_combined, warmup, repeats)
    flops = indexer_flops(batch, shape["index_heads"], shape["index_dim"], c4_len)
    mfu = flops / (logits_us * 1e-6) / 1e12 / fp8_peak_tflops
    result = {
        "batch_size": batch,
        "kv_len": logical_kv_len,
        "compressed_kv_len": c4_len,
        "logits_latency_us": round(logits_us, 3),
        "topk_latency_us": round(topk_us, 3),
        "total_latency_us": round(total_us, 3),
        "logits_mfu": round(mfu, 3),
    }
    print(json.dumps(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--batch-sizes", default="16,24,26,32")
    parser.add_argument("--kv-lens", default="16384,32768,40960,65536")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--fp8-tflops", type=float, default=296.0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    batch_sizes = parse_int_list(args.batch_sizes)
    kv_lens = parse_int_list(args.kv_lens)
    # attention TP does not change the replicated indexer, but loading the
    # model config through one validator keeps all DSV4 dimensions checked.
    shape = load_dsv4_shape(args.config_path, attention_tp_size=2)
    if (shape["index_heads"], shape["index_dim"]) != (64, 128):
        raise SystemExit(
            f"expected DSV4 indexer shape (64, 128), got "
            f"({shape['index_heads']}, {shape['index_dim']})"
        )
    torch.set_default_device("cuda")
    rows = [
        benchmark_point(
            batch, kv_len, shape, args.warmup, args.repeats, args.fp8_tflops
        )
        for batch in batch_sizes
        for kv_len in kv_lens
    ]
    validate_matrix_rows(rows, batch_sizes, kv_lens)
    output_dir = Path(args.output_dir)
    detail_fields = (
        "batch_size",
        "kv_len",
        "compressed_kv_len",
        "logits_latency_us",
        "topk_latency_us",
        "total_latency_us",
        "logits_mfu",
    )
    detail_path = output_dir / "indexer-64-128-topk1024.csv"
    write_csv(detail_path, detail_fields, rows)

    # Keep a compatibility table matching InferSim's existing H800 logits
    # schema. s_kv is the physical C4 sequence length seen by DeepGEMM.
    compatibility = [
        {
            "batchsize": row["batch_size"],
            "next_n": 1,
            "s_kv": row["compressed_kv_len"],
            "latency_us": row["logits_latency_us"],
            "mfu": row["logits_mfu"],
        }
        for row in rows
    ]
    logits_path = output_dir / "logits-64-128.csv"
    write_csv(
        logits_path,
        ("batchsize", "next_n", "s_kv", "latency_us", "mfu"),
        compatibility,
    )
    print(f"Wrote {detail_path}")
    print(f"Wrote {logits_path}")


if __name__ == "__main__":
    main()
