"""Benchmark the exact sparse-decode FlashMLA shape used by DeepSeek-V4.

For TP8+DP4 attention DP, attention is TP2 and the local Q-head count is 64.
Both the SWA cache and the compressed C4/C128 cache are passed to the same
``flash_mla_with_kvcache`` call used by SGLang's DSV4 backend.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from dsv4_bench_utils import (
    aligned_topk,
    attended_lengths,
    attention_flops,
    ceil_div,
    compressed_length,
    load_dsv4_shape,
    parse_int_list,
    physical_page_size,
    validate_matrix_rows,
    write_csv,
)


DSV4_NOPE_DIM = 448
DSV4_ROPE_DIM = 64
DSV4_SCALE_BYTES = 8  # Seven UE8M0 scales plus one padding byte.
DSV4_KV_BYTES_PER_TOKEN = DSV4_NOPE_DIM + DSV4_ROPE_DIM * 2 + DSV4_SCALE_BYTES
FLASHMLA_PAGE_ALIGNMENT = 576


def make_dsv4_cache(
    total_pages: int, block_size: int, head_dim: int
) -> torch.Tensor:
    """Build the exact packed DSV4 KV-cache view consumed by FlashMLA.

    DSV4 does not use FlashMLA's generic FP8 cache layout.  It stores 448
    NoPE values in FP8, 64 RoPE values in BF16, and eight scale/padding bytes
    per token.  Reuse SGLang's production quantizer and page writer so this
    standalone benchmark exercises the same 584-byte layout as serving.
    """
    from sglang.kernels.ops.attention.dsv4.index_buf_accessor import SetKAndS
    from sglang.kernels.ops.attention.dsv4.quant_k_cache import (
        quant_to_nope_fp8_rope_bf16_pack_triton,
    )

    assert head_dim == DSV4_NOPE_DIM + DSV4_ROPE_DIM == 512
    total_tokens = total_pages * block_size
    source = (
        torch.randn(total_tokens, head_dim, dtype=torch.bfloat16, device="cuda")
        / 10
    )
    packed = quant_to_nope_fp8_rope_bf16_pack_triton(source)

    page_bytes = block_size * DSV4_KV_BYTES_PER_TOKEN
    padded_page_bytes = ceil_div(page_bytes, FLASHMLA_PAGE_ALIGNMENT) * 576
    raw_cache = torch.zeros(
        total_pages,
        padded_page_bytes,
        dtype=torch.uint8,
        device="cuda",
    )
    locations = torch.arange(total_tokens, dtype=torch.int64, device="cuda")
    SetKAndS.execute(
        SimpleNamespace(page_size=block_size), raw_cache, locations, packed
    )

    # Match DeepseekV4AttnBackend: discard per-page tail padding, then expose
    # the four-dimensional byte view expected by sparse_decode_fwd.
    cache = raw_cache[:, :page_bytes].view(
        total_pages, block_size, 1, DSV4_KV_BYTES_PER_TOKEN
    )
    assert cache.shape[-1] == 584
    return cache


def make_cache_and_indices(
    batch: int, logical_len: int, block_size: int, head_dim: int
):
    pages_per_request = ceil_div(logical_len, block_size)
    total_pages = batch * pages_per_request
    cache = make_dsv4_cache(total_pages, block_size, head_dim)
    selected = logical_len
    padded = aligned_topk(selected)
    indices = torch.full((batch, 1, padded), -1, dtype=torch.int32, device="cuda")
    for request in range(batch):
        start = request * pages_per_request * block_size
        indices[request, 0, :selected] = torch.arange(
            start, start + selected, dtype=torch.int32, device="cuda"
        )
    lengths = torch.full((batch,), selected, dtype=torch.int32, device="cuda")
    return cache, indices, lengths


def benchmark_point(
    batch: int,
    kv_len: int,
    ratio: int,
    shape: dict[str, int],
    repeats: int,
    warmup: int,
    peak_tflops: float,
):
    from sgl_kernel.flash_mla import flash_mla_with_kvcache, get_mla_metadata

    torch.manual_seed(1234 + batch + kv_len + ratio)
    q = torch.randn(
        batch,
        1,
        shape["local_heads"],
        shape["head_dim"],
        dtype=torch.bfloat16,
        device="cuda",
    )
    attn_sink = torch.randn(shape["local_heads"], dtype=torch.float32, device="cuda")

    swa_len, extra_len = attended_lengths(
        kv_len, ratio, shape["swa_window"], shape["index_topk"]
    )
    swa_cache, swa_indices, swa_lengths = make_cache_and_indices(
        batch, swa_len, shape["swa_window"], shape["head_dim"]
    )

    compressed = compressed_length(kv_len, ratio)
    selected_extra = extra_len
    extra_page = physical_page_size(ratio)
    pages_per_request = ceil_div(compressed, extra_page)
    total_pages = batch * pages_per_request
    extra_cache = make_dsv4_cache(
        total_pages, extra_page, shape["head_dim"]
    )
    extra_indices = torch.full(
        (batch, 1, aligned_topk(selected_extra)), -1, dtype=torch.int32, device="cuda"
    )
    for request in range(batch):
        start = request * pages_per_request * extra_page
        # C4 selects top-k compressed tokens; sequential valid indices provide
        # the same kernel shape without adding indexer noise to this timing.
        extra_indices[request, 0, :selected_extra] = torch.arange(
            start, start + selected_extra, dtype=torch.int32, device="cuda"
        )
    extra_lengths = torch.full(
        (batch,), selected_extra, dtype=torch.int32, device="cuda"
    )
    metadata = get_mla_metadata()[0]

    kwargs = dict(
        q=q,
        k_cache=swa_cache,
        head_dim_v=shape["value_dim"],
        block_table=None,
        cache_seqlens=None,
        tile_scheduler_metadata=metadata,
        softmax_scale=shape["head_dim"] ** -0.5,
        is_fp8_kvcache=True,
        indices=swa_indices,
        topk_length=swa_lengths,
        attn_sink=attn_sink,
        extra_k_cache=extra_cache,
        extra_indices_in_kvcache=extra_indices,
        extra_topk_length=extra_lengths,
    )

    for _ in range(warmup):
        flash_mla_with_kvcache(**kwargs)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(repeats):
        start.record()
        flash_mla_with_kvcache(**kwargs)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    latency_us = sorted(samples)[len(samples) // 2]
    flops = attention_flops(
        batch,
        shape["local_heads"],
        shape["head_dim"],
        shape["value_dim"],
        swa_len + selected_extra,
    )
    achieved_tflops = flops / (latency_us * 1e-6) / 1e12
    print(
        json.dumps(
            {
                "ratio": ratio,
                "batch_size": batch,
                "kv_len": kv_len,
                "swa_tokens": swa_len,
                "extra_tokens": selected_extra,
                "latency_us": round(latency_us, 3),
                "mfu": round(achieved_tflops / peak_tflops, 3),
            }
        )
    )
    return {
        "dtype": "bf16",
        "kv_dtype": "fp8",
        "batch_size": batch,
        "kv_len": kv_len,
        "latency_us": round(latency_us, 3),
        "mfu": round(achieved_tflops / peak_tflops, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--attention-tp-size", type=int, default=2)
    parser.add_argument("--batch-sizes", default="16,24,26,32")
    parser.add_argument("--kv-lens", default="16384,32768,40960,65536")
    parser.add_argument("--ratios", default="4,128")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--bf16-tflops", type=float, default=148.0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    batch_sizes = parse_int_list(args.batch_sizes)
    kv_lens = parse_int_list(args.kv_lens)
    ratios = parse_int_list(args.ratios)
    if any(ratio not in (4, 128) for ratio in ratios):
        raise SystemExit("--ratios supports only 4 and 128")
    shape = load_dsv4_shape(args.config_path, args.attention_tp_size)
    if shape["local_heads"] not in (32, 64, 128):
        raise SystemExit(
            f"FlashMLA does not support local_heads={shape['local_heads']}"
        )
    print("DSV4 shape:", json.dumps(shape, sort_keys=True))
    torch.set_default_device("cuda")

    output_dir = Path(args.output_dir)
    fields = ("dtype", "kv_dtype", "batch_size", "kv_len", "latency_us", "mfu")
    for ratio in ratios:
        rows = [
            benchmark_point(
                batch, kv_len, ratio, shape, args.repeats, args.warmup, args.bf16_tflops
            )
            for batch in batch_sizes
            for kv_len in kv_lens
        ]
        validate_matrix_rows(rows, batch_sizes, kv_lens)
        path = (
            output_dir / f"attn-{shape['local_heads']}-{shape['head_dim']}-c{ratio}.csv"
        )
        write_csv(path, fields, rows)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
