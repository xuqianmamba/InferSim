"""Shared shape and timing helpers for DeepSeek-V4 kernel benchmarks."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Sequence


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def parse_int_list(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0 for item in result):
        raise ValueError(
            f"expected a comma-separated list of positive integers: {value!r}"
        )
    return result


def load_dsv4_shape(config_path: str | Path, attention_tp_size: int) -> dict[str, int]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    global_heads = int(config["num_attention_heads"])
    if global_heads % attention_tp_size:
        raise ValueError(
            f"num_attention_heads={global_heads} is not divisible by "
            f"attention_tp_size={attention_tp_size}"
        )

    # DeepSeek-V4 stores the complete Q/K head width in head_dim. It already
    # includes qk_rope_head_dim (448 NoPE + 64 RoPE = 512 for V4-Pro).
    head_dim = int(config["head_dim"])
    rope_dim = int(config["qk_rope_head_dim"])
    if head_dim != 512:
        raise ValueError(
            f"the current FlashMLA DSV4 kernel requires head_dim=512, got {head_dim}"
        )
    if rope_dim >= head_dim:
        raise ValueError(f"invalid qk_rope_head_dim={rope_dim} for head_dim={head_dim}")

    return {
        "global_heads": global_heads,
        "local_heads": global_heads // attention_tp_size,
        "head_dim": head_dim,
        "value_dim": int(config.get("v_head_dim", head_dim)),
        "swa_window": int(config.get("sliding_window", 128)),
        "index_heads": int(config.get("index_n_heads", 64)),
        "index_dim": int(config.get("index_head_dim", 128)),
        "index_topk": int(config.get("index_topk", 1024)),
        "page_size": 256,
    }


def compressed_length(kv_len: int, ratio: int) -> int:
    # This mirrors DSV4AttnMetadata: compressed lengths use floor division,
    # clamped to one token for short requests.
    return max(kv_len // ratio, 1)


def physical_page_size(ratio: int) -> int:
    if ratio not in (4, 128):
        raise ValueError(f"unsupported DSV4 compression ratio: {ratio}")
    return 256 // ratio


def attended_lengths(
    kv_len: int, ratio: int, swa_window: int, index_topk: int
) -> tuple[int, int]:
    swa = min(kv_len, swa_window)
    compressed = compressed_length(kv_len, ratio)
    extra = min(index_topk, compressed) if ratio == 4 else compressed
    return swa, extra


def aligned_topk(length: int, alignment: int = 64) -> int:
    return ceil_div(length, alignment) * alignment


def write_csv(
    path: str | Path, fieldnames: Sequence[str], rows: Iterable[dict]
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def attention_flops(
    batch_size: int, heads: int, head_dim: int, value_dim: int, attended: int
) -> int:
    return batch_size * heads * (2 * head_dim * attended + 2 * attended * value_dim)


def indexer_flops(batch_size: int, heads: int, dim: int, compressed_kv_len: int) -> int:
    return 2 * batch_size * heads * dim * compressed_kv_len


def validate_matrix_rows(
    rows: Sequence[dict], batch_sizes: Sequence[int], kv_lens: Sequence[int]
) -> None:
    expected = {(batch, kv_len) for batch in batch_sizes for kv_len in kv_lens}
    actual = {(int(row["batch_size"]), int(row["kv_len"])) for row in rows}
    if actual != expected:
        raise RuntimeError(
            f"benchmark matrix mismatch: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
