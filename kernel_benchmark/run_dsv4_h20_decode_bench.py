"""Run and validate all DeepSeek-V4 H20 decode microbenchmarks."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

MATRIX_FILES = {
    "attn-64-512-c4.csv": (
        "dtype",
        "kv_dtype",
        "batch_size",
        "kv_len",
        "latency_us",
        "mfu",
    ),
    "attn-64-512-c128.csv": (
        "dtype",
        "kv_dtype",
        "batch_size",
        "kv_len",
        "latency_us",
        "mfu",
    ),
    "indexer-64-128-topk1024.csv": (
        "batch_size",
        "kv_len",
        "compressed_kv_len",
        "logits_latency_us",
        "topk_latency_us",
        "total_latency_us",
        "logits_mfu",
    ),
}


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def validate(output_dir: Path, expected_rows: int) -> None:
    for name, expected_fields in MATRIX_FILES.items():
        path = output_dir / name
        if not path.is_file():
            raise RuntimeError(f"missing benchmark output: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            if tuple(reader.fieldnames or ()) != expected_fields:
                raise RuntimeError(
                    f"unexpected fields in {path}: {reader.fieldnames}; expected {expected_fields}"
                )
        if len(rows) != expected_rows:
            raise RuntimeError(f"{path} has {len(rows)} rows, expected {expected_rows}")
        print(f"OK {path}: rows={len(rows)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-sizes", default="16,24,26,32")
    parser.add_argument("--kv-lens", default="16384,32768,40960,65536")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    common = [
        "--config-path",
        str(Path(args.config_path).resolve()),
        "--batch-sizes",
        args.batch_sizes,
        "--kv-lens",
        args.kv_lens,
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--output-dir",
        str(output_dir),
    ]
    run(
        [
            sys.executable,
            str(script_dir / "flashmla_dsv4_sparse_decode.py"),
            "--attention-tp-size",
            "2",
            *common,
        ]
    )
    run(
        [
            sys.executable,
            str(script_dir / "deepgemm_dsv4_indexer_decode.py"),
            *common,
        ]
    )
    batch_count = len([item for item in args.batch_sizes.split(",") if item.strip()])
    kv_count = len([item for item in args.kv_lens.split(",") if item.strip()])
    validate(output_dir, batch_count * kv_count)
    print("RUN_COMPLETE")


if __name__ == "__main__":
    main()
