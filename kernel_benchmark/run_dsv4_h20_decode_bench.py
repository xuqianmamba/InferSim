"""Run and validate all DeepSeek-V4 H20 decode microbenchmarks."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
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
    "logits-64-128.csv": (
        "batchsize",
        "next_n",
        "s_kv",
        "latency_us",
        "mfu",
    ),
    "groupedgemm-decode-dsv4-tp8dp1.csv": (
        "num_experts",
        "num_gpus",
        "num_local_experts",
        "topk",
        "hidden_size",
        "intermediate_size",
        "batch_size_per_gpu",
        "tokens_per_expert",
        "up_proj_us",
        "up_mfu",
        "down_proj_us",
        "down_mfu",
    ),
}


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def validate(output_dir: Path, attention_rows: int, moe_rows: int) -> None:
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
        expected_rows = moe_rows if name.startswith("groupedgemm-") else attention_rows
        if len(rows) != expected_rows:
            raise RuntimeError(f"{path} has {len(rows)} rows, expected {expected_rows}")
        print(f"OK {path}: rows={len(rows)}")


def replace_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def install_results(output_dir: Path, repository: Path) -> None:
    """Replace DSV4 H20 lookup data only after the full run validates."""
    dsa_dir = repository / "bench_data" / "dsa" / "decode" / "h20"
    for name in (
        "attn-64-512-c4.csv",
        "attn-64-512-c128.csv",
        "indexer-64-128-topk1024.csv",
        "logits-64-128.csv",
    ):
        replace_file(output_dir / name, dsa_dir / name)

    source = output_dir / "groupedgemm-decode-dsv4-tp8dp1.csv"
    target = repository / "bench_data" / "grouped_gemm" / "decode" / "h20" / "data.csv"
    with source.open(newline="", encoding="utf-8") as handle:
        new_reader = csv.DictReader(handle)
        new_rows = list(new_reader)
        fields = tuple(new_reader.fieldnames or ())
    expected = MATRIX_FILES[source.name]
    if fields != expected:
        raise RuntimeError(f"unexpected MoE fields: {fields}; expected {expected}")

    replace_key = tuple(new_rows[0][field] for field in fields[:6])
    kept_rows = []
    if target.is_file():
        with target.open(newline="", encoding="utf-8") as handle:
            old_reader = csv.DictReader(handle)
            if tuple(old_reader.fieldnames or ()) != fields:
                raise RuntimeError(f"unexpected existing MoE fields in {target}")
            for row in old_reader:
                key = tuple(row[field] for field in fields[:6])
                if key != replace_key:
                    kept_rows.append(row)

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(kept_rows + new_rows)
    os.replace(temporary, target)
    print(f"INSTALLED DSV4 TP8DP1 lookup data under {repository / 'bench_data'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-sizes", default="16,24,26,32")
    parser.add_argument("--kv-lens", default="16384,32768,40960,65536")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--install",
        action="store_true",
        help="Overwrite the repository's validated H20 DSV4 lookup data.",
    )
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
            "8",
            "--kernel-query-heads",
            "64",
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
    run(
        [
            sys.executable,
            str(script_dir / "flashinfer_dsv4_mxfp4_moe_decode.py"),
            "--world-size",
            "8",
            "--tp-size",
            "8",
            "--config-path",
            str(Path(args.config_path).resolve()),
            "--batch-sizes",
            args.batch_sizes,
            "--warmup",
            str(args.warmup),
            "--repeats",
            str(args.repeats),
            "--output-dir",
            str(output_dir),
        ]
    )
    batch_count = len([item for item in args.batch_sizes.split(",") if item.strip()])
    kv_count = len([item for item in args.kv_lens.split(",") if item.strip()])
    validate(output_dir, batch_count * kv_count, batch_count)
    if args.install:
        install_results(output_dir, script_dir.parent)
    print("RUN_COMPLETE")


if __name__ == "__main__":
    main()
