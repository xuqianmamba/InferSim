"""Run and validate all DeepSeek-V4 H20 decode microbenchmarks."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

LEGACY_MOE_FIELDS = (
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
)

MOE_FIELDS = LEGACY_MOE_FIELDS + (
    "total_latency_us",
    "total_mfu",
    "kernel_kind",
    "backend",
    "activation_dtype",
    "weight_dtype",
    "mfu_peak_tflops",
    "tp_size",
    "ep_size",
    "tune_max_num_tokens",
    "execution_mode",
    "active_experts",
    "max_tokens_per_expert",
    "routing_mode",
    "swiglu_limit",
)

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
    "groupedgemm-decode-dsv4-tp8dp1.csv": MOE_FIELDS,
}
MOE_FILE = "groupedgemm-decode-dsv4-tp8dp1.csv"


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def validate(output_dir: Path, attention_rows: int, moe_rows: int) -> None:
    for name, expected_fields in MATRIX_FILES.items():
        if name == MOE_FILE:
            continue
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
        if len(rows) != attention_rows:
            raise RuntimeError(
                f"{path} has {len(rows)} rows, expected {attention_rows}"
            )
        print(f"OK {path}: rows={len(rows)}")
    validate_moe(output_dir, moe_rows)


def validate_moe(output_dir: Path, expected_rows: int) -> None:
    path = output_dir / MOE_FILE
    expected_fields = MATRIX_FILES[MOE_FILE]
    if not path.is_file():
        raise RuntimeError(f"missing benchmark output: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if tuple(reader.fieldnames or ()) != expected_fields:
            raise RuntimeError(
                f"unexpected fields in {path}: {reader.fieldnames}; "
                f"expected {expected_fields}"
            )
    if len(rows) != expected_rows:
        raise RuntimeError(f"{path} has {len(rows)} rows, expected {expected_rows}")
    for row in rows:
        if row["kernel_kind"] != "cutlass_grouped_gemm_pair":
            raise RuntimeError(f"unexpected MoE timing semantics in {path}: {row}")
        if abs(
            float(row["total_latency_us"])
            - float(row["up_proj_us"])
            - float(row["down_proj_us"])
        ) > 0.002:
            raise RuntimeError(f"MoE grouped-GEMM total is inconsistent: {row}")
    print(f"OK {path}: rows={len(rows)}")


def replace_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def install_moe_results(output_dir: Path, repository: Path) -> None:
    """Install only the measured TP8DP1 batch rows, preserving all others."""
    source = output_dir / MOE_FILE
    target = repository / "bench_data" / "grouped_gemm" / "decode" / "h20" / "data.csv"
    with source.open(newline="", encoding="utf-8") as handle:
        new_reader = csv.DictReader(handle)
        new_rows = list(new_reader)
        fields = tuple(new_reader.fieldnames or ())
    expected = MATRIX_FILES[source.name]
    if fields != expected:
        raise RuntimeError(f"unexpected MoE fields: {fields}; expected {expected}")
    if not new_rows:
        raise RuntimeError(f"no MoE rows to install: {source}")

    # Include batch_size_per_gpu in the replacement key.  A BS16-only refresh
    # must not delete valid BS24/26/32 rows for the same model shape.
    key_fields = LEGACY_MOE_FIELDS[:7]
    replace_keys = {
        tuple(row[field] for field in key_fields) for row in new_rows
    }
    kept_rows = []
    old_fields = ()
    if target.is_file():
        backup = output_dir / "grouped_gemm-decode-h20-data.before.csv"
        shutil.copy2(target, backup)
        print(f"BACKUP {target} -> {backup}")
        with target.open(newline="", encoding="utf-8") as handle:
            old_reader = csv.DictReader(handle)
            old_fields = tuple(old_reader.fieldnames or ())
            missing_legacy = set(LEGACY_MOE_FIELDS) - set(old_fields)
            if missing_legacy:
                raise RuntimeError(
                    f"existing MoE table lacks legacy fields {missing_legacy}: {target}"
                )
            for row in old_reader:
                key = tuple(row[field] for field in key_fields)
                if key not in replace_keys:
                    kept_rows.append(row)

    # New optional columns are appended without breaking historical rows.
    output_fields = fields + tuple(field for field in old_fields if field not in fields)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(kept_rows + new_rows)
    os.replace(temporary, target)
    print(f"INSTALLED DSV4 TP8DP1 grouped-GEMM data: {target}")


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
    install_moe_results(output_dir, repository)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-sizes", default="16,24,26,32")
    parser.add_argument("--kv-lens", default="16384,32768,40960,65536")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--execution-mode",
        choices=("eager",),
        default="eager",
        help="Eager single-kernel profiling mode for the MXFP4 MoE benchmark.",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Overwrite the repository's validated H20 DSV4 lookup data.",
    )
    parser.add_argument(
        "--moe-only",
        action="store_true",
        help="Run, validate, and optionally install only the routed-MoE lookup.",
    )
    args = parser.parse_args()
    if args.install and args.execution_mode != "eager":
        raise SystemExit("only eager single-kernel MoE results may be installed")

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
    if not args.moe_only:
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
            "--execution-mode",
            args.execution_mode,
            "--output-dir",
            str(output_dir),
        ]
    )
    batch_count = len([item for item in args.batch_sizes.split(",") if item.strip()])
    if args.moe_only:
        validate_moe(output_dir, batch_count)
        if args.install:
            install_moe_results(output_dir, script_dir.parent)
    else:
        kv_count = len([item for item in args.kv_lens.split(",") if item.strip()])
        validate(output_dir, batch_count * kv_count, batch_count)
        if args.install:
            install_results(output_dir, script_dir.parent)
    print("RUN_COMPLETE")


if __name__ == "__main__":
    main()
