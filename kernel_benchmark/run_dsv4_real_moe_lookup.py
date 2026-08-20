"""Build the DSV4 H20 routed-MoE lookup from real SGLang decode batches.

For every requested batch size this driver starts an eager TP8DP1 SGLang
server under Nsys, submits exactly that many 40K -> 1K requests, lets the
``sglang_moe_capture`` hook replay one real 61-layer decode step, imports the
trace, and keeps only the two CUTLASS GroupProblemShape kernels per layer.

The generated CSV therefore contains production routing, checkpoint weights,
FlashInfer arguments, and tactic/workspace state.  Routing, SwiGLU, finalize,
and inter-kernel gaps are deliberately excluded from the lookup value.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import signal
import socket
import sqlite3
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Iterable

from run_dsv4_h20_decode_bench import MOE_FIELDS, MOE_FILE, install_moe_results


DEFAULT_NSYS = Path(
    "/opt/nvidia/nsight-compute/2025.3.0/host/target-linux-x64/nsys"
)
DEFAULT_IMPORTER = Path(
    "/opt/nvidia/nsight-compute/2025.3.0/host/"
    "linux-desktop-glibc_2_11_3-x64/QdstrmImporter"
)
DEFAULT_IMPORT_LIBS = Path("/home/logs/kaiying/tools/nsys-2025-import-libs")


def parse_int_list(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError(
            f"expected comma-separated positive integers, got {value!r}"
        )
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError(f"duplicate batch size in {value!r}")
    return result


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, check=True, **kwargs)


def wait_ready(port: int, timeout: float, process: subprocess.Popen) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/v1/models"
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"Nsys/SGLang exited before readiness (exit={return_code})"
            )
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    print(f"Server ready: {url}", flush=True)
                    return
        except Exception as error:  # readiness polling intentionally broad
            last_error = error
        time.sleep(2)
    raise TimeoutError(f"server did not become ready within {timeout}s: {last_error}")


def terminate_process_group(process: subprocess.Popen, timeout: float = 30) -> None:
    # Nsys may exit before every process it launched.  The process-group
    # leader can therefore be gone while SGLang workers still own the port.
    # Always signal the group created by start_new_session=True.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    if process.poll() is not None:
        return
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def ensure_port_available(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.settimeout(1)
        if client.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(
                f"port {port} is already serving a process; stop it before this run"
            )


def wait_port_available(port: int, timeout: float = 60) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            ensure_port_available(port)
            return
        except RuntimeError:
            time.sleep(1)
    ensure_port_available(port)


def ensure_report(
    point_dir: Path,
    prefix: Path,
    importer: Path,
    import_lib_dir: Path | None,
) -> Path:
    report = prefix.with_suffix(".nsys-rep")
    if report.is_file() and report.stat().st_size:
        return report
    qdstrm = prefix.with_suffix(".qdstrm")
    if not qdstrm.is_file() or not qdstrm.stat().st_size:
        raise RuntimeError(f"neither Nsys report nor Qdstrm was generated: {prefix}")
    environment = os.environ.copy()
    if import_lib_dir is not None:
        old = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = str(import_lib_dir) + (
            os.pathsep + old if old else ""
        )
    print(f"Importing {qdstrm}", flush=True)
    with (point_dir / "qdstrm_import.log").open("a", encoding="utf-8") as log:
        run(
            [str(importer), "-i", str(qdstrm), "-o", str(report), "-f"],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
        )
    if not report.is_file() or not report.stat().st_size:
        raise RuntimeError(f"Qdstrm import did not generate report: {report}")
    return report


def export_sqlite(nsys: Path, report: Path) -> Path:
    sqlite_path = report.with_suffix(".sqlite")
    if sqlite_path.is_file():
        sqlite_path.unlink()
    run(
        [
            str(nsys),
            "export",
            "--type=sqlite",
            "--force-overwrite=true",
            "--output",
            str(sqlite_path),
            str(report),
        ]
    )
    if not sqlite_path.is_file() or not sqlite_path.stat().st_size:
        raise RuntimeError(f"Nsys export did not generate SQLite: {sqlite_path}")
    return sqlite_path


def grouped_gemm_durations(sqlite_path: Path) -> list[float]:
    connection = sqlite3.connect(sqlite_path)
    try:
        rows = connection.execute(
            """
            SELECT k.start, k.end, s.value
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
            JOIN StringIds AS s ON s.id = k.demangledName
            WHERE lower(s.value) LIKE '%cutlass::device_kernel%'
              AND lower(s.value) LIKE '%groupproblemshape%'
            ORDER BY k.start
            """
        ).fetchall()
    finally:
        connection.close()
    return [(int(end) - int(start)) / 1000.0 for start, end, _ in rows]


def summarize_pairs(
    durations_us: list[float], expected_pairs: int
) -> dict[str, float | int]:
    expected_kernels = expected_pairs * 2
    if len(durations_us) != expected_kernels:
        raise RuntimeError(
            "grouped-GEMM kernel count mismatch: "
            f"expected={expected_kernels}, actual={len(durations_us)}"
        )
    gate_up = durations_us[0::2]
    down = durations_us[1::2]
    up_mean = statistics.mean(gate_up)
    down_mean = statistics.mean(down)
    return {
        "pairs": expected_pairs,
        "up_mean_us": up_mean,
        "up_median_us": statistics.median(gate_up),
        "down_mean_us": down_mean,
        "down_median_us": statistics.median(down),
        "total_mean_us": up_mean + down_mean,
        "total_median_us": statistics.median(gate_up)
        + statistics.median(down),
    }


def useful_mfu(flops: int, latency_us: float, peak_tflops: float) -> float:
    return flops / (latency_us * 1e-6) / (peak_tflops * 1e12)


def build_lookup_row(
    batch_size: int,
    metadata: dict,
    summary: dict[str, float | int],
    *,
    peak_tflops: float,
    hidden_size: int,
    local_intermediate_size: int,
    num_experts: int,
    topk: int,
    tp_size: int,
) -> dict[str, str | int | float]:
    up_us = float(summary["up_mean_us"])
    down_us = float(summary["down_mean_us"])
    total_us = up_us + down_us
    up_flops = 4 * batch_size * topk * hidden_size * local_intermediate_size
    down_flops = 2 * batch_size * topk * hidden_size * local_intermediate_size
    active = metadata.get("active_experts_per_call", [])
    return {
        "num_experts": num_experts,
        "num_gpus": tp_size,
        "num_local_experts": num_experts,
        "topk": topk,
        "hidden_size": hidden_size,
        "intermediate_size": local_intermediate_size,
        "batch_size_per_gpu": batch_size,
        "tokens_per_expert": round(batch_size * topk / num_experts),
        "up_proj_us": f"{up_us:.3f}",
        "up_mfu": f"{useful_mfu(up_flops, up_us, peak_tflops):.6f}",
        "down_proj_us": f"{down_us:.3f}",
        "down_mfu": f"{useful_mfu(down_flops, down_us, peak_tflops):.6f}",
        "total_latency_us": f"{total_us:.3f}",
        "total_mfu": f"{useful_mfu(up_flops + down_flops, total_us, peak_tflops):.6f}",
        "kernel_kind": "cutlass_grouped_gemm_pair_production_nsys_graph",
        "backend": "flashinfer_mxfp4_sm90",
        "activation_dtype": "bf16",
        "weight_dtype": "mxfp4_e2m1",
        "mfu_peak_tflops": f"{peak_tflops:.1f}",
        "tp_size": tp_size,
        "ep_size": 1,
        "tune_max_num_tokens": 1 << (batch_size - 1).bit_length(),
        "execution_mode": "graph",
        "active_experts": f"{statistics.mean(active):.3f}" if active else "",
        "max_tokens_per_expert": metadata.get("max_tokens_per_expert", ""),
        "routing_mode": "production_sglang_dsv4",
        "swiglu_limit": "10.0",
    }


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MOE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def capture_point(args: argparse.Namespace, batch_size: int, point_dir: Path) -> dict:
    point_dir.mkdir(parents=True, exist_ok=True)
    prefix = point_dir / f"dsv4_real_moe_bs{batch_size}"
    server_log_path = point_dir / "server.log"
    bench_log_path = point_dir / "bench.log"
    capture_metadata = point_dir / "capture_complete.json"

    if args.resume and capture_metadata.is_file():
        metadata = json.loads(capture_metadata.read_text(encoding="utf-8"))
        if metadata.get("status") == "complete":
            print(f"Reusing completed real capture for BS={batch_size}", flush=True)
            report = ensure_report(
                point_dir, prefix, args.importer, args.import_lib_dir
            )
            sqlite_path = report.with_suffix(".sqlite")
            if not sqlite_path.is_file() or not sqlite_path.stat().st_size:
                sqlite_path = export_sqlite(args.nsys, report)
            expected_pairs = int(metadata["expected_grouped_gemm_pairs"])
            summary = summarize_pairs(
                grouped_gemm_durations(sqlite_path), expected_pairs
            )
            return build_lookup_row(
                batch_size,
                metadata,
                summary,
                peak_tflops=args.peak_tflops,
                hidden_size=args.hidden_size,
                local_intermediate_size=args.local_intermediate_size,
                num_experts=args.num_experts,
                topk=args.topk,
                tp_size=args.tp_size,
            )

    environment = os.environ.copy()
    hook_dir = Path(__file__).resolve().parent / "sglang_moe_capture"
    environment.update(
        {
            "PYTHONPATH": str(hook_dir)
            + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""),
            "SGL_MOE_CAPTURE": "1",
            "SGL_MOE_CAPTURE_BS": str(batch_size),
            "SGL_MOE_CAPTURE_TP_RANK": str(args.tp_rank),
            "SGL_MOE_CAPTURE_CALLS": str(args.num_layers),
            "SGL_MOE_CAPTURE_WARMUP": str(args.replay_warmup),
            "SGL_MOE_CAPTURE_REPEATS": str(args.replay_repeats),
            "SGL_MOE_CAPTURE_DIR": str(point_dir),
            "SGL_MOE_CAPTURE_SAVE_TENSORS": "1" if args.save_tensors else "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    if args.cuda_visible_devices:
        environment["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    server_command = [
        str(args.nsys),
        "profile",
        "--trace=cuda,nvtx",
        "--sample=none",
        "--cpuctxsw=none",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop-shutdown",
        "--force-overwrite=true",
        "-o",
        str(prefix),
        str(args.python),
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(args.model),
        "--served-model-name",
        args.served_model_name,
        "--host",
        "0.0.0.0",
        "--port",
        str(args.port),
        "--tp-size",
        str(args.tp_size),
        "--trust-remote-code",
        "--moe-runner-backend",
        "flashinfer_mxfp4",
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--chunked-prefill-size",
        str(args.chunked_prefill_size),
        "--max-running-requests",
        str(max(args.max_running_requests, batch_size)),
        "--disable-cuda-graph",
        "--disable-radix-cache",
        "--random-seed",
        "1234",
        "--decode-log-interval",
        "1",
    ]

    print(f"\n===== Capturing real SGLang MoE BS={batch_size} =====", flush=True)
    ensure_port_available(args.port)
    server_log = server_log_path.open("w", encoding="utf-8")
    server = subprocess.Popen(
        server_command,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        env=environment,
        start_new_session=True,
    )
    bench: subprocess.Popen | None = None
    try:
        wait_ready(args.port, args.server_ready_timeout, server)
        benchmark_command = [
            str(args.python),
            "-m",
            "sglang.benchmark.serving",
            "--backend",
            "sglang",
            "--dataset-name",
            "random",
            "--dataset-path",
            str(args.dataset),
            "--random-input",
            str(args.input_len),
            "--random-output",
            str(args.output_len),
            "--random-range-ratio",
            "1",
            "--num-prompts",
            str(batch_size),
            "--max-concurrency",
            str(batch_size),
            "--warmup-requests",
            "0",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--model",
            str(args.model),
            "--tokenizer",
            str(args.model),
            "--seed",
            "5678",
            "--output-file",
            str(point_dir / "bench.jsonl"),
        ]
        bench_log = bench_log_path.open("w", encoding="utf-8")
        bench = subprocess.Popen(
            benchmark_command,
            stdout=bench_log,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )

        deadline = time.monotonic() + args.capture_timeout
        while time.monotonic() < deadline:
            if capture_metadata.is_file():
                metadata = json.loads(capture_metadata.read_text(encoding="utf-8"))
                if metadata.get("status") == "complete":
                    break
            if server.poll() is not None and not capture_metadata.is_file():
                raise RuntimeError(
                    f"server exited before capture completed; see {server_log_path}"
                )
            time.sleep(2)
        else:
            raise TimeoutError(
                f"MoE capture did not complete within {args.capture_timeout}s"
            )

        if bench.poll() is None:
            terminate_process_group(bench, timeout=10)
        try:
            server.wait(timeout=args.nsys_finalize_timeout)
        except subprocess.TimeoutExpired:
            # The capture is complete; terminating only this child group lets
            # Nsys finish a Qdstrm that can be imported below.
            terminate_process_group(server, timeout=30)
    finally:
        if bench is not None and bench.poll() is None:
            terminate_process_group(bench, timeout=10)
        terminate_process_group(server, timeout=30)
        wait_port_available(args.port, timeout=60)
        server_log.close()
        if "bench_log" in locals():
            bench_log.close()

    metadata = json.loads(capture_metadata.read_text(encoding="utf-8"))
    expected_pairs = int(metadata["expected_grouped_gemm_pairs"])
    report = ensure_report(point_dir, prefix, args.importer, args.import_lib_dir)
    sqlite_path = export_sqlite(args.nsys, report)
    summary = summarize_pairs(grouped_gemm_durations(sqlite_path), expected_pairs)
    summary.update(
        {
            "batch_size": batch_size,
            "capture_metadata": metadata,
            "report": str(report),
            "sqlite": str(sqlite_path),
        }
    )
    (point_dir / "grouped_gemm_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"BS={batch_size}: gate/up={summary['up_mean_us']:.3f} us, "
        f"down={summary['down_mean_us']:.3f} us, "
        f"total={summary['total_mean_us']:.3f} us, pairs={expected_pairs}",
        flush=True,
    )
    return build_lookup_row(
        batch_size,
        metadata,
        summary,
        peak_tflops=args.peak_tflops,
        hidden_size=args.hidden_size,
        local_intermediate_size=args.local_intermediate_size,
        num_experts=args.num_experts,
        topk=args.topk,
        tp_size=args.tp_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate DSV4 TP8DP1 H20 MoE lookup data from real SGLang batches."
    )
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=parse_int_list, default=parse_int_list("16,24,26,32"))
    parser.add_argument("--input-len", type=int, default=40960)
    parser.add_argument("--output-len", type=int, default=1000)
    parser.add_argument("--port", type=int, default=30003)
    parser.add_argument("--served-model-name", default="DeepSeek-V4-Pro")
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--num-layers", type=int, default=61)
    parser.add_argument("--num-experts", type=int, default=384)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--local-intermediate-size", type=int, default=384)
    parser.add_argument("--peak-tflops", type=float, default=148.0)
    parser.add_argument("--chunked-prefill-size", type=int, default=8192)
    parser.add_argument("--max-running-requests", type=int, default=32)
    parser.add_argument("--mem-fraction-static", type=float, default=0.90)
    parser.add_argument("--replay-warmup", type=int, default=1)
    parser.add_argument("--replay-repeats", type=int, default=10)
    parser.add_argument("--server-ready-timeout", type=float, default=900)
    parser.add_argument("--capture-timeout", type=float, default=1800)
    parser.add_argument("--nsys-finalize-timeout", type=float, default=180)
    parser.add_argument("--cuda-visible-devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--nsys", type=Path, default=DEFAULT_NSYS)
    parser.add_argument("--importer", type=Path, default=DEFAULT_IMPORTER)
    parser.add_argument("--import-lib-dir", type=Path, default=DEFAULT_IMPORT_LIBS)
    parser.add_argument("--save-tensors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse completed per-BS captures in the output directory.",
    )
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()

    for required in (args.python, args.model, args.dataset, args.nsys, args.importer):
        if not required.exists():
            raise SystemExit(f"required path does not exist: {required}")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for batch_size in args.batch_sizes:
        point_dir = args.output_dir / f"bs{batch_size}"
        rows.append(capture_point(args, batch_size, point_dir))

    output_csv = args.output_dir / MOE_FILE
    write_csv(output_csv, rows)
    print(f"CSV_COMPLETE {output_csv}", flush=True)
    if args.install:
        install_moe_results(args.output_dir, Path(__file__).resolve().parent.parent)
    print("RUN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
