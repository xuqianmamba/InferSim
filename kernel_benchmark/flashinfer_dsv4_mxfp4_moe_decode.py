"""Benchmark the two DeepSeek-V4 MXFP4 grouped GEMMs directly.

This benchmark calls FlashInfer's patched ``run_grouped_gemm_only`` FFI entry.
FC1 stops immediately after the gate/up CUTLASS grouped GEMM and FC2 stops
immediately after the down-projection CUTLASS grouped GEMM.  There is no SGLang
server, CUDA graph, Nsight, torch profiler, SwiGLU, routing finalize, or fused
MoE span in the recorded latency.  Routing/workspace preparation and SM90
weight interleave happen outside the CUDA-event timing region.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from dsv4_bench_utils import parse_int_list, write_csv


GROUP_SIZE = 32
GROUPED_GEMM_NAME_MARKERS = (
    "cutlass::device_kernel",
    "gemmuniversal",
    "groupproblemshape",
)
CSV_FIELDS = (
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


def next_power_of_2(value: int) -> int:
    if value <= 0:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


def is_cutlass_grouped_gemm_kernel(name: str) -> bool:
    """Return whether a profiler event is one production grouped GEMM kernel."""
    lowered = name.lower()
    return all(marker in lowered for marker in GROUPED_GEMM_NAME_MARKERS)


def profiler_event_name(event) -> str:
    """Read names from FunctionEvent attributes or Kineto accessors."""
    value = getattr(event, "name", "")
    if callable(value):
        value = value()
    return str(value)


def profiler_event_duration_us(event) -> float:
    """Read a CUDA kernel duration from a raw ``torch.profiler`` event."""
    # Low-level ``_KinetoEvent`` exposes nanoseconds through a method.  Use it
    # before the microsecond fields used by FunctionEvent/Kernel records.
    duration_ns = getattr(event, "duration_ns", None)
    if duration_ns is not None:
        value = duration_ns() if callable(duration_ns) else duration_ns
        if float(value) > 0:
            return float(value) / 1_000.0

    for attribute in (
        # ``FunctionEvent.kernels`` contains ``Kernel`` records whose timing
        # field is named ``duration``.  Top-level CUDA ``FunctionEvent``
        # objects use one of the device/CUDA timing fields below instead.
        "duration",
        "device_time",
        "device_time_total",
        "cuda_time",
        "cuda_time_total",
    ):
        value = getattr(event, attribute, None)
        if value is not None and float(value) > 0:
            return float(value)

    time_range = getattr(event, "time_range", None)
    if time_range is not None and hasattr(time_range, "elapsed_us"):
        value = float(time_range.elapsed_us())
        if value > 0:
            return value
    raise RuntimeError(f"CUDA profiler event has no positive duration: {event!r}")


def _nested_profiler_kernels(event) -> tuple[object, ...]:
    """Return CUDA kernel records attached to a CPU ``FunctionEvent``.

    Depending on the PyTorch/Kineto version, ``profile.events()`` either
    exposes CUDA kernels as top-level events or attaches them to the launching
    CPU event through ``FunctionEvent.kernels``.  Normalize the latter without
    assuming that every profiler event implements the attribute.
    """
    kernels = getattr(event, "kernels", None)
    if kernels is None:
        return ()
    if callable(kernels):
        kernels = kernels()
    return tuple(kernels or ())


def _event_start_ns(event) -> int | None:
    value = getattr(event, "start_ns", None)
    if value is None:
        return None
    value = value() if callable(value) else value
    return int(value)


def _collect_direct_matches(events) -> tuple[list[tuple[str, float]], set[str]]:
    """Collect top-level kernel events, sorted by Kineto launch timestamp."""
    matched: list[tuple[int | None, int, str, float]] = []
    candidates: set[str] = set()
    for index, event in enumerate(events):
        name = profiler_event_name(event)
        if "cutlass::device_kernel" in name.lower():
            candidates.add(name)
        if is_cutlass_grouped_gemm_kernel(name):
            matched.append(
                (
                    _event_start_ns(event),
                    index,
                    name,
                    profiler_event_duration_us(event),
                )
            )

    # Kineto events can be returned in correlation order rather than strict
    # launch order.  When timestamps are available, restore CUDA launch order.
    if matched and all(start is not None for start, _, _, _ in matched):
        matched.sort(key=lambda item: (item[0], item[1]))
    return [(name, duration) for _, _, name, duration in matched], candidates


def _collect_nested_matches(events) -> tuple[list[tuple[str, float]], set[str]]:
    """Collect kernel records attached to FunctionEvent.kernels."""
    matched: list[tuple[str, float]] = []
    candidates: set[str] = set()
    for event in events:
        for kernel in _nested_profiler_kernels(event):
            name = profiler_event_name(kernel)
            if "cutlass::device_kernel" in name.lower():
                candidates.add(name)
            if is_cutlass_grouped_gemm_kernel(name):
                matched.append((name, profiler_event_duration_us(kernel)))
    return matched, candidates


def extract_grouped_gemm_samples(
    events,
    repeats: int,
    *,
    kineto_events=None,
) -> dict[str, object]:
    """Extract ordered FC1/FC2 kernel samples from raw profiler events.

    One FlashInfer MXFP4 MoE invocation must launch exactly two matching
    CUTLASS GroupProblemShape kernels.  Pairing is by launch order, not by
    kernel name, because FC1 and FC2 may use the same demangled kernel name.
    """
    events = tuple(events)
    direct_matches, direct_candidates = _collect_direct_matches(events)
    nested_matches, nested_candidates = _collect_nested_matches(events)
    kineto_matches: list[tuple[str, float]] = []
    kineto_candidates: set[str] = set()
    if kineto_events is not None:
        kineto_matches, kineto_candidates = _collect_direct_matches(
            tuple(kineto_events)
        )

    expected = 2 * repeats
    # Never add representations together: profiler versions may expose the
    # same launches in two or all three locations.  Low-level Kineto events
    # are the most faithful source; nested kernels are the portable fallback.
    if len(kineto_matches) == expected:
        matched = kineto_matches
        event_source = "kineto_results.events"
    elif len(nested_matches) == expected:
        matched = nested_matches
        event_source = "FunctionEvent.kernels"
    elif len(direct_matches) == expected:
        matched = direct_matches
        event_source = "top-level events"
    else:
        raise RuntimeError(
            "expected exactly two CUTLASS grouped GEMM kernels per eager call: "
            f"kineto_matched={len(kineto_matches)}, "
            f"nested_matched={len(nested_matches)}, "
            f"direct_matched={len(direct_matches)}, expected={expected}, "
            f"repeats={repeats}; Kineto CUTLASS candidates="
            f"{sorted(kineto_candidates)}; nested CUTLASS candidates="
            f"{sorted(nested_candidates)}; direct CUTLASS candidates="
            f"{sorted(direct_candidates)}"
        )

    up_samples = [duration for _, duration in matched[0::2]]
    down_samples = [duration for _, duration in matched[1::2]]
    return {
        "up_samples_us": up_samples,
        "down_samples_us": down_samples,
        "up_median_us": statistics.median(up_samples),
        "down_median_us": statistics.median(down_samples),
        "kernel_names": sorted({name for name, _ in matched}),
        "kernel_event_source": event_source,
    }


def load_shape(
    config_path: str | Path, world_size: int, tp_size: int
) -> dict[str, object]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if world_size % tp_size:
        raise ValueError("world_size must be divisible by tp_size")
    ep_size = world_size // tp_size
    experts_value = config.get("num_routed_experts")
    if experts_value is None:
        experts_value = config["n_routed_experts"]
    experts = int(experts_value)
    if experts % ep_size:
        raise ValueError("routed experts must be divisible by EP size")
    intermediate = int(config["moe_intermediate_size"])
    if intermediate % tp_size:
        raise ValueError("moe_intermediate_size must be divisible by TP size")
    return {
        "num_experts": experts,
        "num_local_experts": experts // ep_size,
        "hidden": int(config["hidden_size"]),
        "intermediate": intermediate // tp_size,
        "topk": int(config["num_experts_per_tok"]),
        "world_size": world_size,
        "tp_size": tp_size,
        "ep_size": ep_size,
        "swiglu_limit": config.get("swiglu_limit"),
    }


def make_topk(
    tokens: int, experts: int, topk: int
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    generator = torch.Generator(device="cuda").manual_seed(1235 + tokens)
    logits = torch.randn(
        tokens, experts, dtype=torch.float32, device="cuda", generator=generator
    )
    weights, ids = torch.topk(torch.softmax(logits, dim=-1), topk, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    ids = ids.to(torch.int32)
    counts = torch.bincount(ids.flatten(), minlength=experts)
    stats = {
        "active_experts": int((counts > 0).sum().item()),
        "max_tokens_per_expert": int(counts.max().item()),
    }
    return weights.to(torch.float32), ids, stats


def prepare_weights(experts: int, hidden: int, intermediate: int):
    from flashinfer.fused_moe import (
        interleave_moe_scales_for_sm90_mixed_gemm,
        interleave_moe_weights_for_sm90_mixed_gemm,
    )

    generator = torch.Generator(device="cuda").manual_seed(1234)
    w13 = torch.randint(
        -128,
        128,
        (experts, 2 * intermediate, hidden // 2),
        dtype=torch.int8,
        device="cuda",
        generator=generator,
    )
    w2 = torch.randint(
        -128,
        128,
        (experts, hidden, intermediate // 2),
        dtype=torch.int8,
        device="cuda",
        generator=generator,
    )
    w13_scale = torch.randint(
        125,
        130,
        (experts, 2 * intermediate, hidden // GROUP_SIZE),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    w2_scale = torch.randint(
        125,
        130,
        (experts, hidden, intermediate // GROUP_SIZE),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )

    # Match FusedMoE's load_up_proj_weight_first loader contract: checkpoint
    # gate/up (w1,w3) is reordered to the production (w3,w1) SwiGLU layout.
    w1, w3 = w13.chunk(2, dim=1)
    s1, s3 = w13_scale.chunk(2, dim=1)
    w31 = torch.cat((w3, w1), dim=1)
    s31 = torch.cat((s3, s1), dim=1)

    return (
        interleave_moe_weights_for_sm90_mixed_gemm(
            w31.view(torch.uint8).contiguous(), "fp4"
        ),
        interleave_moe_weights_for_sm90_mixed_gemm(
            w2.view(torch.uint8).contiguous(), "fp4"
        ),
        interleave_moe_scales_for_sm90_mixed_gemm(s31, group_size=GROUP_SIZE).view(
            torch.int32
        ),
        interleave_moe_scales_for_sm90_mixed_gemm(
            w2_scale, group_size=GROUP_SIZE
        ).view(torch.int32),
    )


def create_grouped_gemm_runner():
    """Build/load the benchmark-only SM90 FFI and construct its runner."""
    from flashinfer_grouped_gemm_only_jit import load_grouped_gemm_only_module

    module = load_grouped_gemm_only_module()
    runner = module.init(
        torch.bfloat16,
        torch.uint8,
        torch.bfloat16,
        False,  # use_deepseek_fp8_block_scale
        True,  # use_w4_group_scaling (BF16 x MXFP4)
        False,  # use_mxfp8_act_scaling
        False,  # input weights are already SM90-interleaved
        False,  # never request fused finalize in the benchmark runner
    )
    if not hasattr(runner, "run_grouped_gemm_only"):
        raise RuntimeError("benchmark-only FlashInfer FFI was not loaded")
    return runner


def tactic_ids_for_stage(runner, stage: int) -> list[int]:
    gemm1_count = int(runner.get_gemm1_tactic_count())
    gemm2_count = int(runner.get_gemm2_tactic_count())
    if stage == 1:
        candidates = range(gemm1_count)
    elif stage == 2:
        candidates = range(gemm1_count, gemm1_count + gemm2_count)
    else:
        raise ValueError(f"invalid grouped GEMM stage: {stage}")
    result = []
    for tactic in candidates:
        try:
            if int(runner.get_tactic_occupancy(tactic)) > 0:
                result.append(tactic)
        except Exception:
            result.append(tactic)
    return result


def measure_grouped_gemm_stage(
    runner,
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    shape: dict[str, object],
    stage: int,
    warmup: int,
    repeats: int,
) -> tuple[float, int, list[float]]:
    """Autotune and time exactly one grouped GEMM using CUDA events."""
    from flashinfer.fused_moe.core import ActivationType

    def launch(tactic: int, prepare: bool) -> None:
        runner.run_grouped_gemm_only(
            hidden_states,
            w13,
            None,
            w2,
            None,
            shape["topk"],
            shape["tp_size"],
            0,
            shape["ep_size"],
            0,
            1,
            0,
            False,
            False,
            stage,
            tactic,
            prepare,
            False,
            ActivationType.Swiglu,
        )

    valid: list[tuple[float, int, list[float]]] = []
    errors: list[str] = []
    for tactic in tactic_ids_for_stage(runner, stage):
        try:
            # The profiler-style preparation creates routing, offsets, TMA
            # descriptors, scales, and workspace outside the timed region.
            launch(tactic, True)
            torch.cuda.synchronize()
            for _ in range(warmup):
                launch(tactic, False)
            torch.cuda.synchronize()

            samples = []
            for _ in range(repeats):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                launch(tactic, False)
                end.record()
                end.synchronize()
                samples.append(float(start.elapsed_time(end)) * 1_000.0)
            valid.append((statistics.median(samples), tactic, samples))
        except Exception as error:
            torch.cuda.synchronize()
            errors.append(f"tactic={tactic}: {error}")

    if not valid:
        raise RuntimeError(
            f"no valid TMA grouped-GEMM tactic for stage {stage}; "
            + "; ".join(errors)
        )
    return min(valid, key=lambda item: item[0])


def benchmark_point(
    tokens: int,
    shape: dict[str, object],
    weights,
    warmup: int,
    repeats: int,
    peak_tflops: float,
    execution_mode: str,
) -> dict:
    generator = torch.Generator(device="cuda").manual_seed(4321 + tokens)
    hidden_states = torch.randn(
        tokens,
        shape["hidden"],
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    topk_weights, topk_ids, routing_stats = make_topk(
        tokens, shape["num_local_experts"], shape["topk"]
    )
    w13, w2, w13_scale, w2_scale = weights
    tune_max_num_tokens = next_power_of_2(tokens)

    if execution_mode != "eager":
        raise ValueError(
            "grouped-GEMM lookup generation uses direct eager FFI launches"
        )

    # The patched runner's profiler preparation creates the exact grouped
    # problem shape and synthetic routing used by FlashInfer's tactic tuner.
    # topk tensors/scales above are used only to report deterministic routing
    # statistics; they are deliberately not in either timed kernel call.
    del topk_weights, topk_ids, w13_scale, w2_scale
    runner = create_grouped_gemm_runner()
    up_latency_us, up_tactic, up_samples = measure_grouped_gemm_stage(
        runner, hidden_states, w13, w2, shape, 1, warmup, repeats
    )
    down_latency_us, down_tactic, down_samples = measure_grouped_gemm_stage(
        runner, hidden_states, w13, w2, shape, 2, warmup, repeats
    )
    total_latency_us = up_latency_us + down_latency_us

    # Gate/up is two HxI GEMMs; down is one IxH GEMM.  Report their useful
    # FLOPs and MFUs separately instead of assigning one fused-span metric to
    # both legacy projection columns.
    one_projection_flops = (
        2
        * shape["hidden"]
        * shape["intermediate"]
        * tokens
        * shape["topk"]
    )
    up_flops = 2 * one_projection_flops
    down_flops = one_projection_flops
    total_flops = up_flops + down_flops
    up_tflops = up_flops / (up_latency_us * 1e-6) / 1e12
    down_tflops = down_flops / (down_latency_us * 1e-6) / 1e12
    total_tflops = total_flops / (total_latency_us * 1e-6) / 1e12
    up_mfu = up_tflops / peak_tflops
    down_mfu = down_tflops / peak_tflops
    total_mfu = total_tflops / peak_tflops
    print(
        json.dumps(
            {
                "batch_size": tokens,
                "up_proj_us": round(up_latency_us, 3),
                "down_proj_us": round(down_latency_us, 3),
                "grouped_gemm_total_us": round(total_latency_us, 3),
                "up_mfu": round(up_mfu, 6),
                "down_mfu": round(down_mfu, 6),
                "total_mfu": round(total_mfu, 6),
                "measured_calls_per_stage": repeats,
                "up_tactic": up_tactic,
                "down_tactic": down_tactic,
                "up_samples_us": [round(value, 3) for value in up_samples],
                "down_samples_us": [round(value, 3) for value in down_samples],
                "timing": "cuda_event_pure_grouped_gemm_ffi",
                "weight_dtype": "mxfp4_e2m1",
                "activation_dtype": "bf16",
                "tp_size": shape["tp_size"],
                "ep_size": shape["ep_size"],
                "tune_max_num_tokens": tune_max_num_tokens,
                "execution_mode": execution_mode,
                **routing_stats,
            }
        )
    )
    return {
        "num_experts": shape["num_experts"],
        "num_gpus": shape["world_size"],
        "num_local_experts": shape["num_local_experts"],
        "topk": shape["topk"],
        "hidden_size": shape["hidden"],
        "intermediate_size": shape["intermediate"],
        "batch_size_per_gpu": tokens,
        "tokens_per_expert": round(
            tokens * shape["topk"] / shape["num_local_experts"]
        ),
        "up_proj_us": round(up_latency_us, 3),
        "up_mfu": round(up_mfu, 6),
        "down_proj_us": round(down_latency_us, 3),
        "down_mfu": round(down_mfu, 6),
        "total_latency_us": round(total_latency_us, 3),
        "total_mfu": round(total_mfu, 6),
        "kernel_kind": "cutlass_grouped_gemm_pair_pure_ffi",
        "backend": "flashinfer_mxfp4_sm90",
        "activation_dtype": "bf16",
        "weight_dtype": "mxfp4_e2m1",
        "mfu_peak_tflops": peak_tflops,
        "tp_size": shape["tp_size"],
        "ep_size": shape["ep_size"],
        "tune_max_num_tokens": tune_max_num_tokens,
        "execution_mode": execution_mode,
        **routing_stats,
        "routing_mode": "synthetic_uniform_softmax",
        "swiglu_limit": (
            "" if shape["swiglu_limit"] is None else shape["swiglu_limit"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--batch-sizes", default="16,24,26,32")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--execution-mode",
        choices=("eager",),
        default="eager",
        help=(
            "Direct eager FFI launches timed separately with CUDA events; "
            "no profiler or CUDA graph is used."
        ),
    )
    parser.add_argument(
        "--peak-tflops",
        type=float,
        default=148.0,
        help="MFU normalization used by InferSim when --use-fp8-gemm is absent.",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    shape = load_shape(args.config_path, args.world_size, args.tp_size)
    batch_sizes = parse_int_list(args.batch_sizes)
    if (shape["world_size"], shape["tp_size"], shape["ep_size"]) != (8, 8, 1):
        raise SystemExit("this benchmark refresh is specifically for TP8DP1")
    print("DSV4 MoE shape:", json.dumps(shape, sort_keys=True))
    torch.set_default_device("cuda")
    weights = prepare_weights(
        shape["num_local_experts"], shape["hidden"], shape["intermediate"]
    )
    rows = [
        benchmark_point(
            batch,
            shape,
            weights,
            args.warmup,
            args.repeats,
            args.peak_tflops,
            args.execution_mode,
        )
        for batch in batch_sizes
    ]
    path = Path(args.output_dir) / "groupedgemm-decode-dsv4-tp8dp1.csv"
    write_csv(path, CSV_FIELDS, rows)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
