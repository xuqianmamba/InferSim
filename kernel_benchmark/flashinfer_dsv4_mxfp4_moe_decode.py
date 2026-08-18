"""Benchmark DeepSeek-V4's production FlashInfer MXFP4 fused-MoE kernel.

The timed region is the same ``cutlass_fused_moe`` call used by SGLang's
``flashinfer_mxfp4`` runner.  Routing, random tensor creation, and FlashInfer's
SM90 weight/scale interleave are deliberately outside the timed region.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dsv4_bench_utils import parse_int_list, write_csv


GROUP_SIZE = 32
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
)


def load_shape(config_path: str | Path, world_size: int, tp_size: int) -> dict[str, int]:
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
    }


def make_topk(tokens: int, experts: int, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(1235 + tokens)
    logits = torch.randn(
        tokens, experts, dtype=torch.float32, device="cuda", generator=generator
    )
    weights, ids = torch.topk(torch.softmax(logits, dim=-1), topk, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights.to(torch.float32), ids.to(torch.int32)


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


def benchmark_point(
    tokens: int,
    shape: dict[str, int],
    weights,
    warmup: int,
    repeats: int,
    peak_tflops: float,
) -> dict:
    from flashinfer.fused_moe import cutlass_fused_moe
    from flashinfer.fused_moe.core import ActivationType

    generator = torch.Generator(device="cuda").manual_seed(4321 + tokens)
    hidden_states = torch.randn(
        tokens,
        shape["hidden"],
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    topk_weights, topk_ids = make_topk(
        tokens, shape["num_local_experts"], shape["topk"]
    )
    output = torch.empty_like(hidden_states)
    w13, w2, w13_scale, w2_scale = weights

    def call() -> None:
        cutlass_fused_moe(
            input=hidden_states,
            token_selected_experts=topk_ids,
            token_final_scales=topk_weights,
            fc1_expert_weights=w13,
            fc2_expert_weights=w2,
            output_dtype=torch.bfloat16,
            quant_scales=[w13_scale, w2_scale],
            fc1_expert_biases=None,
            fc2_expert_biases=None,
            swiglu_alpha=None,
            swiglu_beta=None,
            swiglu_limit=None,
            use_w4_group_scaling=True,
            activation_type=ActivationType.Swiglu,
            output=output,
        )

    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(repeats):
        start.record()
        call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    latency_us = sorted(samples)[len(samples) // 2]

    # Same useful-FLOP model as layers/moe.py: gate/up/down = three GEMMs.
    flops = (
        2
        * shape["hidden"]
        * shape["intermediate"]
        * tokens
        * shape["topk"]
        * 3
    )
    achieved_tflops = flops / (latency_us * 1e-6) / 1e12
    mfu = achieved_tflops / peak_tflops
    print(
        json.dumps(
            {
                "batch_size": tokens,
                "latency_us": round(latency_us, 3),
                "achieved_tflops": round(achieved_tflops, 3),
                "mfu": round(mfu, 6),
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
        # The fused call cannot attribute elapsed time to FC1/FC2 separately.
        # Keep the established InferSim convention: store combined latency and
        # combined useful MFU in both columns; the simulator consumes max(MFU).
        "up_proj_us": round(latency_us, 3),
        "up_mfu": round(mfu, 6),
        "down_proj_us": round(latency_us, 3),
        "down_mfu": round(mfu, 6),
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
            batch, shape, weights, args.warmup, args.repeats, args.peak_tflops
        )
        for batch in batch_sizes
    ]
    path = Path(args.output_dir) / "groupedgemm-decode-dsv4-tp8dp1.csv"
    write_csv(path, CSV_FIELDS, rows)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
