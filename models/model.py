import math

from comm.comm import Comm
from flops.flops import get_attn_gflops, get_moe_gflops
from hardware.gpu import gpu_map
from kvcache.kvcache import get_kvcache_size
from layers.attn import create_attention
from layers.moe import MoE
from params.params import get_attn_params_size, get_expert_params_size


def get_dsv4_runtime_layer_latency_us(config, c4_latency_us, c128_latency_us):
    """Return the layer-mix-weighted DSV4 runtime critical-path latency.

    ``c4_latency_us`` and ``c128_latency_us`` are full decoder-layer GPU spans,
    not isolated attention-kernel times.  Keeping this calculation separate
    prevents standalone kernel lookup latencies from being confused with the
    overlapped/fused runtime critical path.
    """

    c4_layers = config.compress_ratio_counts.get(4, 0)
    c128_layers = config.compress_ratio_counts.get(128, 0)
    calibrated_layers = c4_layers + c128_layers
    if calibrated_layers != config.num_hidden_layers:
        raise ValueError(
            "DSV4 runtime calibration requires every hidden layer to be C4 "
            f"or C128, got C4={c4_layers}, C128={c128_layers}, "
            f"hidden_layers={config.num_hidden_layers}"
        )

    return (
        c4_layers * c4_latency_us + c128_layers * c128_latency_us
    ) / calibrated_layers


class Model:
    def __init__(self, args, config):
        self.gpu = gpu_map[args.device_type]
        self.gpu_memory_gb = (
            args.gpu_memory_gb
            if getattr(args, "gpu_memory_gb", None) is not None
            else self.gpu.mem
        )
        self.args = args
        self.config = config

    def print_weights_info(self):
        print("{s:{c}^{n}}".format(s="Model Weights", n=50, c="-"))
        tp_size = self.args.tp_size
        attn_params_bytes = get_attn_params_size(
            self.config, self.args.use_fp8_gemm, tp_size
        )
        expert_params_bytes = get_expert_params_size(
            self.config, self.args.use_fp8_gemm, tp_size
        )
        print(
            "{:<40} {:<10.2f}".format(
                "One attn params size (MB):", attn_params_bytes / 1024 / 1024
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "One expert params size (MB):", expert_params_bytes / 1024 / 1024
            )
        )
        # Unified split rule:
        #   tp_size == 1 -> attn is DP (full), MoE routed experts split by EP (= world_size)
        #   tp_size >  1 -> attn and MoE both split by TP only (no EP/world_size division)
        if tp_size == 1:
            ep_size = self.args.world_size
            experts_on_gpu = (
                self.config.num_shared_experts
                + self.config.num_routed_experts / ep_size
            )
        else:
            experts_on_gpu = (
                self.config.num_shared_experts + self.config.num_routed_experts
            )
        params_per_gpu = attn_params_bytes + expert_params_bytes * experts_on_gpu
        params_per_gpu = params_per_gpu / 1024 / 1024 / 1024
        params_per_gpu *= self.config.num_hidden_layers
        self.kvcache_mem = (
            self.gpu_memory_gb - params_per_gpu - 15 - 5
        )  # 15GB for runtime, 5GB for encoder
        print("{:<40} {:<10.2f}".format("Per GPU params size (GB):", params_per_gpu))

    def print_kvcache_info(self):
        print("{s:{c}^{n}}".format(s="KV Cache", n=50, c="-"))
        print("{:<40} {:<10.2f}".format("KV cache space (GB):", self.kvcache_mem))
        context_len = self.args.target_isl + self.args.target_osl

        if self.args.decode_bs is None:
            target_bs = math.ceil(self.args.target_tgs * self.args.target_tpot / 1000)
        else:
            target_bs = self.args.decode_bs
        print("{:<40} {:<10}".format("Input seq len:", self.args.target_isl))
        print("{:<40} {:<10}".format("Output seq len:", self.args.target_osl))
        print("{:<40} {:<10}".format("Target decode batchsize:", target_bs))
        target_kvcache_bytes = (
            self.kvcache_mem * 1024 * 1024 * 1024 / target_bs / context_len
        )
        kvcache_bytes = get_kvcache_size(
            self.config, self.args.use_fp8_kv, self.args.tp_size
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Target per-token KV cache size (KB):", target_kvcache_bytes / 1024
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Current per-token KV cache size (KB):", kvcache_bytes / 1024
            )
        )
        if kvcache_bytes > target_kvcache_bytes:
            print("!Error: need smaller kvcache")
        self.kvcache_bytes = kvcache_bytes
        self.target_bs = target_bs

    def print_flops_info(self):
        print("{s:{c}^{n}}".format(s="FLOPs", n=50, c="-"))
        print(
            "{:<40} {:<10}".format("Num hidden layers:", self.config.num_hidden_layers)
        )
        # per-token per-layer gflops
        self.avg_context_len = int(self.args.target_isl + self.args.target_osl / 2)
        attn_core_gflops, other_gflops = get_attn_gflops(
            self.config, self.avg_context_len, self.args.tp_size, absorb=True
        )
        moe_gflops = get_moe_gflops(self.config, self.args.tp_size)
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token per-layer attn core (GFLOPs):", attn_core_gflops
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token per-layer MoE/FFN (GFLOPs):", moe_gflops
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token per-layer others (GFLOPs):", other_gflops
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token attn core (GFLOPs):",
                attn_core_gflops * self.config.num_hidden_layers,
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token MoE (GFLOPs):", moe_gflops * self.config.num_hidden_layers
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token others (GFLOPs):",
                other_gflops * self.config.num_hidden_layers,
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token total (GFLOPs):",
                (attn_core_gflops + moe_gflops + other_gflops)
                * self.config.num_hidden_layers,
            )
        )

    def prefill(self):
        print("{s:{c}^{n}}".format(s="Prefilling", n=50, c="-"))
        print(
            "{:<40} {:<10}".format("Max prefill tokens:", self.args.max_prefill_tokens)
        )
        attn = create_attention(
            self.config,
            self.args.use_fp8_gemm,
            self.args.use_fp8_kv,
            self.args.tp_size,
            getattr(self.args, "prefill_attn_mfu", None),
        )
        attn_core_time = attn.prefill_attn_core(
            self.args.target_isl, self.kvcache_bytes, self.args.device_type
        )
        attn_other_time = attn.prefill_attn_others(
            self.args.max_prefill_tokens, self.args.device_type
        )
        attn_core_time *= math.ceil(self.args.max_prefill_tokens / self.args.target_isl)

        moe = MoE(
            self.config,
            self.args.use_fp8_gemm,
            self.args.tp_size,
            getattr(self.args, "prefill_moe_mfu", None),
        )
        moe_time = moe.prefill_moe(
            self.args.max_prefill_tokens, self.args.device_type, self.args.world_size
        )

        comm = Comm(
            self.config,
            self.gpu,
            self.args.world_size,
            self.args.num_nodes,
            self.args.enable_deepep,
        )
        comm_time1, comm_time2 = comm.prefill_comm(self.args.max_prefill_tokens)
        print("{:<40} {:<10.2f}".format("Comm before MoE/FFN (us):", comm_time1 * 1e6))
        print("{:<40} {:<10.2f}".format("Comm after MoE/FFN (us):", comm_time2 * 1e6))

        # TP all_reduce communication time
        tp_comm_time = comm.tp_all_reduce(
            self.args.max_prefill_tokens, self.args.tp_size
        )
        if self.args.tp_size > 1:
            print("{:<40} {:<10.2f}".format("TP all_reduce (us):", tp_comm_time * 1e6))

        num_tokens = self.args.max_prefill_tokens
        if self.args.enable_tbo:
            num_tokens *= 2
            ttft = max(
                (attn_core_time + attn_other_time) / self.args.sm_ratio, comm_time1
            )
            ttft += max(
                (attn_core_time + attn_other_time) / self.args.sm_ratio, comm_time2
            )
            ttft += max(moe_time / self.args.sm_ratio, comm_time1)
            ttft += max(moe_time / self.args.sm_ratio, comm_time2)
        else:
            ttft = attn_core_time
            ttft += moe_time
            ttft += attn_other_time
            ttft += comm_time1 + comm_time2
            ttft += tp_comm_time  # Add TP communication time
        ttft *= self.config.num_hidden_layers
        ttft *= 1000  # convert to ms
        ttft += 30  # for scheduler

        print("{:<40} {:<10.2f}".format("TTFT (ms):", ttft))
        print(
            "{:<40} {:<10.0f}".format(
                "Throughput (TGS:tok/GPU/s):", num_tokens / self.args.tp_size / (ttft / 1000)
            )
        )

    def decoding(self):
        print("{s:{c}^{n}}".format(s="Decoding", n=50, c="-"))
        attn = create_attention(
            self.config, self.args.use_fp8_gemm, self.args.use_fp8_kv, self.args.tp_size
        )
        attn_core_time = attn.decode_attn_core(
            self.target_bs,
            self.avg_context_len,
            self.kvcache_bytes,
            self.args.device_type,
        )
        attn_other_time = attn.decode_attn_others(self.target_bs, self.args.device_type)

        moe = MoE(self.config, self.args.use_fp8_gemm, self.args.tp_size)
        moe_time = moe.decode_moe(
            self.target_bs, self.args.device_type, self.args.world_size
        )

        comm = Comm(
            self.config,
            self.gpu,
            self.args.world_size,
            self.args.num_nodes,
            self.args.enable_deepep,
        )
        comm_time1, comm_time2 = comm.decode_comm(self.target_bs)
        print("{:<40} {:<10.2f}".format("Comm before MoE/FFN (us):", comm_time1 * 1e6))
        print("{:<40} {:<10.2f}".format("Comm after MoE/FFN (us):", comm_time2 * 1e6))

        # TP all_reduce communication time
        tp_comm_time = comm.tp_all_reduce(self.target_bs, self.args.tp_size)
        if self.args.tp_size > 1:
            print("{:<40} {:<10.2f}".format("TP all_reduce (us):", tp_comm_time * 1e6))

        num_tokens = self.target_bs
        if self.args.enable_tbo:
            num_tokens *= 2
            analytical_layer_time = max(
                attn_core_time + attn_other_time, moe_time + comm_time1 + comm_time2
            )
            analytical_layer_time *= 2
        else:
            analytical_layer_time = attn_core_time
            analytical_layer_time += attn_other_time
            analytical_layer_time += moe_time
            analytical_layer_time += comm_time1 + comm_time2
            analytical_layer_time += tp_comm_time  # Add TP communication time
        scheduler_overhead_ms = getattr(
            self.args, "decode_scheduler_overhead_ms", None
        )
        if scheduler_overhead_ms is None:
            scheduler_overhead_ms = 5

        analytical_tpot_ms = (
            analytical_layer_time * self.config.num_hidden_layers * 1000
            + scheduler_overhead_ms
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Analytical layer latency (us):", analytical_layer_time * 1e6
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Analytical TPOT (ms):", analytical_tpot_ms
            )
        )

        c4_layer_latency_us = getattr(
            self.args, "dsv4_c4_layer_latency_us", None
        )
        c128_layer_latency_us = getattr(
            self.args, "dsv4_c128_layer_latency_us", None
        )
        if c4_layer_latency_us is not None:
            runtime_layer_latency_us = get_dsv4_runtime_layer_latency_us(
                self.config,
                c4_layer_latency_us,
                c128_layer_latency_us,
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Runtime C4 layer latency (us):", c4_layer_latency_us
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Runtime C128 layer latency (us):", c128_layer_latency_us
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Runtime weighted layer latency (us):",
                    runtime_layer_latency_us,
                )
            )
            tpot = (
                runtime_layer_latency_us
                * self.config.num_hidden_layers
                / 1000
                + scheduler_overhead_ms
            )
            print(
                "{:<40} {:<10}".format(
                    "Decode latency model:", "runtime-calibrated"
                )
            )
        else:
            tpot = analytical_tpot_ms
            print(
                "{:<40} {:<10}".format(
                    "Decode latency model:", "analytical lookup"
                )
            )

        print("{:<40} {:<10.2f}".format("TPOT (ms):", tpot))
        print(
            "{:<40} {:<10.0f}".format(
                "Throughput (TGS:tok/GPU/s):",
                num_tokens / self.args.tp_size / (tpot / 1000),
            )
        )
        if tpot > self.args.target_tpot:
            print("!Error: TPOT > SLO, need smaller GFLOPs to speedup")
