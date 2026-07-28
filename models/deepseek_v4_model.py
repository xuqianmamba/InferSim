"""Table-driven end-to-end simulator for DeepSeek-V4."""

import math

from comm.comm import Comm
from config.model_config import get_deepseek_v4_architecture_fingerprint
from hardware.gpu import gpu_map
from kernel_sim.dsv4 import (
    DSV4BenchmarkData,
    canonical_dsv4_attention_backend,
    canonical_dsv4_kv_dtype,
    canonical_dsv4_moe_key,
)


class DeepSeekV4Model:
    """Sum exact measured C4/C128 attention, mHC and MoE layer latencies."""

    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.device_type = self._arg("device_type", "H20")
        self.gpu = gpu_map[self.device_type]
        self.world_size = self._arg("world_size", 1)
        self.tp_size = self._arg("tp_size", 1)
        self.num_nodes = self._arg("num_nodes", 1)
        if (
            self.tp_size <= 0
            or self.world_size < self.tp_size
            or self.world_size % self.tp_size
        ):
            raise ValueError("world_size must be a positive multiple of tp_size")
        if self.config.num_attention_heads % self.tp_size:
            raise ValueError(
                "DeepSeek-V4 num_attention_heads must be divisible by tp_size"
            )

        self.attention_weight_dtype = (
            self._arg("dsv4_attention_weight_dtype", None)
            or config.attention_weight_dtype
        )
        self.expert_weight_dtype = (
            self._arg("dsv4_expert_weight_dtype", None) or config.expert_dtype
        )
        self.kv_dtype = canonical_dsv4_kv_dtype(self._arg("dsv4_kv_dtype", "fp8_e4m3"))
        self.compress_state_dtype = self._arg("dsv4_compress_state_dtype", "fp32")
        self.attn_backend = canonical_dsv4_attention_backend(
            self._arg("dsv4_attn_backend", "dsv4")
        )
        self.moe_backend, self.expert_weight_dtype = canonical_dsv4_moe_key(
            self._arg("dsv4_moe_backend", "flashinfer_mxfp4"),
            self.expert_weight_dtype,
        )
        self.mhc_mode = self._arg("dsv4_mhc_mode", "fused_post_pre")
        self.benchmarks = DSV4BenchmarkData(
            self._arg("dsv4_bench_data_dir", None),
            architecture_fingerprint=get_deepseek_v4_architecture_fingerprint(config),
        )

        self.target_isl = self._arg("target_isl", 4096)
        self.target_osl = self._arg("target_osl", 2048)
        self.max_prefill_tokens = self._arg("max_prefill_tokens", 4096)
        self.decode_past_len = self._arg("dsv4_decode_past_len", None)
        if self.decode_past_len is None:
            self.decode_past_len = int(self.target_isl + self.target_osl / 2)
        decode_bs = self._arg("decode_bs", None)
        if decode_bs is None:
            decode_bs = math.ceil(
                self._arg("target_tgs", 2560) * self._arg("target_tpot", 50) / 1000
            )
        self.target_bs = decode_bs

    def _arg(self, name, default):
        return getattr(self.args, name, default)

    def _prefill_batch_size(self):
        explicit = self._arg("dsv4_prefill_batch_size", None)
        if explicit is not None:
            return explicit
        return math.ceil(self.max_prefill_tokens / self.target_isl)

    def _comm_time_us(self, phase, num_tokens):
        comm = Comm(
            self.config,
            self.gpu,
            self.world_size,
            self.num_nodes,
            self._arg("enable_deepep", False),
        )
        if self.tp_size > 1:
            # One reduction follows attention and another follows MoE.
            per_layer = 2 * comm.tp_all_reduce(num_tokens, self.tp_size)
        elif phase == "prefill":
            before_moe, after_moe = comm.prefill_comm(num_tokens)
            per_layer = before_moe + after_moe
        else:
            before_moe, after_moe = comm.decode_comm(num_tokens)
            per_layer = before_moe + after_moe
        return per_layer * self.config.num_hidden_layers * 1e6

    def _phase_latency_us(self, phase, batch_size, q_len, past_len):
        attention = {}
        for ratio, layer_count in self.config.attention_ratio_counts.items():
            one_layer_us = self.benchmarks.attention_latency_us(
                phase=phase,
                device_type=self.device_type,
                tp_size=self.tp_size,
                backend=self.attn_backend,
                attention_weight_dtype=self.attention_weight_dtype,
                kv_dtype=self.kv_dtype,
                compress_state_dtype=self.compress_state_dtype,
                ratio=ratio,
                batch_size=batch_size,
                q_len=q_len,
                past_len=past_len,
            )
            attention[ratio] = layer_count * one_layer_us

        mhc_us = self.config.num_hidden_layers * self.benchmarks.mhc_latency_us(
            phase=phase,
            device_type=self.device_type,
            tp_size=self.tp_size,
            backend=self.attn_backend,
            attention_weight_dtype=self.attention_weight_dtype,
            batch_size=batch_size,
            q_len=q_len,
            mode=self.mhc_mode,
        )
        moe_us = self.config.num_hidden_layers * self.benchmarks.moe_latency_us(
            phase=phase,
            device_type=self.device_type,
            tp_size=self.tp_size,
            world_size=self.world_size,
            num_nodes=self.num_nodes,
            backend=self.moe_backend,
            expert_weight_dtype=self.expert_weight_dtype,
            batch_size=batch_size,
            q_len=q_len,
        )
        return {
            "attention": attention,
            "mhc": mhc_us,
            "moe": moe_us,
            "comm": self._comm_time_us(phase, batch_size * q_len),
        }

    @staticmethod
    def _print_phase_latency(name, timings):
        for ratio, latency_us in sorted(timings["attention"].items()):
            print("{:<40} {:<10.2f}".format(f"C{ratio} attention (us):", latency_us))
        print("{:<40} {:<10.2f}".format("mHC (us):", timings["mhc"]))
        print("{:<40} {:<10.2f}".format("MoE (us):", timings["moe"]))
        print("{:<40} {:<10.2f}".format("Communication (us):", timings["comm"]))
        total = (
            sum(timings["attention"].values())
            + timings["mhc"]
            + timings["moe"]
            + timings["comm"]
        )
        print("{:<40} {:<10.2f}".format(f"{name} total (us):", total))
        return total

    def print_weights_info(self):
        print("{s:{c}^{n}}".format(s="Model Weights", n=50, c="-"))
        print(
            "{:<40} {:<10}".format(
                "V4 attention weight dtype:", self.attention_weight_dtype
            )
        )
        print(
            "{:<40} {:<10}".format("V4 expert weight dtype:", self.expert_weight_dtype)
        )
        print("{:<40} {:<10}".format("V4 MoE backend:", self.moe_backend))

    def print_kvcache_info(self):
        print("{s:{c}^{n}}".format(s="KV Cache", n=50, c="-"))
        print("{:<40} {:<10}".format("V4 KV dtype:", self.kv_dtype))
        print("{:<40} {:<10}".format("Input seq len:", self.target_isl))
        print("{:<40} {:<10}".format("Output seq len:", self.target_osl))
        print("{:<40} {:<10}".format("Decode batch size:", self.target_bs))

    def print_flops_info(self):
        print("{s:{c}^{n}}".format(s="V4 Kernel Data", n=50, c="-"))
        groups = ", ".join(
            f"C{ratio} x {layer_count}"
            for ratio, layer_count in self.config.attention_ratio_counts.items()
        )
        print("{:<40} {:<10}".format("Attention layer groups:", groups))
        print("{:<40} {:<10}".format("mHC mode:", self.mhc_mode))
        print(
            "Attention, mHC and MoE use exact measured rows; no generic "
            "MLA/MHA MFU fallback is applied."
        )

    def _ensure_supported_mode(self):
        if self._arg("enable_tbo", False):
            raise ValueError(
                "--enable-tbo requires overlapping V4 component measurements"
            )

    def prefill(self):
        self._ensure_supported_mode()
        print("{s:{c}^{n}}".format(s="Prefilling", n=50, c="-"))
        batch_size = self._prefill_batch_size()
        timings = self._phase_latency_us("prefill", batch_size, self.target_isl, 0)
        total_us = self._print_phase_latency("Prefill", timings)
        ttft_ms = total_us / 1000 + 30
        tokens = batch_size * self.target_isl
        print("{:<40} {:<10.2f}".format("TTFT (ms):", ttft_ms))
        print(
            "{:<40} {:<10.0f}".format(
                "Throughput (TGS:tok/GPU/s):",
                tokens / self.tp_size / (ttft_ms / 1000),
            )
        )

    def decoding(self):
        self._ensure_supported_mode()
        print("{s:{c}^{n}}".format(s="Decoding", n=50, c="-"))
        timings = self._phase_latency_us(
            "decode", self.target_bs, 1, self.decode_past_len
        )
        total_us = self._print_phase_latency("Decode", timings)
        overhead_ms = self._arg("decode_scheduler_overhead_ms", None)
        if overhead_ms is None:
            overhead_ms = 5
        tpot_ms = total_us / 1000 + overhead_ms
        print("{:<40} {:<10.2f}".format("TPOT (ms):", tpot_ms))
        print(
            "{:<40} {:<10.0f}".format(
                "Throughput (TGS:tok/GPU/s):",
                self.target_bs / self.tp_size / (tpot_ms / 1000),
            )
        )
        if tpot_ms > self._arg("target_tpot", 50):
            print("!Error: TPOT > SLO, need smaller GFLOPs to speedup")
