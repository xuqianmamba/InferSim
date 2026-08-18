from flops.flops import gemm_flops
from hardware.gpu import TFLOPS_TO_GFLOPS, gpu_map
from mfu.mfu import (
    get_attn_decode_mfu,
    get_attn_decode_perf,
    get_attn_prefill_mfu,
    get_dsa_decode_perf,
    get_dsa_indexer_decode_perf,
    get_gemm_mfu,
)


def get_gemm_mfu_and_latency(m, k, n, device_type, use_fp8_gemm):
    gpu = gpu_map[device_type]
    gflops = gemm_flops(m, k, n) / 1e9
    mfu = get_gemm_mfu(device_type, m, k, n)
    latency = gflops / (gpu.fp16_tflops * TFLOPS_TO_GFLOPS * mfu)
    if use_fp8_gemm:
        latency = gflops / (gpu.fp8_tflops * TFLOPS_TO_GFLOPS * mfu)
    # print(f"Debug: gemm m:{m} k:{k} n:{n}")
    return latency


class MHA:
    def __init__(
        self, config, use_fp8_gemm, use_fp8_kv, tp_size, prefill_mfu=None
    ):
        self.use_fp8_gemm = use_fp8_gemm
        self.use_fp8_kv = use_fp8_kv
        self.config = config
        self.tp_size = tp_size
        self.prefill_mfu = prefill_mfu

    def get_attn_core_gflops(self, bs, kv_len):
        # TP shards attention heads
        tp_num_heads = self.config.num_attention_heads // self.tp_size
        attn_core = (
            gemm_flops(
                bs, tp_num_heads * self.config.head_dim, kv_len
            )
            * 2
        )
        return attn_core / 1e9

    def decode_attn_core(self, bs, kv_len, kvcache_bytes, device_type):
        gpu = gpu_map[device_type]
        attn_core_gflops = self.get_attn_core_gflops(1, kv_len)
        attn_core_mfu, measured_latency = get_attn_decode_perf(
            self.config, bs, kv_len, device_type, self.use_fp8_kv, self.tp_size
        )
        if measured_latency is not None:
            attn_core_time = measured_latency
        else:
            attn_core_time = bs * attn_core_gflops / (
                gpu.fp16_tflops * TFLOPS_TO_GFLOPS * attn_core_mfu
            )
        kv_load_time = (
            kvcache_bytes
            * kv_len
            * bs
            / self.config.num_full_attn_layers
            / 1024
            / 1024
            / 1024
            / gpu.mem_bw
        )

        print("{:<40} {:<10.2f}".format("Attn core MFU:", attn_core_mfu))
        print(
            "{:<40} {:<10.2f}".format("Attn core latency (us):", attn_core_time * 1e6)
        )
        print("{:<40} {:<10.2f}".format("KV loading latency (us):", kv_load_time * 1e6))

        if measured_latency is not None:
             # Decode benchmark latency already includes KV-cache reads.
             return attn_core_time
        return max(attn_core_time, kv_load_time)

    def decode_attn_others(self, bs, device_type):
        # TP shards heads; hidden_size is NOT sharded
        tp_num_heads = self.config.num_attention_heads // self.tp_size
        tp_num_kv_heads = self.config.num_key_value_heads // self.tp_size
        q_multiplier = 2 if self.config.attn_output_gate else 1
        qkv_output_size = (
            q_multiplier * tp_num_heads + 2 * tp_num_kv_heads
        ) * self.config.head_dim

        qkv_proj = get_gemm_mfu_and_latency(
            m=bs,
            k=self.config.hidden_size,
            n=qkv_output_size,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        print("{:<40} {:<10.2f}".format("QKV_proj latency (us):", qkv_proj * 1e6))

        o_proj = get_gemm_mfu_and_latency(
            m=bs,
            k=tp_num_heads * self.config.head_dim,
            n=self.config.hidden_size,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        print("{:<40} {:<10.2f}".format("O_proj latency (us):", o_proj * 1e6))
        return qkv_proj + o_proj

    def prefill_attn_core(self, seq_len, kvcache_bytes, device_type):
        gpu = gpu_map[device_type]
        attn_core_gflops = self.get_attn_core_gflops(1, seq_len)
        if self.prefill_mfu is None:
            attn_core_mfu = get_attn_prefill_mfu(
                self.config, seq_len, device_type, self.tp_size
            )
        else:
            attn_core_mfu = self.prefill_mfu
        attn_core_time = (
            seq_len
            * attn_core_gflops
            / 1.8
            / (gpu.fp16_tflops * TFLOPS_TO_GFLOPS * attn_core_mfu)
        )
        kv_load_time = (
            kvcache_bytes
            * seq_len
            / self.config.num_full_attn_layers
            / 1024
            / 1024
            / 1024
            / gpu.mem_bw
        )

        print("{:<40} {:<10.2f}".format("Attn core MFU:", attn_core_mfu))
        print(
            "{:<40} {:<10.2f}".format("Attn core latency (us):", attn_core_time * 1e6)
        )
        print("{:<40} {:<10.2f}".format("KV loading latency (us):", kv_load_time * 1e6))

        return max(attn_core_time, kv_load_time)

    def prefill_attn_others(self, seq_len, device_type):
        return self.decode_attn_others(seq_len, device_type)


class _MLACore(MHA):
    def __init__(self, config, use_fp8_gemm, use_fp8_kv, tp_size):
        self.use_fp8_gemm = use_fp8_gemm
        self.use_fp8_kv = use_fp8_kv
        self.config = config
        self.tp_size = tp_size

    def get_attn_core_gflops_absorb(self, bs, kv_len):
        attn_core = gemm_flops(
            bs,
            self.config.num_attention_heads
            * (self.config.kv_lora_rank + self.config.qk_rope_head_dim),
            kv_len,
        ) + gemm_flops(
            bs, kv_len, self.config.num_attention_heads * self.config.kv_lora_rank
        )
        return attn_core / 1e9

    def get_attn_core_gflops_noabsorb(self, bs, kv_len):
        attn_core = gemm_flops(
            bs,
            self.config.num_attention_heads
            * (self.config.qk_nope_head_dim + self.config.qk_rope_head_dim),
            kv_len,
        ) + gemm_flops(
            bs, kv_len, self.config.num_attention_heads * self.config.v_head_dim
        )
        return attn_core / 1e9

    def decode_attn_core(self, bs, kv_len, kvcache_bytes, device_type):
        gpu = gpu_map[device_type]
        attn_core_gflops = self.get_attn_core_gflops_absorb(1, kv_len)
        attn_core_mfu = get_attn_decode_mfu(
            self.config, bs, kv_len, device_type, self.use_fp8_kv, self.tp_size
        )
        attn_core_time = (
            bs
            * attn_core_gflops
            / (gpu.fp16_tflops * TFLOPS_TO_GFLOPS * attn_core_mfu)
        )
        kv_load_time = (
            kvcache_bytes
            * kv_len
            * bs
            / self.config.num_hidden_layers
            / 1024
            / 1024
            / 1024
            / gpu.mem_bw
        )

        print("{:<40} {:<10.2f}".format("Attn core MFU:", attn_core_mfu))
        print(
            "{:<40} {:<10.2f}".format("Attn core latency (us):", attn_core_time * 1e6)
        )
        print("{:<40} {:<10.2f}".format("KV loading latency (us):", kv_load_time * 1e6))

        return max(attn_core_time, kv_load_time)


class DSA(MHA):
    """DeepSeek-V4 sparse attention with per-layer compression ratios.

    The returned core latency is a weighted per-layer average.  ``Model``
    multiplies it by ``num_hidden_layers`` exactly once, preserving the
    existing simulator interface while accounting for C1/C4/C128 layers.
    """

    def __init__(self, config, use_fp8_gemm, use_fp8_kv, tp_size):
        self.use_fp8_gemm = use_fp8_gemm
        self.use_fp8_kv = use_fp8_kv
        self.config = config
        self.tp_size = tp_size

    def _attended(self, kv_len, ratio):
        if ratio == 0:
            return kv_len
        swa = min(kv_len, self.config.swa_window)
        compressed = max(kv_len // ratio, 1)
        extra = (
            min(self.config.index_topk, compressed)
            if ratio == 4
            else compressed
        )
        return swa + extra

    def _useful_flops(self, bs, kv_len, ratio):
        local_heads = self.config.num_attention_heads // self.tp_size
        attended = self._attended(kv_len, ratio)
        return (
            bs
            * local_heads
            * 2
            * attended
            * (self.config.head_dim + self.config.v_head_dim)
        )

    def _analytical_attention(self, bs, kv_len, ratio, device_type):
        gpu = gpu_map[device_type]
        flops = self._useful_flops(bs, kv_len, ratio)
        attended = self._attended(kv_len, ratio)
        kv_bytes_per_element = 1 if self.use_fp8_kv else 2
        # Sparse KV is shared by query heads. This analytical fallback is only
        # used when a measured H20 lookup row is unavailable (notably C1).
        bytes_moved = (
            bs
            * attended
            * (self.config.head_dim + self.config.v_head_dim)
            * kv_bytes_per_element
        )
        compute_s = flops / (gpu.fp16_tflops * 1e12)
        memory_s = bytes_moved / (gpu.mem_bw * 1e9)
        latency = max(compute_s, memory_s)
        mfu = flops / max(latency * gpu.fp16_tflops * 1e12, 1)
        return mfu, latency

    def _analytical_indexer(self, bs, kv_len, device_type):
        gpu = gpu_map[device_type]
        compressed = max(kv_len // 4, 1)
        flops = (
            2
            * bs
            * self.config.index_n_heads
            * self.config.index_head_dim
            * compressed
        )
        bytes_moved = (
            bs
            * compressed
            * self.config.index_head_dim
            * (1 if self.use_fp8_kv else 2)
        )
        latency = max(
            flops / (gpu.fp16_tflops * 1e12),
            bytes_moved / (gpu.mem_bw * 1e9),
        )
        return flops / max(latency * gpu.fp16_tflops * 1e12, 1), latency

    def decode_attn_core(self, bs, kv_len, kvcache_bytes, device_type):
        del kvcache_bytes
        total = 0.0
        print("{:<40} {:<10}".format("DSV4 attention model:", "H20 lookup"))
        print(
            "{:<40} {:<10}".format(
                "DSV4 local query heads:",
                self.config.num_attention_heads // self.tp_size,
            )
        )
        for ratio in (0, 4, 128):
            count = self.config.compress_ratio_counts.get(ratio, 0)
            if not count:
                continue
            if ratio == 0:
                mfu, latency = self._analytical_attention(
                    bs, kv_len, ratio, device_type
                )
                source = "analytical fallback"
            else:
                mfu, latency = get_dsa_decode_perf(
                    self.config,
                    bs,
                    kv_len,
                    device_type,
                    self.use_fp8_kv,
                    self.tp_size,
                    ratio,
                )
                source = "H20 lookup"
                if latency is None:
                    _, latency = self._analytical_attention(
                        bs, kv_len, ratio, device_type
                    )
                    source = "analytical fallback"
            indexer_latency = 0.0
            indexer_mfu = None
            if ratio == 4:
                indexer_mfu, indexer_latency = get_dsa_indexer_decode_perf(
                    self.config, bs, kv_len, device_type
                )
                if indexer_latency is None:
                    indexer_mfu, indexer_latency = self._analytical_indexer(
                        bs, kv_len, device_type
                    )
            layer_latency = latency + indexer_latency
            total += count * layer_latency
            print(
                f"DSA C{ratio or 1}: layers={count}, source={source}, "
                f"attn={latency * 1e6:.2f} us, useful_mfu={mfu:.3f}, "
                f"indexer={indexer_latency * 1e6:.2f} us"
            )
            if indexer_mfu is not None:
                print(f"DSA C4 indexer logits MFU: {indexer_mfu:.3f}")
        average = total / self.config.num_hidden_layers
        print(
            "{:<40} {:<10.2f}".format(
                "DSA weighted core latency/layer (us):", average * 1e6
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "DSA all-layer core latency (us):", total * 1e6
            )
        )
        return average

    def decode_attn_others(self, bs, device_type):
        local_heads = self.config.num_attention_heads // self.tp_size
        pieces = [
            (
                "Q_down_proj",
                get_gemm_mfu_and_latency(
                    bs,
                    self.config.hidden_size,
                    self.config.q_lora_rank,
                    device_type,
                    self.use_fp8_gemm,
                ),
            ),
            (
                "Q_up_proj",
                get_gemm_mfu_and_latency(
                    bs,
                    self.config.q_lora_rank,
                    local_heads * self.config.head_dim,
                    device_type,
                    self.use_fp8_gemm,
                ),
            ),
            (
                "KV_proj",
                get_gemm_mfu_and_latency(
                    bs,
                    self.config.hidden_size,
                    self.config.head_dim,
                    device_type,
                    self.use_fp8_gemm,
                ),
            ),
            (
                "O_down_proj",
                get_gemm_mfu_and_latency(
                    bs,
                    local_heads * self.config.v_head_dim,
                    self.config.o_lora_rank,
                    device_type,
                    self.use_fp8_gemm,
                ),
            ),
            (
                "O_up_proj",
                get_gemm_mfu_and_latency(
                    bs,
                    self.config.o_lora_rank,
                    self.config.hidden_size,
                    device_type,
                    self.use_fp8_gemm,
                ),
            ),
            (
                "Indexer_Q_proj",
                get_gemm_mfu_and_latency(
                    bs,
                    self.config.q_lora_rank,
                    self.config.index_n_heads * self.config.index_head_dim,
                    device_type,
                    self.use_fp8_gemm,
                ),
            ),
            (
                "Indexer_K_proj",
                get_gemm_mfu_and_latency(
                    bs,
                    self.config.hidden_size,
                    self.config.index_head_dim,
                    device_type,
                    self.use_fp8_gemm,
                ),
            ),
        ]
        for name, latency in pieces:
            print(f"{name + ' latency (us):':<40} {latency * 1e6:<10.2f}")
        return sum(latency for _, latency in pieces)

    def prefill_attn_core(self, seq_len, kvcache_bytes, device_type):
        raise NotImplementedError(
            "DSV4 prefill is not modeled by this decode patch; use --decode-only"
        )


class MLA(_MLACore):
    """MLA implementation built on the shared attention-core helpers."""

    def decode_attn_others(self, bs, device_type):
        # TP shards attention heads; hidden_size and lora ranks are NOT sharded
        tp_num_heads = self.config.num_attention_heads // self.tp_size
        q_down_proj = get_gemm_mfu_and_latency(
            m=bs,
            k=self.config.hidden_size,
            n=self.config.q_lora_rank,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        print("{:<40} {:<10.2f}".format("Q_down_proj latency (us):", q_down_proj * 1e6))

        q_up_proj = get_gemm_mfu_and_latency(
            m=bs,
            k=self.config.q_lora_rank,
            n=tp_num_heads * self.config.qk_head_dim,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        print("{:<40} {:<10.2f}".format("Q_up_proj latency (us):", q_up_proj * 1e6))

        kv_down_proj = get_gemm_mfu_and_latency(
            m=bs,
            k=self.config.hidden_size,
            n=self.config.kv_lora_rank + self.config.qk_rope_head_dim,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        print(
            "{:<40} {:<10.2f}".format("KV_down_proj latency (us):", kv_down_proj * 1e6)
        )

        bmm_q_wk = get_gemm_mfu_and_latency(
            m=bs,
            k=tp_num_heads * self.config.qk_nope_head_dim,
            n=self.config.kv_lora_rank,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        print("{:<40} {:<10.2f}".format("bmm_q_wk latency (us):", bmm_q_wk * 1e6))

        bmm_o_wv = get_gemm_mfu_and_latency(
            m=bs,
            k=tp_num_heads * self.config.kv_lora_rank,
            n=self.config.v_head_dim,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        print("{:<40} {:<10.2f}".format("bmm_o_wv latency (us):", bmm_o_wv * 1e6))

        o_proj = get_gemm_mfu_and_latency(
            m=bs,
            k=tp_num_heads * self.config.v_head_dim,
            n=self.config.hidden_size,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        print("{:<40} {:<10.2f}".format("O_proj latency (us):", o_proj * 1e6))
        return q_down_proj + q_up_proj + kv_down_proj + bmm_q_wk + bmm_o_wv + o_proj

    def prefill_attn_core(self, seq_len, kvcache_bytes, device_type):
        gpu = gpu_map[device_type]
        attn_core_gflops = self.get_attn_core_gflops_noabsorb(1, seq_len)
        attn_core_mfu = get_attn_prefill_mfu(self.config, seq_len, device_type, self.tp_size)
        attn_core_time = (
            seq_len
            * attn_core_gflops
            / 1.8
            / (gpu.fp16_tflops * TFLOPS_TO_GFLOPS * attn_core_mfu)
        )
        kv_load_time = (
            kvcache_bytes
            * seq_len
            / self.config.num_hidden_layers
            / 1024
            / 1024
            / 1024
            / gpu.mem_bw
        )

        print("{:<40} {:<10.2f}".format("Attn core MFU:", attn_core_mfu))
        print(
            "{:<40} {:<10.2f}".format("Attn core latency (us):", attn_core_time * 1e6)
        )
        print("{:<40} {:<10.2f}".format("KV loading latency (us):", kv_load_time * 1e6))

        return max(attn_core_time, kv_load_time)


def create_attention(
    config, use_fp8_gemm, use_fp8_kv, tp_size, prefill_mfu=None
):
    if config.attn_type == "MHA/GQA":
        return MHA(config, use_fp8_gemm, use_fp8_kv, tp_size, prefill_mfu)
    elif config.attn_type == "MLA":
        return MLA(config, use_fp8_gemm, use_fp8_kv, tp_size)
    elif config.attn_type == "DSA":
        return DSA(config, use_fp8_gemm, use_fp8_kv, tp_size)
