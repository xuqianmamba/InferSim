import json
from collections import Counter


class ModelConfig:
    def __init__(
        self,
        config_path,
    ):
        d = dict()
        with open(config_path, "r") as f:
            d = json.load(f)

        root_model_type = d["model_type"]
        if root_model_type in ["qwen3_5", "qwen3_5_moe"]:
            d = d["text_config"]

        self.model_type = d["model_type"]
        self.is_qwen3_5_moe = (
            root_model_type == "qwen3_5_moe"
            and self.model_type == "qwen3_5_moe_text"
        )

        self.hidden_size = d["hidden_size"]
        self.num_hidden_layers = d["num_hidden_layers"]
        self.expert_dtype = d.get("expert_dtype")

        # Non-hybrid models use full attention in every hidden layer.
        self.num_full_attn_layers = self.num_hidden_layers
        self.num_linear_attn_layers = 0
        self.is_hybrid_linear = d.get("full_attention_interval") is not None
        if self.is_hybrid_linear:
            self.num_full_attn_layers = (
                self.num_hidden_layers // d["full_attention_interval"]
            )
            self.num_linear_attn_layers = (
                self.num_hidden_layers - self.num_full_attn_layers
            )
            self.linear_conv_kernel_dim = d["linear_conv_kernel_dim"]
            self.linear_key_head_dim = d["linear_key_head_dim"]
            self.linear_num_key_heads = d["linear_num_key_heads"]
            self.linear_value_head_dim = d["linear_value_head_dim"]
            self.linear_num_value_heads = d["linear_num_value_heads"]

        self.is_dsv4 = bool(d.get("compress_ratios")) or any(
            "DeepseekV4" in architecture
            for architecture in d.get("architectures", [])
        )
        self.attn_type = "MHA/GQA"
        if self.is_dsv4:
            self.attn_type = "DSA"
        elif "kv_lora_rank" in d:
            self.attn_type = "MLA"

        # attn
        self.attn_output_gate = d.get("attn_output_gate", False)
        if self.attn_type == "MHA/GQA":
            self.num_attention_heads = d["num_attention_heads"]
            self.num_key_value_heads = d["num_key_value_heads"]
            if "head_dim" in d:
                self.head_dim = d["head_dim"]
            else:
                self.head_dim = self.hidden_size // self.num_attention_heads
        elif self.attn_type == "MLA":
            self.q_lora_rank = d["q_lora_rank"]
            self.qk_nope_head_dim = d["qk_nope_head_dim"]
            self.qk_rope_head_dim = d["qk_rope_head_dim"]
            self.kv_lora_rank = d["kv_lora_rank"]
            self.num_attention_heads = d["num_attention_heads"]
            self.v_head_dim = d["v_head_dim"]
            self.index_topk = d.get("index_topk")
            self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        elif self.attn_type == "DSA":
            # DeepSeek-V4 stores the complete Q/K width in head_dim.  It
            # already includes the RoPE component (448 NoPE + 64 RoPE = 512
            # for V4-Pro); it must not be interpreted as 512 + 64.
            self.num_attention_heads = d["num_attention_heads"]
            self.num_key_value_heads = d.get("num_key_value_heads", 1)
            self.head_dim = d["head_dim"]
            self.qk_rope_head_dim = d["qk_rope_head_dim"]
            self.qk_nope_head_dim = self.head_dim - self.qk_rope_head_dim
            self.qk_head_dim = self.head_dim
            self.v_head_dim = d.get("v_head_dim", self.head_dim)
            self.q_lora_rank = d["q_lora_rank"]
            self.o_lora_rank = d.get("o_lora_rank", self.hidden_size)
            self.index_n_heads = d.get("index_n_heads", 64)
            self.index_head_dim = d.get("index_head_dim", 128)
            self.index_topk = d.get("index_topk", 1024)
            self.swa_window = d.get("sliding_window", 128)
            self.compress_ratios = tuple(d["compress_ratios"])
            if len(self.compress_ratios) != self.num_hidden_layers:
                raise ValueError(
                    "compress_ratios must contain one entry per hidden layer"
                )
            unsupported = set(self.compress_ratios) - {0, 4, 128}
            if unsupported:
                raise ValueError(
                    f"unsupported DeepSeek-V4 compression ratios: {unsupported}"
                )
            self.compress_ratio_counts = Counter(self.compress_ratios)

        # FFN/MoE
        self.is_moe = True
        if "num_routed_experts" in d:
            self.num_routed_experts = d["num_routed_experts"]
        elif "n_routed_experts" in d:
            self.num_routed_experts = d["n_routed_experts"]
        elif "num_experts" in d:
            self.num_routed_experts = d["num_experts"]
        else:
            self.is_moe = False
            self.num_routed_experts = 1

        if self.is_moe:
            self.num_experts_per_tok = d["num_experts_per_tok"]
            self.intermediate_size = d["moe_intermediate_size"]
            self.shared_expert_intermediate_size = d.get(
                "shared_expert_intermediate_size"
            )
            if self.shared_expert_intermediate_size is not None:
                if self.shared_expert_intermediate_size % self.intermediate_size != 0:
                    raise ValueError(
                        "InferSim currently requires shared_expert_intermediate_size "
                        "to be a multiple of moe_intermediate_size"
                    )
                self.num_shared_experts = (
                    self.shared_expert_intermediate_size // self.intermediate_size
                )
            else:
                self.num_shared_experts = d.get(
                    "num_shared_experts", d.get("n_shared_experts", 0)
                )
                self.shared_expert_intermediate_size = (
                    self.num_shared_experts * self.intermediate_size
                )
        else:
            self.num_experts_per_tok = 1
            self.intermediate_size = d["intermediate_size"]
            self.num_shared_experts = 0
            self.shared_expert_intermediate_size = 0
