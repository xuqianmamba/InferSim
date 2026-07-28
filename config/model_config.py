import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionLayerGroup:
    """A consecutive run of DeepSeek-V4 attention layers."""

    compress_ratio: int
    num_layers: int


def _normalize_dtype(value):
    value = str(value).lower()
    if value in {"fp8", "e4m3", "e5m2", "float8_e4m3fn", "float8_e5m2"}:
        return "fp8"
    if value in {"bf16", "bfloat16"}:
        return "bf16"
    if value in {"fp4", "nvfp4", "mxfp4"}:
        return "fp4"
    return value


def get_deepseek_v4_architecture_fingerprint(config):
    """Return a stable key for benchmark rows belonging to one V4 variant."""
    if not config.is_deepseek_v4:
        raise ValueError(
            "DeepSeek-V4 architecture fingerprint requested for another model"
        )
    architecture = {
        "model_type": config.model_type,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "compress_ratios": config.compress_ratios,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "q_lora_rank": config.q_lora_rank,
        "o_lora_rank": config.o_lora_rank,
        "o_groups": config.num_output_groups,
        "qk_rope_head_dim": config.qk_rope_head_dim,
        "index_n_heads": config.index_n_heads,
        "index_head_dim": config.index_head_dim,
        "index_topk": config.index_topk,
        "sliding_window": config.sliding_window,
        "hc_mult": config.hc_mult,
        "hc_sinkhorn_iters": config.hc_sinkhorn_iters,
        "routed_experts": config.num_routed_experts,
        "shared_experts": config.num_shared_experts,
        "experts_per_tok": config.num_experts_per_tok,
        "moe_intermediate_size": config.intermediate_size,
        "attention_weight_dtype": config.attention_weight_dtype,
        "expert_dtype": config.expert_dtype,
    }
    serialized = json.dumps(architecture, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


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
        self.is_deepseek_v4 = self.model_type == "deepseek_v4"
        self.is_qwen3_5_moe = (
            root_model_type == "qwen3_5_moe" and self.model_type == "qwen3_5_moe_text"
        )

        self.hidden_size = d["hidden_size"]
        self.num_hidden_layers = d["num_hidden_layers"]

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

        self.attn_type = "MHA/GQA"
        if self.is_deepseek_v4:
            self.attn_type = "DSV4"
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
        elif self.attn_type == "DSV4":
            self.num_attention_heads = d["num_attention_heads"]
            self.num_key_value_heads = d["num_key_value_heads"]
            self.head_dim = d["head_dim"]
            self.q_lora_rank = d["q_lora_rank"]
            self.o_lora_rank = d["o_lora_rank"]
            self.num_output_groups = d["o_groups"]
            self.qk_rope_head_dim = d["qk_rope_head_dim"]
            self.qk_nope_head_dim = self.head_dim - self.qk_rope_head_dim
            self.index_topk = d["index_topk"]
            self.index_n_heads = d.get("index_n_heads", 64)
            self.index_head_dim = d.get("index_head_dim", 128)
            self.sliding_window = d.get("sliding_window", 128)
            self.hc_mult = d.get("hc_mult", 4)
            self.hc_sinkhorn_iters = d.get("hc_sinkhorn_iters", 20)
            quantization_config = d.get("quantization_config", {})
            self.attention_weight_dtype = _normalize_dtype(
                d.get(
                    "attention_weight_dtype",
                    quantization_config.get("fmt", d.get("torch_dtype", "bf16")),
                )
            )
            self.expert_dtype = _normalize_dtype(d.get("expert_dtype", "bf16"))

            all_compress_ratios = list(d["compress_ratios"])
            if len(all_compress_ratios) < self.num_hidden_layers:
                raise ValueError(
                    "DeepSeek-V4 compress_ratios must contain one entry for "
                    "each transformer layer"
                )
            self.compress_ratios = all_compress_ratios[: self.num_hidden_layers]
            self.nextn_compress_ratios = all_compress_ratios[self.num_hidden_layers :]
            invalid_ratios = set(self.compress_ratios).difference({0, 4, 128})
            if invalid_ratios:
                raise ValueError(
                    "Unsupported DeepSeek-V4 compression ratio(s): "
                    + ", ".join(str(ratio) for ratio in sorted(invalid_ratios))
                )
            self.attention_layer_groups = tuple(
                self._group_compress_ratios(self.compress_ratios)
            )
            self.attention_ratio_counts = {}
            for group in self.attention_layer_groups:
                self.attention_ratio_counts[group.compress_ratio] = (
                    self.attention_ratio_counts.get(group.compress_ratio, 0)
                    + group.num_layers
                )

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

        if self.is_deepseek_v4:
            self.dsv4_architecture_fingerprint = (
                get_deepseek_v4_architecture_fingerprint(self)
            )

    @staticmethod
    def _group_compress_ratios(compress_ratios):
        if not compress_ratios:
            return []
        groups = []
        current_ratio = compress_ratios[0]
        count = 0
        for ratio in compress_ratios:
            if ratio != current_ratio:
                groups.append(AttentionLayerGroup(current_ratio, count))
                current_ratio = ratio
                count = 0
            count += 1
        groups.append(AttentionLayerGroup(current_ratio, count))
        return groups
