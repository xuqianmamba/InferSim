import csv
import os

from hardware.gpu import gpu_map


def _interpolate_exact_batch(
    rows, target_bs, kv_len, bs_index, kv_index, latency_index, mfu_index
):
    """Interpolate latency/MFU along KV length for an exact batch size."""
    same_bs = [row for row in rows if int(row[bs_index]) == target_bs]
    if not same_bs:
        return None
    same_bs.sort(key=lambda row: int(row[kv_index]))
    for row in same_bs:
        if int(row[kv_index]) == kv_len:
            return float(row[mfu_index]), float(row[latency_index]) / 1e6
    lower = [row for row in same_bs if int(row[kv_index]) < kv_len]
    upper = [row for row in same_bs if int(row[kv_index]) > kv_len]
    if not lower or not upper:
        return None
    lo, hi = lower[-1], upper[0]
    lo_kv, hi_kv = int(lo[kv_index]), int(hi[kv_index])
    weight = (kv_len - lo_kv) / (hi_kv - lo_kv)
    latency_us = float(lo[latency_index]) + weight * (
        float(hi[latency_index]) - float(lo[latency_index])
    )
    mfu = float(lo[mfu_index]) + weight * (
        float(hi[mfu_index]) - float(lo[mfu_index])
    )
    return mfu, latency_us / 1e6


def get_dsa_decode_perf(
    config, target_bs, kv_len, device_type, use_fp8_kv, tp_size, ratio
):
    """Return measured DSV4 sparse-attention performance.

    SGLang's FlashMLA backend pads 1..64 local query heads to 64 and 65..128
    to 128.  Latency therefore follows the padded-head benchmark, while the
    returned MFU is scaled back to useful (unpadded) local heads.
    """
    local_heads = config.num_attention_heads // tp_size
    if local_heads <= 64:
        kernel_heads = 64
    elif local_heads <= 128:
        kernel_heads = 128
    else:
        kernel_heads = local_heads
    file_name = (
        f"bench_data/dsa/decode/{device_type.lower()}/"
        f"attn-{kernel_heads}-{config.head_dim}-c{ratio}.csv"
    )
    if not os.path.exists(file_name):
        print(f"Warning: {file_name} not exists")
        return gpu_map[device_type].mfu, None
    kv_dtype = "fp8" if use_fp8_kv else "bf16"
    with open(file_name, newline="") as handle:
        rows = [
            row for row in csv.reader(handle)
            if row and row[0] != "dtype" and row[1] == kv_dtype
        ]
    if not rows:
        print(f"Warning: {file_name} has no {kv_dtype} rows")
        return gpu_map[device_type].mfu, None
    result = _interpolate_exact_batch(rows, target_bs, kv_len, 2, 3, 4, 5)
    if result is None:
        nearest = min(
            rows,
            key=lambda row: abs(int(row[2]) - target_bs)
            + abs(int(row[3]) - kv_len) / max(kv_len, 1),
        )
        return float(nearest[5]) * local_heads / kernel_heads, None
    mfu, latency = result
    return round(mfu * local_heads / kernel_heads, 6), latency


def get_dsa_indexer_decode_perf(config, target_bs, kv_len, device_type):
    file_name = (
        f"bench_data/dsa/decode/{device_type.lower()}/"
        f"indexer-{config.index_n_heads}-{config.index_head_dim}-"
        f"topk{config.index_topk}.csv"
    )
    if not os.path.exists(file_name):
        print(f"Warning: {file_name} not exists")
        return None, None
    with open(file_name, newline="") as handle:
        rows = [
            row for row in csv.reader(handle)
            if row and row[0] != "batch_size"
        ]
    # batch, original KV, compressed KV, logits, top-k, fused total, logits MFU
    result = _interpolate_exact_batch(rows, target_bs, kv_len, 0, 1, 5, 6)
    if result is None:
        return None, None
    mfu, latency = result
    return round(mfu, 6), latency


def get_attn_decode_mfu(config, target_bs, kv_len, device_type, use_fp8_kv, tp_size):
    mfu, _ = get_attn_decode_perf(
        config, target_bs, kv_len, device_type, use_fp8_kv, tp_size
    )
    return mfu


def get_attn_decode_perf(
    config, target_bs, kv_len, device_type, use_fp8_kv, tp_size
):
    """Return decode attention MFU and an applicable measured latency.

    An absolute latency is returned only for an exact batch-size match and an
    exact or interpolated KV length. Other cases return an MFU for analytical
    scaling and ``None`` for latency.
    """
    gpu = gpu_map[device_type]
    tp_num_heads = config.num_attention_heads // tp_size
    if config.attn_type == "MHA/GQA":
        tp_num_kv_heads = config.num_key_value_heads // tp_size
        head_dim = config.head_dim
        file_name = f"bench_data/mha/decode/{device_type.lower()}/{tp_num_heads}-{tp_num_kv_heads}-{head_dim}.csv"
    elif config.attn_type == "MLA":
        head_dim = f"{config.kv_lora_rank}-{config.qk_rope_head_dim}"
        file_name = f"bench_data/mla/decode/{device_type.lower()}/{tp_num_heads}-{head_dim}.csv"
    if not os.path.exists(file_name):
        print(f"Warning: {file_name} not exists")
        return gpu.mfu, None

    # row: dtype,kv_dtype,batch_size,kv_len,latency,mfu
    kv_dtype = "fp8" if use_fp8_kv else "bf16"
    rows = list()
    with open(file_name, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row[1] != kv_dtype:
                continue
            rows.append(row)

    if not rows:
        print(f"Warning: no {kv_dtype} rows found in {file_name}")
        return gpu.mfu, None

    exact_bs_rows = [row for row in rows if int(row[2]) == target_bs]
    if exact_bs_rows:
        exact_bs_rows.sort(key=lambda row: int(row[3]))

        for row in exact_bs_rows:
            if int(row[3]) == kv_len:
                return round(float(row[5]), 6), float(row[4]) / 1e6

        lower = [row for row in exact_bs_rows if int(row[3]) < kv_len]
        upper = [row for row in exact_bs_rows if int(row[3]) > kv_len]
        if lower and upper:
            lo = lower[-1]
            hi = upper[0]
            lo_kv = int(lo[3])
            hi_kv = int(hi[3])
            ratio = (kv_len - lo_kv) / (hi_kv - lo_kv)
            latency_us = float(lo[4]) + ratio * (float(hi[4]) - float(lo[4]))
            mfu = float(lo[5]) + ratio * (float(hi[5]) - float(lo[5]))
            return round(mfu, 6), latency_us / 1e6

        endpoint = (
            exact_bs_rows[0]
            if kv_len < int(exact_bs_rows[0][3])
            else exact_bs_rows[-1]
        )
        return round(float(endpoint[5]), 6), None

    nearest_bs = min(
        {int(row[2]) for row in rows}, key=lambda bs: abs(bs - target_bs)
    )
    same_bs_rows = [row for row in rows if int(row[2]) == nearest_bs]
    closest_row = min(same_bs_rows, key=lambda row: abs(int(row[3]) - kv_len))

    return round(float(closest_row[5]), 6), None


def get_attn_prefill_mfu(config, seq_len, device_type, tp_size):
    gpu = gpu_map[device_type]
    tp_num_heads = config.num_attention_heads // tp_size
    if config.attn_type == "MHA/GQA":
        tp_num_kv_heads = config.num_key_value_heads // tp_size
        head_dim = config.head_dim
        file_name = f"bench_data/mha/prefill/{device_type.lower()}/{tp_num_heads}-{tp_num_kv_heads}-{head_dim}.csv"
    elif config.attn_type == "MLA":
        head_dim = f"{config.qk_nope_head_dim}-{config.qk_rope_head_dim}"
        file_name = f"bench_data/mla/prefill/{device_type.lower()}/{tp_num_heads}-{head_dim}.csv"
    if not os.path.exists(file_name):
        print(f"Warning: {file_name} not exist.")
        return 0.9

    # row: dtype,seq_len,latecy_us,mfu
    rows = list()
    with open(file_name, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            rows.append(row)

    mfu = gpu.mfu
    # mfu_seq_len = 1
    for row in rows:
        sql = int(row[1])
        if sql <= seq_len:
            # mfu_seq_len = sql
            mfu = float(row[3])
        else:
            break

    return round(mfu, 6)


def get_groupedgemm_decode_perf(
    config, target_bs, device_type, num_gpus, use_fp8, tp_size=1
):
    """Return the best matching decode MoE benchmark row.

    Historical rows contain independently measured up/down GEMM MFUs. New
    MXFP4 rows contain direct timings of the two production CUTLASS grouped
    GEMMs (gate/up and down), with ``total_latency_s`` equal to their sum.
    That latency is authoritative: converting it to an MFU and back through a
    caller-selected FP16/FP8 peak is lossy and incorrect for W4A16 kernels.
    """
    gpu = gpu_map[device_type]
    file_name = f"bench_data/grouped_gemm/decode/{device_type.lower()}/data.csv"
    if not os.path.exists(file_name):
        print(f"warning: {file_name} not exists")
        return {
            "up_mfu": gpu.mfu,
            "down_mfu": gpu.mfu,
            "total_mfu": None,
            "total_latency_s": None,
            "kernel_kind": "",
            "backend": "",
            "activation_dtype": "",
            "weight_dtype": "",
            "execution_mode": "",
            "source": "default",
        }

    # The first 12 columns are the historical schema.  Optional columns after
    # that describe a production grouped-GEMM pair and are consumed by name.
    ep_size = num_gpus // tp_size
    expected_num_local_experts = config.num_routed_experts // ep_size
    rows = list()
    with open(file_name, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["num_experts"]) != config.num_routed_experts:
                continue
            if int(row["num_gpus"]) != num_gpus:
                continue
            if int(row["num_local_experts"]) != expected_num_local_experts:
                continue
            if int(row["topk"]) != config.num_experts_per_tok:
                continue
            if int(row["hidden_size"]) != config.hidden_size:
                continue
            if int(row["intermediate_size"]) != config.intermediate_size // tp_size:
                continue
            rows.append(row)

    if len(rows) == 0:
        print("Warning: grouped_gemm decode mfu not found, will use default mfu.")
        return {
            "up_mfu": gpu.mfu,
            "down_mfu": gpu.mfu,
            "total_mfu": None,
            "total_latency_s": None,
            "kernel_kind": "",
            "backend": "",
            "activation_dtype": "",
            "weight_dtype": "",
            "execution_mode": "",
            "source": "default",
        }

    expert_dtype = str(getattr(config, "expert_dtype", "")).lower()
    if expert_dtype in {"fp4", "mxfp4", "nvfp4"}:
        production_rows = [
            row
            for row in rows
            if row.get("backend") == "flashinfer_mxfp4_sm90"
            and row.get("weight_dtype") == "mxfp4_e2m1"
            and row.get("execution_mode") == "eager"
            and (not row.get("tp_size") or int(row["tp_size"]) == tp_size)
            and (not row.get("ep_size") or int(row["ep_size"]) == ep_size)
        ]
        if production_rows:
            rows = production_rows

    closest_row = min(
        rows, key=lambda row: abs(int(row["batch_size_per_gpu"]) - target_bs)
    )
    total_latency_us = closest_row.get("total_latency_us", "").strip()
    total_mfu = closest_row.get("total_mfu", "").strip()
    result = {
        "up_mfu": round(float(closest_row["up_mfu"]), 6),
        "down_mfu": round(float(closest_row["down_mfu"]), 6),
        "total_mfu": round(float(total_mfu), 6) if total_mfu else None,
        "total_latency_s": (
            float(total_latency_us) / 1e6 if total_latency_us else None
        ),
        "kernel_kind": closest_row.get("kernel_kind", ""),
        "backend": closest_row.get("backend", ""),
        "activation_dtype": closest_row.get("activation_dtype", ""),
        "weight_dtype": closest_row.get("weight_dtype", ""),
        "execution_mode": closest_row.get("execution_mode", ""),
        "source": file_name,
    }
    return result


def get_groupedgemm_decode_mfu(
    config, target_bs, device_type, num_gpus, use_fp8, tp_size=1
):
    """Compatibility wrapper for callers that only understand legacy MFUs."""
    perf = get_groupedgemm_decode_perf(
        config, target_bs, device_type, num_gpus, use_fp8, tp_size
    )

    return perf["up_mfu"], perf["down_mfu"]

def get_groupedgemm_prefill_mfu(config, seq_len, device_type, num_gpus, use_fp8, tp_size=1):
    gpu = gpu_map[device_type]
    file_name = f"bench_data/grouped_gemm/prefill/{device_type.lower()}/data.csv"
    if not os.path.exists(file_name):
        print(f"Warning: {file_name} not exists")
        return gpu.mfu, gpu.mfu

    # row: num_experts,num_gpus,num_local_experts,topk,hidden_size,intermediate_size,seq_len_per_gpu,tokens_per_expert,up_proj_us,up_mfu,down_proj_us,down_mfu
    ep_size = num_gpus // tp_size
    expected_num_local_experts = config.num_routed_experts // ep_size
    rows = list()
    with open(file_name, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if int(row[0]) != config.num_routed_experts:
                continue
            if int(row[1]) != num_gpus:
                continue
            if int(row[2]) != expected_num_local_experts:
                continue
            if int(row[3]) != config.num_experts_per_tok:
                continue
            if int(row[4]) != config.hidden_size:
                continue
            if int(row[5]) != config.intermediate_size // tp_size:
                continue
            rows.append(row)

    if len(rows) == 0:
        print("Warning: grouped_gemm prefill mfu not found, will use default mfu.")

    mfu1 = gpu.mfu
    mfu2 = gpu.mfu
    for row in rows:
        sql = int(row[6])
        if sql <= seq_len:
            mfu1 = float(row[9])
            mfu2 = float(row[11])
        else:
            break

    return round(mfu1, 6), round(mfu2, 6)


def get_gemm_mfu(device_type, m, k, n):
    gpu = gpu_map[device_type]
    file_name = f"bench_data/gemm/{device_type.lower()}/data.csv"
    if not os.path.exists(file_name):
        print(f"Warning: {file_name} not exists")
        return gpu.mfu

    mfu_k = 0
    mfu_n = 0
    dist = 1e9
    # row: m,k,n,latency_us,mfu
    rows = list()
    with open(file_name, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            k_ = int(row[1])
            n_ = int(row[2])
            if k_ < k or n_ < n:
                continue
            if (k - k_) ** 2 + (n - n_) ** 2 < dist:
                dist = (k - k_) ** 2 + (n - n_) ** 2
                mfu_k = k_
                mfu_n = n_
            rows.append(row)

    mfu = gpu.mfu
    matched = False
    for row in rows:
        m_ = int(row[0])
        k_ = int(row[1])
        n_ = int(row[2])
        if k_ == mfu_k and n_ == mfu_n and m_ <= m:
            mfu = float(row[4])
            matched = True

    if not matched:
        print(
            f"Warning: GEMM MFU not found for m={m}, k={k}, n={n}; "
            f"using default MFU {gpu.mfu}."
        )
    return round(mfu, 6)


def get_linear_attn_prefill_latency(config, seq_len, device_type):
    file_name = f"bench_data/gdn/prefill/{device_type.lower()}/{config.linear_conv_kernel_dim}-{config.linear_num_key_heads}-{config.linear_key_head_dim}-{config.linear_num_value_heads}-{config.linear_value_head_dim}.csv"
    if not os.path.exists(file_name):
        assert False, f"Error: {file_name} not exist."

    # seq_len,conv,kkt,tril,wu,chunk_gdn,chunk_o
    rows = list()
    with open(file_name, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            rows.append(row)
    idx = 0
    diff = 1e9
    for i in range(len(rows)):
        sq = int(rows[i][0])
        if abs(sq - seq_len) < diff:
            diff = abs(sq - seq_len)
            idx = i
    ratio = int(rows[idx][0]) / seq_len
    t = 0
    for i in range(1, len(rows[idx])):
        t += float(rows[idx][i]) / ratio
    return t  # latency in us


def get_linear_attn_decode_latency(config, batchsize, device_type):
    file_name = f"bench_data/gdn/decode/{device_type.lower()}/{config.linear_conv_kernel_dim}-{config.linear_num_key_heads}-{config.linear_key_head_dim}-{config.linear_num_value_heads}-{config.linear_value_head_dim}.csv"
    if not os.path.exists(file_name):
        assert False, f"Error: {file_name} not exist."

    # batchsize,conv,gdn_update
    rows = list()
    with open(file_name, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            rows.append(row)
    idx = 0
    diff = 1e9
    for i in range(len(rows)):
        bs = int(rows[i][0])
        if abs(bs - batchsize) < diff:
            diff = abs(bs - batchsize)
            idx = i
    ratio = int(rows[idx][0]) / batchsize
    t = 0
    for i in range(1, len(rows[idx])):
        t += float(rows[idx][i]) / ratio
    return t  # latency in us
