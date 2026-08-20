"""Capture and replay real SGLang MXFP4 MoE calls under Nsys.

Enable this module by prepending its directory to ``PYTHONPATH`` and setting
``SGL_MOE_CAPTURE=1``.  It wraps FlashInfer's public fused-MoE entrypoint but
does not change the normal result.  Once it has observed one decode step worth
of matching calls, it replays those calls back-to-back inside a
cudaProfilerStart/Stop range.  The replay uses the real SGLang hidden states,
routing tensors, checkpoint weights, quantization scales, TP/EP topology, and
tuning bucket.  Nsys can then select only the two CUTLASS grouped-GEMM kernels.

This capture is intentionally for an eager SGLang debug run.  Python hooks do
not execute on every CUDA Graph replay, so start the server with
``--disable-cuda-graph`` while collecting the calibration trace.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path


def _enabled(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes"}


if _enabled("SGL_MOE_CAPTURE"):
    import torch
    import flashinfer.fused_moe as fused_moe

    _original_cutlass_fused_moe = fused_moe.cutlass_fused_moe
    _captured_calls: list[dict] = []
    _capture_complete = False

    _target_bs = int(os.environ.get("SGL_MOE_CAPTURE_BS", "16"))
    _target_tp_rank = int(os.environ.get("SGL_MOE_CAPTURE_TP_RANK", "0"))
    _calls_per_step = int(os.environ.get("SGL_MOE_CAPTURE_CALLS", "61"))
    _warmup_replays = int(os.environ.get("SGL_MOE_CAPTURE_WARMUP", "1"))
    _formal_replays = int(os.environ.get("SGL_MOE_CAPTURE_REPEATS", "10"))
    _output_dir = Path(
        os.environ.get("SGL_MOE_CAPTURE_DIR", "/tmp/sglang_moe_capture")
    )

    def _clone_runtime_call(kwargs: dict) -> dict:
        replay = dict(kwargs)
        replay["input"] = kwargs["input"].detach().clone()
        replay["token_selected_experts"] = (
            kwargs["token_selected_experts"].detach().clone()
        )
        replay["token_final_scales"] = kwargs["token_final_scales"].detach().clone()
        replay["output"] = torch.empty_like(kwargs["output"])
        # Weight, scale, bias, and limit tensors deliberately remain references
        # to the tensors loaded by the production SGLang model.
        return replay

    def _routing_summary(calls: list[dict]) -> dict:
        hist = torch.zeros(384, dtype=torch.int64)
        maximum = 0
        active_counts = []
        for call in calls:
            ids = call["token_selected_experts"].detach().to("cpu", torch.int64)
            counts = torch.bincount(ids.reshape(-1), minlength=384)
            hist += counts
            active_counts.append(int((counts > 0).sum()))
            maximum = max(maximum, int(counts.max()))
        return {
            "calls": len(calls),
            "batch_size": _target_bs,
            "topk": int(calls[0]["token_selected_experts"].shape[1]),
            "active_experts_per_call": active_counts,
            "max_tokens_per_expert": maximum,
            "aggregate_expert_histogram": hist.tolist(),
            "warmup_replays": _warmup_replays,
            "formal_replays": _formal_replays,
            "expected_grouped_gemm_pairs": len(calls) * _formal_replays,
        }

    def _run_isolated_replay() -> None:
        global _capture_complete
        if _capture_complete:
            return
        _capture_complete = True
        _output_dir.mkdir(parents=True, exist_ok=True)
        try:
            torch.cuda.synchronize()
            for _ in range(_warmup_replays):
                for replay_kwargs in _captured_calls:
                    _original_cutlass_fused_moe(**replay_kwargs)
            torch.cuda.synchronize()

            cudart = torch.cuda.cudart()
            cudart.cudaProfilerStart()
            for _ in range(_formal_replays):
                for replay_kwargs in _captured_calls:
                    _original_cutlass_fused_moe(**replay_kwargs)
            torch.cuda.synchronize()
            cudart.cudaProfilerStop()

            metadata = _routing_summary(_captured_calls)
            metadata.update(
                {
                    "status": "complete",
                    "tp_rank": _target_tp_rank,
                    "device": torch.cuda.current_device(),
                    "execution": "isolated_real_sglang_moe_replay",
                }
            )
            (_output_dir / "capture_complete.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
            if _enabled("SGL_MOE_CAPTURE_SAVE_TENSORS", "1"):
                torch.save(
                    [
                        {
                            "hidden_states": call["input"].detach().cpu(),
                            "topk_ids": call["token_selected_experts"].detach().cpu(),
                            "topk_weights": call["token_final_scales"].detach().cpu(),
                        }
                        for call in _captured_calls
                    ],
                    _output_dir / "runtime_inputs.pt",
                )
            print(
                "SGL_MOE_CAPTURE_COMPLETE "
                f"calls={len(_captured_calls)} repeats={_formal_replays} "
                f"pairs={metadata['expected_grouped_gemm_pairs']} "
                f"dir={_output_dir}",
                flush=True,
            )
        except Exception as error:
            failure = {
                "status": "failed",
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
            (_output_dir / "capture_failed.json").write_text(
                json.dumps(failure, indent=2), encoding="utf-8"
            )
            print(f"SGL_MOE_CAPTURE_FAILED: {error!r}", file=sys.stderr, flush=True)
            traceback.print_exc()

    def _capturing_cutlass_fused_moe(*args, **kwargs):
        global _captured_calls
        result = _original_cutlass_fused_moe(*args, **kwargs)
        if _capture_complete or args:
            return result
        input_tensor = kwargs.get("input")
        if input_tensor is None or input_tensor.shape[0] != _target_bs:
            return result
        if int(kwargs.get("tp_rank", -1)) != _target_tp_rank:
            return result
        if torch.cuda.is_current_stream_capturing():
            return result

        _captured_calls.append(_clone_runtime_call(kwargs))
        if len(_captured_calls) == _calls_per_step:
            _run_isolated_replay()
        return result

    fused_moe.cutlass_fused_moe = _capturing_cutlass_fused_moe
    print(
        "SGL_MOE_CAPTURE_ARMED "
        f"bs={_target_bs} tp_rank={_target_tp_rank} calls={_calls_per_step} "
        f"repeats={_formal_replays} dir={_output_dir}",
        flush=True,
    )
