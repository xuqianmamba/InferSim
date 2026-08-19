import json
import tempfile
import unittest
from pathlib import Path

from flashinfer_dsv4_mxfp4_moe_decode import (
    extract_grouped_gemm_samples,
    is_cutlass_grouped_gemm_kernel,
    load_shape,
    next_power_of_2,
)


GROUPED_GEMM_NAME = (
    "void cutlass::device_kernel<cutlass::gemm::kernel::"
    "GemmUniversal<cutlass::gemm::GroupProblemShape<cutlass::gemm::GroupScheduleMode::kDeviceOnly>>>"
)


class FakeEvent:
    def __init__(self, name: str, duration_us: float):
        self.name = name
        self.device_time_total = duration_us


class DSV4MoeBenchTest(unittest.TestCase):
    def test_extracts_two_ordered_grouped_gemms_per_replay(self):
        events = [
            FakeEvent(GROUPED_GEMM_NAME, 151.0),
            FakeEvent("doActivationKernel", 20.0),
            FakeEvent(GROUPED_GEMM_NAME, 135.0),
            FakeEvent("finalizeMoeRoutingKernel", 10.0),
            FakeEvent(GROUPED_GEMM_NAME, 149.0),
            FakeEvent(GROUPED_GEMM_NAME, 133.0),
        ]
        samples = extract_grouped_gemm_samples(events, repeats=2)
        self.assertEqual(samples["up_samples_us"], [151.0, 149.0])
        self.assertEqual(samples["down_samples_us"], [135.0, 133.0])
        self.assertEqual(samples["up_median_us"], 150.0)
        self.assertEqual(samples["down_median_us"], 134.0)

    def test_rejects_incomplete_grouped_gemm_capture(self):
        events = [FakeEvent(GROUPED_GEMM_NAME, 151.0)]
        with self.assertRaisesRegex(RuntimeError, "exactly two"):
            extract_grouped_gemm_samples(events, repeats=1)

    def test_grouped_gemm_name_filter_is_specific(self):
        self.assertTrue(is_cutlass_grouped_gemm_kernel(GROUPED_GEMM_NAME))
        self.assertFalse(
            is_cutlass_grouped_gemm_kernel(
                "void cutlass::device_kernel<non_grouped_gemm>"
            )
        )

    def test_production_tuning_bucket(self):
        self.assertEqual(next_power_of_2(16), 16)
        self.assertEqual(next_power_of_2(24), 32)
        self.assertEqual(next_power_of_2(26), 32)
        self.assertEqual(next_power_of_2(32), 32)

    def test_tp8dp1_shape(self):
        config = {
            "n_routed_experts": 384,
            "hidden_size": 7168,
            "moe_intermediate_size": 3072,
            "num_experts_per_tok": 6,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            shape = load_shape(path, world_size=8, tp_size=8)
        self.assertEqual(shape["num_local_experts"], 384)
        self.assertEqual(shape["intermediate"], 384)
        self.assertEqual(shape["topk"], 6)
        self.assertEqual(shape["tp_size"], 8)
        self.assertEqual(shape["ep_size"], 1)


if __name__ == "__main__":
    unittest.main()
