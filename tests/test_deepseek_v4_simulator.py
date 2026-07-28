import unittest
from pathlib import Path
from types import SimpleNamespace

from config.model_config import ModelConfig
from kernel_sim.dsv4 import DSV4BenchmarkData, MissingDSV4BenchmarkData
from models.deepseek_v4_model import DeepSeekV4Model

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "deepseek_v4"


class DeepSeekV4SimulatorTest(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig(FIXTURE_ROOT / "config.json")
        self.data_root = FIXTURE_ROOT / "bench_data" / "dsv4"

    def test_config_groups_base_layers_and_excludes_mtp(self):
        self.assertTrue(self.config.is_deepseek_v4)
        self.assertEqual(self.config.attn_type, "DSV4")
        self.assertEqual(self.config.nextn_compress_ratios, [0])
        self.assertEqual(self.config.attention_ratio_counts, {128: 2, 4: 1})
        self.assertEqual(
            self.config.dsv4_architecture_fingerprint,
            "2a60524de4a88a10",
        )

    def test_lookup_requires_an_exact_measured_shape(self):
        data = DSV4BenchmarkData(
            self.data_root,
            architecture_fingerprint=(self.config.dsv4_architecture_fingerprint),
        )
        self.assertEqual(
            data.attention_latency_us(
                phase="decode",
                device_type="H20",
                tp_size=16,
                backend="dsv4",
                attention_weight_dtype="fp8",
                kv_dtype="fp8_e4m3",
                compress_state_dtype="fp32",
                ratio=4,
                batch_size=4,
                q_len=1,
                past_len=12,
            ),
            20,
        )
        with self.assertRaises(MissingDSV4BenchmarkData):
            data.attention_latency_us(
                phase="decode",
                device_type="H20",
                tp_size=16,
                backend="dsv4",
                attention_weight_dtype="fp8",
                kv_dtype="fp8_e4m3",
                compress_state_dtype="fp32",
                ratio=4,
                batch_size=5,
                q_len=1,
                past_len=12,
            )

    def test_packaged_h20_tp8_measurement_is_readable(self):
        data = DSV4BenchmarkData(
            Path(__file__).parents[1] / "bench_data" / "dsv4",
            architecture_fingerprint="dc8a15770f7f8647",
        )
        self.assertAlmostEqual(
            data.attention_latency_us(
                phase="prefill",
                device_type="H20",
                tp_size=8,
                backend="dsv4",
                attention_weight_dtype="fp8",
                kv_dtype="fp8_e4m3",
                compress_state_dtype="fp32",
                ratio=128,
                batch_size=1,
                q_len=4096,
                past_len=0,
            ),
            3988.013406,
        )

    def test_model_sums_ratio_groups_mhc_and_moe(self):
        args = SimpleNamespace(
            device_type="H20",
            world_size=16,
            tp_size=16,
            num_nodes=2,
            dsv4_bench_data_dir=str(self.data_root),
            target_isl=8,
            target_osl=8,
            max_prefill_tokens=8,
            decode_bs=4,
            dsv4_decode_past_len=12,
            target_tpot=50,
            enable_deepep=False,
            enable_tbo=False,
        )
        model = DeepSeekV4Model(args, self.config)
        model._comm_time_us = lambda phase, tokens: 7
        timings = model._phase_latency_us("decode", 4, 1, 12)
        self.assertEqual(timings["attention"], {128: 20, 4: 20})
        self.assertEqual(timings["mhc"], 12)
        self.assertEqual(timings["moe"], 15)
        self.assertEqual(timings["comm"], 7)


if __name__ == "__main__":
    unittest.main()
