import contextlib
import io
import unittest
from pathlib import Path

from config.model_config import ModelConfig
from layers.attn import DSA
from mfu.mfu import get_dsa_decode_perf, get_dsa_indexer_decode_perf
from models.model import get_dsv4_runtime_layer_latency_us


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "tests" / "fixtures" / "deepseek_v4_pro.json"


class DSV4SimulatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ModelConfig(CONFIG)

    def test_config_shape_and_layer_mix(self):
        self.assertEqual(self.config.attn_type, "DSA")
        self.assertEqual(self.config.head_dim, 512)
        self.assertEqual(self.config.qk_nope_head_dim, 448)
        self.assertEqual(self.config.compress_ratio_counts[4], 30)
        self.assertEqual(self.config.compress_ratio_counts[128], 31)
        self.assertEqual(self.config.compress_ratio_counts[0], 0)
        self.assertEqual(self.config.extra_compress_ratios, (0,))
        self.assertEqual(self.config.num_routed_experts, 384)
        self.assertEqual(self.config.num_shared_experts, 1)

    def test_tp8_uses_h64_latency_and_quarter_useful_mfu(self):
        with contextlib.chdir(ROOT):
            mfu, latency = get_dsa_decode_perf(
                self.config, 16, 40960, "H20", True, 8, 4
            )
        self.assertAlmostEqual(latency * 1e6, 67.424, places=3)
        self.assertAlmostEqual(mfu, 0.242 / 4, places=6)

    def test_indexer_lookup(self):
        with contextlib.chdir(ROOT):
            mfu, latency = get_dsa_indexer_decode_perf(
                self.config, 16, 40960, "H20"
            )
        self.assertAlmostEqual(latency * 1e6, 44.512, places=3)
        self.assertAlmostEqual(mfu, 0.275, places=6)

    def test_weighted_decode_core_reports_layer_mix(self):
        attention = DSA(self.config, True, True, 8)
        output = io.StringIO()
        with contextlib.chdir(ROOT), contextlib.redirect_stdout(output):
            latency = attention.decode_attn_core(16, 40960, 0, "H20")
        self.assertGreater(latency, 0)
        text = output.getvalue()
        self.assertIn("DSA C4: layers=30", text)
        self.assertIn("DSA C128: layers=31", text)
        self.assertNotIn("DSA C1:", text)

    def test_runtime_layer_calibration_uses_dsv4_layer_mix(self):
        latency_us = get_dsv4_runtime_layer_latency_us(
            self.config,
            c4_latency_us=636.574,
            c128_latency_us=573.050,
        )
        expected = (30 * 636.574 + 31 * 573.050) / 61
        self.assertAlmostEqual(latency_us, expected, places=6)


if __name__ == "__main__":
    unittest.main()
