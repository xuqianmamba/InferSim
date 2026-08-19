import json
import tempfile
import unittest
from pathlib import Path

from flashinfer_dsv4_mxfp4_moe_decode import load_shape, next_power_of_2


class DSV4MoeBenchTest(unittest.TestCase):
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
