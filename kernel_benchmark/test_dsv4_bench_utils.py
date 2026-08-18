import json
import tempfile
import unittest
from pathlib import Path

from dsv4_bench_utils import (
    aligned_topk,
    attended_lengths,
    compressed_length,
    load_dsv4_shape,
    physical_page_size,
)


class DSV4BenchUtilsTest(unittest.TestCase):
    def test_pro_tp8_dp4_attention_shape(self):
        config = {
            "num_attention_heads": 128,
            "head_dim": 512,
            "qk_rope_head_dim": 64,
            "v_head_dim": 512,
            "index_n_heads": 64,
            "index_head_dim": 128,
            "index_topk": 1024,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            shape = load_dsv4_shape(path, attention_tp_size=2)
        self.assertEqual(shape["local_heads"], 64)
        self.assertEqual(shape["head_dim"], 512)
        self.assertEqual(shape["index_heads"], 64)

    def test_compressed_and_attended_lengths(self):
        self.assertEqual(compressed_length(40960, 4), 10240)
        self.assertEqual(compressed_length(40960, 128), 320)
        self.assertEqual(attended_lengths(40960, 4, 128, 1024), (128, 1024))
        self.assertEqual(attended_lengths(40960, 128, 128, 1024), (128, 320))
        self.assertEqual(physical_page_size(4), 64)
        self.assertEqual(physical_page_size(128), 2)
        self.assertEqual(aligned_topk(320), 320)
        self.assertEqual(aligned_topk(321), 384)


if __name__ == "__main__":
    unittest.main()
