import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from config.model_config import ModelConfig
from layers.attn import DSA
from layers.moe import MoE
from mfu.mfu import (
    get_dsa_decode_perf,
    get_dsa_indexer_decode_perf,
    get_groupedgemm_decode_perf,
)
from models.model import get_dsv4_runtime_layer_latency_us


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "tests" / "fixtures" / "deepseek_v4_pro.json"


def read_csv_row(path, **expected):
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if all(row[key] == str(value) for key, value in expected.items()):
                return row
    raise AssertionError(f"No matching row in {path}: {expected}")


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
        row = read_csv_row(
            ROOT / "bench_data/dsa/decode/h20/attn-64-512-c4.csv",
            kv_dtype="fp8",
            batch_size=16,
            kv_len=40960,
        )
        with contextlib.chdir(ROOT):
            mfu, latency = get_dsa_decode_perf(
                self.config, 16, 40960, "H20", True, 8, 4
            )
        self.assertAlmostEqual(latency * 1e6, float(row["latency_us"]), places=3)
        self.assertAlmostEqual(mfu, float(row["mfu"]) / 4, places=6)

    def test_indexer_lookup(self):
        row = read_csv_row(
            ROOT
            / "bench_data/dsa/decode/h20/indexer-64-128-topk1024.csv",
            batch_size=16,
            kv_len=40960,
        )
        with contextlib.chdir(ROOT):
            mfu, latency = get_dsa_indexer_decode_perf(
                self.config, 16, 40960, "H20"
            )
        self.assertAlmostEqual(
            latency * 1e6, float(row["total_latency_us"]), places=3
        )
        self.assertAlmostEqual(mfu, float(row["logits_mfu"]), places=6)

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

    def test_mxfp4_grouped_gemm_lookup_returns_direct_latency(self):
        fields = (
            "num_experts,num_gpus,num_local_experts,topk,hidden_size,"
            "intermediate_size,batch_size_per_gpu,tokens_per_expert,"
            "up_proj_us,up_mfu,down_proj_us,down_mfu,total_latency_us,"
            "total_mfu,kernel_kind,backend,activation_dtype,weight_dtype,"
            "mfu_peak_tflops,tp_size,ep_size,tune_max_num_tokens,"
            "execution_mode,active_experts,max_tokens_per_expert,"
            "routing_mode,swiglu_limit"
        ).split(",")
        row = dict.fromkeys(fields, "")
        row.update(
            num_experts=384,
            num_gpus=8,
            num_local_experts=384,
            topk=6,
            hidden_size=7168,
            intermediate_size=384,
            batch_size_per_gpu=16,
            tokens_per_expert=0,
            up_proj_us=214.0,
            up_mfu=0.033,
            down_proj_us=107.5,
            down_mfu=0.033,
            total_latency_us=321.5,
            total_mfu=0.033,
            kernel_kind="cutlass_grouped_gemm_pair",
            backend="flashinfer_mxfp4_sm90",
            activation_dtype="bf16",
            weight_dtype="mxfp4_e2m1",
            tp_size=8,
            ep_size=1,
            tune_max_num_tokens=16,
            execution_mode="eager",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "bench_data/grouped_gemm/decode/h20/data.csv"
            path.parent.mkdir(parents=True)
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                stale_graph_row = dict(row)
                stale_graph_row["execution_mode"] = "graph"
                stale_graph_row["total_latency_us"] = 999.0
                writer.writerow(stale_graph_row)
                writer.writerow(row)
            with contextlib.chdir(root):
                bf16 = get_groupedgemm_decode_perf(
                    self.config, 16, "H20", 8, False, 8
                )
                fp8_flag = get_groupedgemm_decode_perf(
                    self.config, 16, "H20", 8, True, 8
                )
        self.assertAlmostEqual(bf16["total_latency_s"] * 1e6, 321.5)
        self.assertEqual(bf16["total_latency_s"], fp8_flag["total_latency_s"])
        self.assertEqual(bf16["weight_dtype"], "mxfp4_e2m1")

    def test_production_nsys_graph_row_wins_at_the_same_batch_size(self):
        fields = (
            "num_experts,num_gpus,num_local_experts,topk,hidden_size,"
            "intermediate_size,batch_size_per_gpu,tokens_per_expert,"
            "up_proj_us,up_mfu,down_proj_us,down_mfu,total_latency_us,"
            "total_mfu,kernel_kind,backend,activation_dtype,weight_dtype,"
            "mfu_peak_tflops,tp_size,ep_size,tune_max_num_tokens,"
            "execution_mode,active_experts,max_tokens_per_expert,"
            "routing_mode,swiglu_limit"
        ).split(",")
        base = dict.fromkeys(fields, "")
        base.update(
            num_experts=384,
            num_gpus=8,
            num_local_experts=384,
            topk=6,
            hidden_size=7168,
            intermediate_size=384,
            batch_size_per_gpu=16,
            tokens_per_expert=0,
            backend="flashinfer_mxfp4_sm90",
            activation_dtype="bf16",
            weight_dtype="mxfp4_e2m1",
            tp_size=8,
            ep_size=1,
            tune_max_num_tokens=16,
        )
        eager = dict(base)
        eager.update(
            up_proj_us=230.273,
            up_mfu=0.031,
            down_proj_us=195.873,
            down_mfu=0.018,
            total_latency_us=426.146,
            total_mfu=0.025,
            kernel_kind="cutlass_grouped_gemm_pair_pure_ffi",
            execution_mode="eager",
        )
        production = dict(base)
        production.update(
            up_proj_us=147.767,
            up_mfu=0.04833,
            down_proj_us=131.826,
            down_mfu=0.027087,
            total_latency_us=279.593,
            total_mfu=0.038315,
            kernel_kind="cutlass_grouped_gemm_pair_production_nsys_graph",
            execution_mode="graph",
            routing_mode="production_sglang_dsv4",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "bench_data/grouped_gemm/decode/h20/data.csv"
            path.parent.mkdir(parents=True)
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(eager)
                writer.writerow(production)
            with contextlib.chdir(root):
                result = get_groupedgemm_decode_perf(
                    self.config, 16, "H20", 8, False, 8
                )
        self.assertAlmostEqual(result["total_latency_s"] * 1e6, 279.593)
        self.assertEqual(result["execution_mode"], "graph")
        self.assertEqual(
            result["kernel_kind"],
            "cutlass_grouped_gemm_pair_production_nsys_graph",
        )

    def test_checked_in_production_moe_lookup_has_all_calibrated_batches(self):
        expected = {
            16: (141.687, 126.484, 268.171),
            24: (185.393, 170.161, 355.554),
            26: (188.421, 172.144, 360.565),
            32: (216.053, 198.261, 414.314),
        }
        with contextlib.chdir(ROOT):
            for batch_size, (up_us, down_us, total_us) in expected.items():
                row = read_csv_row(
                    ROOT / "bench_data/grouped_gemm/decode/h20/data.csv",
                    num_experts=384,
                    num_gpus=8,
                    num_local_experts=384,
                    topk=6,
                    hidden_size=7168,
                    intermediate_size=384,
                    batch_size_per_gpu=batch_size,
                    kernel_kind="cutlass_grouped_gemm_pair_production_nsys_graph",
                    routing_mode="production_sglang_dsv4",
                )
                self.assertAlmostEqual(float(row["up_proj_us"]), up_us, places=3)
                self.assertAlmostEqual(float(row["down_proj_us"]), down_us, places=3)
                self.assertAlmostEqual(
                    float(row["total_latency_us"]), total_us, places=3
                )
                perf = get_groupedgemm_decode_perf(
                    self.config, batch_size, "H20", 8, False, 8
                )
                self.assertAlmostEqual(
                    perf["total_latency_s"] * 1e6, total_us, places=3
                )
                self.assertEqual(
                    perf["kernel_kind"],
                    "cutlass_grouped_gemm_pair_production_nsys_graph",
                )

    def test_mxfp4_grouped_gemm_latency_includes_its_weight_loading(self):
        perf = {
            "up_mfu": 0.033,
            "down_mfu": 0.033,
            "total_mfu": 0.033,
            "total_latency_s": 321.5e-6,
            "kernel_kind": "cutlass_grouped_gemm_pair",
            "backend": "flashinfer_mxfp4_sm90",
            "activation_dtype": "bf16",
            "weight_dtype": "mxfp4_e2m1",
            "execution_mode": "eager",
            "source": "test",
        }
        with mock.patch("layers.moe.get_groupedgemm_decode_perf", return_value=perf), mock.patch(
            "layers.moe.load_moe_weights_time", return_value=1.0
        ), mock.patch("layers.moe.get_gemm_perf", return_value=(0.1, 0.0)):
            with contextlib.redirect_stdout(io.StringIO()):
                latency = MoE(self.config, False, 8).decode_moe(16, "H20", 8)
        self.assertAlmostEqual(latency, 321.5e-6)


if __name__ == "__main__":
    unittest.main()
