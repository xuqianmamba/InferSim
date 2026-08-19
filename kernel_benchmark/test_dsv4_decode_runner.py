import csv
import tempfile
import unittest
from pathlib import Path

from run_dsv4_h20_decode_bench import (
    LEGACY_MOE_FIELDS,
    MATRIX_FILES,
    install_results,
)


class DSV4DecodeRunnerTest(unittest.TestCase):
    def test_install_replaces_only_matching_dsv4_moe_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            repository = root / "repository"
            output.mkdir()

            for name, fields in MATRIX_FILES.items():
                with (output / name).open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    if name.startswith("groupedgemm-"):
                        row = dict.fromkeys(fields, "")
                        row.update(
                            dict(
                                zip(
                                    LEGACY_MOE_FIELDS,
                                    (384, 8, 384, 6, 7168, 384, 16, 0, 100, 0.1, 100, 0.1),
                                )
                            )
                        )
                        row.update(
                            up_proj_us=60,
                            up_mfu=0.1,
                            down_proj_us=40,
                            down_mfu=0.1,
                            total_latency_us=100,
                            total_mfu=0.1,
                            kernel_kind="cutlass_grouped_gemm_pair",
                            backend="flashinfer_mxfp4_sm90",
                            activation_dtype="bf16",
                            weight_dtype="mxfp4_e2m1",
                            tp_size=8,
                            ep_size=1,
                            tune_max_num_tokens=16,
                            execution_mode="graph",
                        )
                        writer.writerow(row)

            target = (
                repository
                / "bench_data"
                / "grouped_gemm"
                / "decode"
                / "h20"
                / "data.csv"
            )
            target.parent.mkdir(parents=True)
            fields = LEGACY_MOE_FIELDS
            with target.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    dict(zip(fields, (96, 1, 96, 8, 2048, 1560, 16, 1, 1, 0.1, 1, 0.1)))
                )
                writer.writerow(
                    dict(zip(fields, (384, 8, 384, 6, 7168, 384, 24, 0, 999, 0.9, 999, 0.9)))
                )

            install_results(output, repository)

            with target.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["num_experts"], "96")
            self.assertEqual(rows[1]["batch_size_per_gpu"], "24")
            self.assertEqual(rows[1]["up_proj_us"], "999")
            self.assertEqual(rows[2]["num_experts"], "384")
            self.assertEqual(rows[2]["batch_size_per_gpu"], "16")
            self.assertEqual(rows[2]["up_proj_us"], "60")
            self.assertEqual(rows[2]["down_proj_us"], "40")
            self.assertEqual(rows[2]["total_latency_us"], "100")
            self.assertEqual(rows[2]["backend"], "flashinfer_mxfp4_sm90")
            self.assertEqual(rows[2]["tune_max_num_tokens"], "16")
            self.assertEqual(rows[0]["total_latency_us"], "")
            self.assertTrue(
                (output / "grouped_gemm-decode-h20-data.before.csv").is_file()
            )


if __name__ == "__main__":
    unittest.main()
