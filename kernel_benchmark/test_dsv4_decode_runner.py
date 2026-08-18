import csv
import tempfile
import unittest
from pathlib import Path

from run_dsv4_h20_decode_bench import MATRIX_FILES, install_results


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
                        writer.writerow(
                            dict(
                                zip(
                                    fields,
                                    (
                                        384,
                                        8,
                                        384,
                                        6,
                                        7168,
                                        384,
                                        16,
                                        0,
                                        100,
                                        0.1,
                                        100,
                                        0.1,
                                    ),
                                )
                            )
                        )

            target = (
                repository
                / "bench_data"
                / "grouped_gemm"
                / "decode"
                / "h20"
                / "data.csv"
            )
            target.parent.mkdir(parents=True)
            fields = MATRIX_FILES["groupedgemm-decode-dsv4-tp8dp1.csv"]
            with target.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(dict(zip(fields, (96, 1, 96, 8, 2048, 1560, 16, 1, 1, 0.1, 1, 0.1))))
                writer.writerow(dict(zip(fields, (384, 8, 384, 6, 7168, 384, 24, 0, 999, 0.9, 999, 0.9))))

            install_results(output, repository)

            with target.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["num_experts"], "96")
            self.assertEqual(rows[1]["num_experts"], "384")
            self.assertEqual(rows[1]["batch_size_per_gpu"], "16")
            self.assertEqual(rows[1]["up_proj_us"], "100")


if __name__ == "__main__":
    unittest.main()
