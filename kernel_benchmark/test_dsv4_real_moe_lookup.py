from __future__ import annotations

import signal
import sqlite3
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from run_dsv4_real_moe_lookup import (
    build_lookup_row,
    grouped_gemm_durations,
    parse_int_list,
    summarize_pairs,
    terminate_process_group,
)


class RealMoeLookupTest(unittest.TestCase):
    @mock.patch("run_dsv4_real_moe_lookup.os.killpg", create=True)
    def test_cleanup_signals_group_even_after_nsys_leader_exited(self, killpg):
        process = mock.Mock()
        process.pid = 1234
        process.poll.return_value = 0
        terminate_process_group(process)
        killpg.assert_called_once_with(1234, signal.SIGTERM)

    def test_parse_int_list(self):
        self.assertEqual(parse_int_list("16,24, 32"), [16, 24, 32])
        with self.assertRaises(Exception):
            parse_int_list("16,16")

    def test_timeline_query_and_pairing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.sqlite"
            connection = sqlite3.connect(path)
            connection.executescript(
                """
                CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                    start INTEGER, end INTEGER, demangledName INTEGER
                );
                INSERT INTO StringIds VALUES
                    (1, 'void cutlass::device_kernel<GemmUniversal<GroupProblemShape>>'),
                    (2, 'unrelated_kernel');
                """
            )
            # Insert out of order; the query must restore launch order.
            connection.executemany(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?)",
                [
                    (400_000, 540_000, 1),
                    (100_000, 220_000, 1),
                    (250_000, 380_000, 1),
                    (50_000, 60_000, 2),
                    (600_000, 750_000, 1),
                ],
            )
            connection.commit()
            connection.close()

            durations = grouped_gemm_durations(path)
            self.assertEqual(durations, [120.0, 130.0, 140.0, 150.0])
            summary = summarize_pairs(durations, expected_pairs=2)
            self.assertAlmostEqual(summary["up_mean_us"], 130.0)
            self.assertAlmostEqual(summary["down_mean_us"], 140.0)
            self.assertAlmostEqual(summary["total_mean_us"], 270.0)

    def test_pair_count_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "kernel count mismatch"):
            summarize_pairs([120.0, 130.0, 140.0], expected_pairs=2)

    def test_lookup_row_uses_two_gemm_mean_and_real_routing_metadata(self):
        metadata = {
            "active_experts_per_call": [60, 62, 64],
            "max_tokens_per_expert": 12,
        }
        summary = {"up_mean_us": 145.447, "down_mean_us": 129.770}
        row = build_lookup_row(
            16,
            metadata,
            summary,
            peak_tflops=148.0,
            hidden_size=7168,
            local_intermediate_size=384,
            num_experts=384,
            topk=6,
            tp_size=8,
        )
        self.assertEqual(row["total_latency_us"], "275.217")
        self.assertEqual(row["kernel_kind"], "cutlass_grouped_gemm_pair_production_nsys_graph")
        self.assertEqual(row["routing_mode"], "production_sglang_dsv4")
        self.assertEqual(row["active_experts"], "62.000")
        self.assertEqual(row["max_tokens_per_expert"], 12)
        self.assertEqual(row["tune_max_num_tokens"], 16)
        self.assertAlmostEqual(float(row["up_mfu"]), 0.04910, places=4)
        self.assertAlmostEqual(float(row["down_mfu"]), 0.02752, places=4)


if __name__ == "__main__":
    unittest.main()
