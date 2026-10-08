"""Validate the aggregation used in the public result tables."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_results import summarize


class ReportedResultTests(unittest.TestCase):
    def test_medians_and_pair_order(self):
        summary = summarize(ROOT / "results" / "reported_runs.csv")
        h1 = summary["H1"]
        throughput = h1["metrics"]["output_throughput_tok_s"]
        self.assertEqual(throughput["A"], 212.71)
        self.assertEqual(throughput["B"], 221.81)
        self.assertEqual(round(throughput["change_percent"], 2), 4.28)
        self.assertTrue(all(change > 0 for change in h1["paired_throughput_change_percent"]))
        self.assertEqual(h1["metrics"]["p99_itl_ms"]["B"], 29.73)

    def test_h0_tail_metric_is_preserved(self):
        h0 = summarize(ROOT / "results" / "reported_runs.csv")["H0"]
        self.assertEqual(h0["metrics"]["output_throughput_tok_s"]["B"], 269.91)
        self.assertGreater(h0["metrics"]["p99_itl_ms"]["change_percent"], 0)


if __name__ == "__main__":
    unittest.main()
