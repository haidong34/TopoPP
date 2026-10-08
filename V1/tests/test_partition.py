"""Compare the planner with an independent recursive oracle."""

import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from common import load_gpu_times
from partition import default_partition, solve_partition


def compositions(layers, stages):
    if stages == 1:
        yield (layers,)
        return
    for count in range(1, layers - stages + 2):
        for tail in compositions(layers - count, stages - 1):
            yield (count, *tail)


class PartitionTests(unittest.TestCase):
    def test_random_costs_against_recursive_oracle(self):
        random_source = random.Random(321)
        for layers in range(4, 11):
            for stages in range(1, min(layers, 4) + 1):
                times = [random_source.uniform(0.1, 4) for _ in range(stages)]
                oracle = min(
                    max(count * latency for count, latency in zip(counts, times))
                    for counts in compositions(layers, stages)
                )
                result = solve_partition(layers, times)
                self.assertAlmostEqual(result["bottleneck"], oracle)
                self.assertEqual(result["ranges"][-1][1], layers)

    def test_reported_h1_partition(self):
        profile = SCRIPTS.parent / "examples" / "h1_proxy_ratios.json"
        _, times, unit = load_gpu_times(json.loads(profile.read_text()))
        self.assertEqual(unit, "relative")
        self.assertEqual(solve_partition(28, times)["partition"], [8, 8, 4, 8])

    def test_invalid_latencies_and_layer_counts(self):
        for value in (0, -1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                solve_partition(4, [1, value])
        for layers, times in ((0, [1]), (1, [1, 2]), (4, [])):
            with self.assertRaises(ValueError):
                solve_partition(layers, times)

    def test_default_remainder(self):
        self.assertEqual(default_partition(28, 4), [7, 7, 7, 7])
        self.assertEqual(default_partition(29, 4), [7, 7, 8, 7])

    def test_cli_and_exhaustive_verifier(self):
        profile = SCRIPTS.parent / "examples" / "h1_proxy_ratios.json"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "new" / "partition.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "partition.py"),
                    "--profile",
                    str(profile),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(output.read_text())["unit"], "relative")
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / "verify_partition.py"), "--profile", str(profile)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Verified 2925 candidates", result.stdout)

    def test_gpu_free_help(self):
        for script in SCRIPTS.glob("*.py"):
            if script.name == "common.py":
                continue
            result = subprocess.run(
                [sys.executable, str(script), "--help"], capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
