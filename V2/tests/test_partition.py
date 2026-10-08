"""Planner tests for scenario costs, memory admission and CLI output."""

import copy
import itertools
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from cost_model import CostModel, default_partition, interpolate
from partition import solve_v2


def fixture(layers=6, times=((1, 2), (3, 1)), weights=(1, 1)):
    return {
        "schema_version": 2,
        "num_layers": layers,
        "gpu_ids": [0, 1],
        "profile_state": "H1",
        "weight_source": "serving_trace_counts",
        "scenarios": [
            {
                "id": str(s),
                "weight": weights[s],
                "stages": [
                    {
                        "gpu_id": i,
                        "block_curve_ms": {"1": t, str(layers - 1): t * (layers - 1)},
                        "fixed_ms": {},
                    }
                    for i, t in enumerate(ts)
                ],
            }
            for s, ts in enumerate(times)
        ],
    }


class PlannerTests(unittest.TestCase):

    def test_default_remainder_matches_version(self):
        self.assertEqual(default_partition(29, 4), [7, 7, 8, 7])
        self.assertEqual(default_partition(30, 4), [7, 8, 8, 7])
        self.assertEqual(default_partition(31, 4), [8, 8, 8, 7])
        self.assertEqual(default_partition(7, 1), [7])

    def test_phase_maxima_not_max_of_phase_average(self):
        model = CostModel(fixture(layers=2, times=((10, 1), (1, 10))))
        self.assertEqual(model.evaluate([1, 1])["objective_ms"], 10)

    def test_multiscenario_against_independent_oracle(self):
        data = fixture(times=((1, 4), (5, 1)), weights=(2, 7))
        # Stage-specific fixed costs must move the optimum correctly.
        data["scenarios"][0]["stages"][0]["fixed_ms"] = {"embedding": 2}
        data["scenarios"][1]["stages"][1]["fixed_ms"] = {"tail": 3}
        model = CostModel(data)
        result, stats = solve_v2(model, top_k=4)
        oracle = []
        for n in range(1, 6):
            score = 2 / 9 * max(n + 2, (6 - n) * 4) + 7 / 9 * max(n * 5, 6 - n + 3)
            oracle.append((score, [n, 6 - n]))
        oracle.sort()
        for result_row, (score, partition) in zip(result, oracle):
            self.assertAlmostEqual(result_row["objective_ms"], score)
            self.assertEqual(result_row["partition"], partition)
        self.assertEqual(stats["feasible"], 5)

    def test_memory_rejects_compute_optimal(self):
        data = fixture(times=((1, 10),), weights=(1,))
        data["memory"] = {
            "stages": [
                {
                    "gpu_id": i,
                    "budget_bytes": 30 if i == 0 else 100,
                    "per_layer_weights_bytes": 10,
                    "per_layer_kv_bytes": 0,
                    "role_weights_bytes": 0,
                    "runtime_reserve_bytes": 0,
                }
                for i in range(2)
            ]
        }
        model = CostModel(data)
        self.assertIsNone(model.evaluate([5, 1]))
        self.assertEqual(solve_v2(model)[0][0]["partition"], [3, 3])

    def test_no_extrapolation_or_fake_baseline(self):
        self.assertEqual(interpolate({1: 2, 4: 8}, 2), 4)
        self.assertIsNone(interpolate({1: 2, 4: 8}, 5))
        data = fixture()
        for s in data["scenarios"]:
            s["stages"][0]["block_curve_ms"] = {"1": 1, "2": 2}
        model = CostModel(data)
        self.assertIsNone(model.evaluate([3, 3]))
        self.assertEqual(solve_v2(model)[1]["feasible"], 2)

    def test_invalid_inputs(self):
        for mutation in (
            lambda d: d.update(complete=False),
            lambda d: d.update(gpu_ids=[0, 0]),
            lambda d: d["scenarios"][0].update(weight=float("nan")),
            lambda d: d["scenarios"][0]["stages"].reverse(),
        ):
            data = fixture()
            mutation(data)
            with self.assertRaises(ValueError):
                CostModel(data)
        with self.assertRaises(ValueError):
            solve_v2(CostModel(fixture()), max_candidates=1)

    def test_supplemental_replaces_without_double_count(self):
        data = fixture()
        data["scenarios"][0]["stages"][0]["fixed_ms"] = {"embedding": 2}
        extra = {
            "profile_state": "H1",
            "scenarios": {
                "0": [
                    {
                        "gpu_id": 0,
                        "replace_fixed_ms": {"embedding": 5},
                        "runtime_extra_ms": 3,
                        "communication_ms": 1,
                    },
                    {"gpu_id": 1},
                ]
            },
        }
        model = CostModel(data, extra)
        self.assertEqual(model.stage_cost(0, 0, 2), 11)
        extra["profile_state"] = "H0"
        with self.assertRaises(ValueError):
            CostModel(data, extra)

    def test_cli_output_creation(self):
        for profile, layers in ((fixture(), 6),):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source, target = root / "input.json", root / "new" / "output.json"
                source.write_text(json.dumps(profile), encoding="utf-8")
                subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPTS / "partition.py"),
                        "--profile",
                        str(source),
                        "--num-layers",
                        str(layers),
                        "--output",
                        str(target),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                output = json.loads(target.read_text(encoding="utf-8"))
                self.assertEqual(sum(output["topopp"]["partition"]), layers)
                self.assertIn("VLLM_PP_LAYER_PARTITION", target.with_suffix(".env").read_text())


if __name__ == "__main__":
    unittest.main()
