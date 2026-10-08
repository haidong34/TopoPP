import ast
import copy
import itertools
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from build_profile_from_traces import build_profile
from cost_model import CostModel
from make_sweeps import make_sweeps


def data():
    geometry = {
        "layers": 6,
        "hidden_size": 16,
        "attention_heads": 4,
        "kv_heads": 2,
        "head_dim": 4,
        "vocab_size": 32,
    }
    memory = {
        "num_layers": 6,
        "gpu_ids": [0, 1],
        "profile_state": "H1",
        "model_geometry": geometry,
        "memory": {
            "stages": [
                {
                    "gpu_id": i,
                    "budget_bytes": 10000,
                    "per_layer_weights_bytes": 10,
                    "per_layer_kv_bytes": 10,
                    "role_weights_bytes": 0,
                    "runtime_reserve_bytes": 0,
                }
                for i in range(2)
            ]
        },
    }
    rows = []
    for run_id, counts in (("default", [3, 3]), ("sweep", [2, 4])):
        for rank in range(2):
            for step in range(6):
                rows.append(
                    {
                        "num_layers": 6,
                        "pp_size": 2,
                        "config_sha256": "abc",
                        "enforce_eager": False,
                        "vllm": "0.10.1.1",
                        "torch": "2.7.1",
                        "profile_state": "H1",
                        "sample_every": 1,
                        "CUDA_VISIBLE_DEVICES": None,
                        "model_geometry": geometry,
                        "execution_signature": {},
                        "run_id": run_id,
                        "pp_rank": rank,
                        "gpu_id": rank,
                        "stage_layers": counts[rank],
                        "phase": "decode",
                        "batch_size": 2,
                        "query_tokens": [1, 1],
                        "context_lengths": [1024, 1024],
                        "local_cuda_ms": counts[rank],
                        "local_wall_ms": 0.5 * counts[rank],
                    }
                )
    return rows, memory


class TraceAndProfileTests(unittest.TestCase):
    def test_reference_weights_not_candidate_sample_counts(self):
        rows, memory = data()
        # Duplicating a sweep must not change baseline weights.
        rows += [copy.deepcopy(row) for row in rows if row["run_id"] == "sweep"]
        profile = build_profile(rows, memory, "default")
        self.assertEqual(profile["scenarios"][0]["weight"], 6)
        self.assertEqual(profile["reference_iteration_coverage"], 1)
        stage = profile["scenarios"][0]["stages"][0]
        self.assertEqual(stage["total_curve_ms"], {"3": 3, "2": 2})
        self.assertEqual(CostModel(profile).evaluate([3, 3])["objective_ms"], 3)

    def test_total_stage_cannot_double_count_tail(self):
        rows, memory = data()
        profile = build_profile(rows, memory, "default")
        sid = profile["scenarios"][0]["id"]
        with self.assertRaises(ValueError):
            CostModel(
                profile,
                {
                    "profile_state": "H1",
                    "scenarios": {sid: [{"gpu_id": 0, "runtime_extra_ms": 1}, {"gpu_id": 1}]},
                },
            )
        profile["scenarios"][0]["stages"][0]["fixed_ms"] = {"embedding": 1}
        with self.assertRaises(ValueError):
            CostModel(profile)

    def test_host_upstream_wait_cannot_change_gpu_cost_or_partition(self):
        from partition import solve_v2

        rows, memory = data()
        reference = build_profile(rows, memory, "default")
        expected = solve_v2(CostModel(reference))[0][0]
        for row in rows:
            if row["pp_rank"] == 1:
                row["local_wall_ms"] = 1000.0
        output = build_profile(rows, memory, "default")
        actual = solve_v2(CostModel(output))[0][0]
        self.assertEqual(actual["partition"], expected["partition"])
        self.assertEqual(actual["objective_ms"], expected["objective_ms"])
        stage = output["scenarios"][0]["stages"][1]
        self.assertEqual(stage["total_curve_ms"]["3"], 3)
        self.assertEqual(stage["host_curve_ms_diagnostic_only"]["3"], 1000)
        self.assertEqual(output["timing_metric"], "local_cuda_ms")

    def test_invalid_host_diagnostics_are_rejected(self):
        for value in (-1, float("nan"), float("inf")):
            rows, memory = data()
            rows[0]["local_wall_ms"] = value
            with self.assertRaises(ValueError):
                build_profile(rows, memory, "default")

    def test_state_backend_model_and_sampling_mismatch_rejected(self):
        for field, value in (
            ("profile_state", "H0"),
            ("enforce_eager", True),
            ("config_sha256", "different"),
            ("sample_every", 2),
        ):
            rows, memory = data()
            rows[-1][field] = value
            with self.assertRaises(ValueError):
                build_profile(rows, memory, "default")
        rows, memory = data()
        memory = copy.deepcopy(memory)
        memory["model_geometry"]["hidden_size"] = 32
        with self.assertRaises(ValueError):
            build_profile(rows, memory, "default")

    def test_partial_bucket_coverage_is_reported(self):
        rows, memory = data()
        extra = copy.deepcopy(rows[0])
        extra["query_tokens"] = [2, 2]
        extra["phase"] = "mixed"
        rows.append(extra)
        profile = build_profile(rows, memory, "default")
        self.assertAlmostEqual(profile["reference_iteration_coverage"], 6 / 7)
        self.assertTrue(any("covered" in warning for warning in profile["warnings"]))

    def test_insufficient_sweeps_rejected(self):
        rows, memory = data()
        with self.assertRaises(ValueError):
            build_profile([r for r in rows if r["run_id"] == "default"], memory, "default")
        with self.assertRaises(ValueError):
            build_profile(rows, memory, "unknown")

    def test_balanced_sweep_totals_and_each_stage_varies(self):
        for stages in (2, 3, 4, 5):
            sweeps = make_sweeps(10 * stages, stages, [4])
            for counts in sweeps:
                self.assertEqual(sum(counts), 10 * stages)
                self.assertGreater(min(counts), 0)
            for i in range(stages):
                self.assertEqual({p[i] for p in sweeps}, {6, 10, 14})
        self.assertEqual(len(make_sweeps(40, 4, [4])), 3)
        with self.assertRaises(ValueError):
            make_sweeps(8, 4, [2])

    def test_script_syntax_and_gpu_free_help(self):
        for path in SCRIPTS.glob("*.py"):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for script in (
            "partition.py",
            "build_profile_from_traces.py",
            "make_sweeps.py",
            "server_a800_14b.py",
            "contend_gpu.py",
        ):
            result = subprocess.run(
                [sys.executable, str(SCRIPTS / script), "--help"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()
