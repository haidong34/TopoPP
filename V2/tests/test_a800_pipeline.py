"""CPU behavior tests for official-vLLM commands, headers, memory and calibration worker."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
from a800_support import (
    benchmark_command,
    checkpoint_bytes,
    geometry,
    memory_profile,
    server_command,
    validate_config,
)
from profile_common import read_json
from server_a800_14b import launch_environment, plan


def checkpoint_file(path, layer_count=2):
    sizes = {
        "model.embed_tokens.weight": [8, 4],
        "model.norm.weight": [4],
        "lm_head.weight": [8, 4],
    }
    sizes.update({f"model.layers.{i}.mlp.weight": [4, 4] for i in range(layer_count)})
    header, offset = {}, 0
    for name, shape in sizes.items():
        length = 2
        for dim in shape:
            length *= dim
        header[name] = {
            "dtype": "BF16",
            "shape": shape,
            "data_offsets": [offset, offset + length],
        }
        offset += length
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))
    return sizes


class A800PipelineTests(unittest.TestCase):
    def setUp(self):
        self.config = read_json(ROOT / "configs" / "a800_qwen3_14b.json")

    def test_config_official_model_and_capacity(self):
        validate_config(self.config)
        self.assertEqual(self.config["expected_geometry"]["layers"], 40)
        self.assertEqual(
            self.config["num_gpu_blocks_override"] * self.config["block_size"], 16384
        )
        for change in (
            {"tp": 4},
            {"pp": 2},
            {"num_gpu_blocks_override": 1},
            {"max_model_len": 1024},
        ):
            config = {**self.config, **change}
            with self.assertRaises(ValueError):
                validate_config(config)

    def test_server_and_ab_have_identical_runtime_flags(self):
        plain = server_command(self.config, "vllm")
        measured = server_command(self.config, "vllm", True)
        self.assertEqual(measured[:-2], plain)
        self.assertEqual(measured[-2:], ["--worker-cls", "topopp_worker.TopoPPWorker"])
        self.assertNotIn("--enforce-eager", plain)
        self.assertIn("--no-enable-prefix-caching", plain)
        self.assertNotIn("nanovllm", " ".join(plain))

    def test_fixed_lengths_and_real_official_benchmark(self):
        command = benchmark_command(self.config, "vllm", Path("results/result.json"), 128)
        self.assertEqual(command[:3], ["vllm", "bench", "serve"])
        self.assertEqual(command[command.index("--random-range-ratio") + 1], "0")
        self.assertEqual(command[command.index("--random-output-len") + 1], "128")
        self.assertIn("--ignore-eos", command)
        self.assertIn("--served-model-name", command)

    def test_headers_and_missing_download_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            file = root / "model.safetensors"
            checkpoint_file(file)
            sizes = checkpoint_bytes(
                root, {"num_hidden_layers": 2, "tie_word_embeddings": False}
            )
            self.assertEqual(sizes["per_layer"], [32, 32])
            self.assertEqual(sizes["embedding"], 64)
            self.assertEqual(sizes["tail"], 72)
            with self.assertRaises(ValueError):
                checkpoint_bytes(root, {"num_hidden_layers": 3, "tie_word_embeddings": False})
            file.write_bytes(file.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                checkpoint_bytes(root, {"num_hidden_layers": 2, "tie_word_embeddings": False})

    def test_checkpoint_index_cannot_escape_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"x": "../outside.safetensors"}})
            )
            with self.assertRaises(ValueError):
                checkpoint_bytes(root, {"num_hidden_layers": 2})

    def test_memory_uses_fixed_pool_not_only_live_tokens(self):
        g = self.config["expected_geometry"]
        model = {
            "model_type": "qwen3",
            "num_hidden_layers": g["layers"],
            "hidden_size": g["hidden_size"],
            "num_attention_heads": g["attention_heads"],
            "num_key_value_heads": g["kv_heads"],
            "head_dim": g["head_dim"],
            "vocab_size": g["vocab_size"],
        }
        profile = memory_profile(
            self.config,
            model,
            {"per_layer": [100] * 40, "embedding": 10, "tail": 20},
            [{"total_bytes": 80 * 1024**3, "free_bytes": 70 * 1024**3}] * 4,
            "H1",
        )
        self.assertEqual(
            profile["memory"]["stages"][0]["per_layer_kv_bytes"], 2 * 8 * 128 * 2 * 16384
        )
        self.assertEqual(profile["memory"]["stages"][0]["role_weights_bytes"], 10)
        self.assertEqual(profile["memory"]["stages"][3]["role_weights_bytes"], 20)

    def test_inherited_partition_and_profiler_are_cleared(self):
        with patch.dict(
            os.environ,
            {
                "VLLM_PP_LAYER_PARTITION": "1,1,1,37",
                "VLLM_TORCH_PROFILER_DIR": "/old",
                "TOPOPP_ENABLE_FILE": "/old",
            },
        ):
            env = launch_environment(False)
            self.assertNotIn("VLLM_PP_LAYER_PARTITION", env)
            self.assertNotIn("VLLM_TORCH_PROFILER_DIR", env)
            self.assertNotIn("TOPOPP_ENABLE_FILE", env)
            self.assertEqual(env["VLLM_USE_V1"], "1")

    def test_cpu_dry_run(self):
        command = [
            sys.executable,
            str(SCRIPTS / "server_a800_14b.py"),
            "calibrate",
            "--run-dir",
            "unused",
            "--dry-run",
        ]
        output = subprocess.run(command, capture_output=True, text=True, check=True)
        result = json.loads(output.stdout)
        self.assertEqual(result["sweeps"], [[10, 10, 10, 10], [16, 4, 16, 4], [4, 16, 4, 16]])

    def test_worker_mixed_prefill_decode_and_no_forced_sync(self):
        # Exercise the worker with event and runtime doubles.
        sync_calls = []

        class Event:
            def __init__(self, **kwargs):
                pass

            def record(self):
                pass

            def query(self):
                return True

            def synchronize(self):
                sync_calls.append(True)

            def elapsed_time(self, other):
                return 2.0

        class Base:
            def load_model(self, *args, **kwargs):
                return "loaded"

        modules = {}
        for name in (
            "torch",
            "vllm",
            "vllm.distributed",
            "vllm.distributed.utils",
            "vllm.v1",
            "vllm.v1.worker",
            "vllm.v1.worker.gpu_worker",
        ):
            modules[name] = ModuleType(name)
        modules["torch"].cuda = SimpleNamespace(Event=Event)
        modules["torch"].__version__ = "2.7.1"
        modules["vllm"].__version__ = "0.10.1.1"
        modules["vllm.distributed"].get_pp_group = lambda: SimpleNamespace(rank_in_group=0)
        modules["vllm.distributed.utils"].get_pp_indices = lambda layers, rank, count: (0, 10)
        modules["vllm.v1.worker.gpu_worker"].Worker = Base
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, modules), patch(
            "atexit.register"
        ):
            spec = importlib.util.spec_from_file_location(
                "tested_topopp_worker", SCRIPTS / "topopp_worker.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            worker = module.TopoPPWorker()
            cfg = SimpleNamespace(
                model_type="qwen3",
                num_hidden_layers=40,
                hidden_size=5120,
                num_attention_heads=40,
                num_key_value_heads=8,
                head_dim=128,
                vocab_size=151936,
                to_json_string=lambda: "{}",
            )
            worker.model_config = SimpleNamespace(hf_config=cfg, enforce_eager=False)
            worker.vllm_config = SimpleNamespace(
                parallel_config=SimpleNamespace(
                    tensor_parallel_size=1,
                    data_parallel_size=1,
                    distributed_executor_backend="mp",
                    pipeline_parallel_size=4,
                ),
                speculative_config=None,
                compilation_config=SimpleNamespace(
                    cudagraph_mode="FULL", cudagraph_capture_sizes=[1, 2, 4]
                ),
                scheduler_config=SimpleNamespace(max_num_batched_tokens=8192, max_num_seqs=8),
            )
            worker.device = SimpleNamespace(index=0)
            requests = {
                "p": SimpleNamespace(num_computed_tokens=0, prompt_token_ids=list(range(10))),
                "d": SimpleNamespace(num_computed_tokens=10, prompt_token_ids=list(range(10))),
            }
            worker.model_runner = SimpleNamespace(
                requests=requests, execute_model=lambda *args: "generated"
            )
            env = {
                "TOPOPP_TRACE_DIR": temp,
                "TOPOPP_PROFILE_STATE": "H1",
                "TOPOPP_RUN_ID": "test",
                "TOPOPP_SAMPLE_EVERY": "1",
            }
            with patch.dict(os.environ, env, clear=True):
                self.assertEqual(worker.load_model(), "loaded")
                output = worker.model_runner.execute_model(
                    SimpleNamespace(
                        total_num_scheduled_tokens=2, num_scheduled_tokens={"p": 1, "d": 1}
                    )
                )
                self.assertEqual(output, "generated")
                row = json.loads(next(Path(temp).glob("*.jsonl")).read_text().strip())
                self.assertEqual(row["phase"], "mixed")
                self.assertEqual(row["query_tokens"], [1, 1])
                self.assertEqual(row["context_lengths"], [0, 10])
                self.assertEqual(sync_calls, [])

    def test_whole_pipeline_plan_from_four_rank_traces(self):
        from a800_support import memory_profile

        g = self.config["expected_geometry"]
        model_config = {
            "model_type": "qwen3",
            "num_hidden_layers": 40,
            "hidden_size": 5120,
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 151936,
        }
        memory = memory_profile(
            self.config,
            model_config,
            {"per_layer": [100] * 40, "embedding": 10, "tail": 20},
            [{"total_bytes": 80 * 1024**3, "free_bytes": 75 * 1024**3}] * 4,
            "H1",
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            environment = root / "environment"
            environment.mkdir()
            (environment / "environment.json").write_text(
                json.dumps(
                    {
                        "experiment_config": self.config,
                        "profile_state": "H1",
                        "model_config_sha256": "raw-config",
                    }
                )
            )
            (environment / "memory_profile.json").write_text(json.dumps(memory))
            for run_id, counts in (
                ("default_h1", [10, 10, 10, 10]),
                ("sweep1_h1", [16, 4, 16, 4]),
                ("sweep2_h1", [4, 16, 4, 16]),
            ):
                directory = root / "traces" / run_id
                directory.mkdir(parents=True)
                for rank in range(4):
                    n = counts[rank]
                    cost = n * (2.2 if rank == 2 else 1) + (0.1 if rank == 3 else 0)
                    rows = [
                        {
                            "num_layers": 40,
                            "pp_size": 4,
                            "config_sha256": "normalized-config",
                            "enforce_eager": False,
                            "vllm": "0.10.1.1",
                            "torch": "2.7.1",
                            "profile_state": "H1",
                            "sample_every": 1,
                            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                            "model_geometry": g,
                            "execution_signature": {},
                            "run_id": run_id,
                            "pp_rank": rank,
                            "gpu_id": rank,
                            "stage_layers": n,
                            "phase": "decode",
                            "batch_size": 2,
                            "query_tokens": [1, 1],
                            "context_lengths": [1024, 1024],
                            "intermediate_shapes": [[4, 5120], [4, 5120]] if rank != 3 else [],
                            "local_cuda_ms": cost,
                            "local_wall_ms": 0.5 * cost,
                        }
                        for _ in range(6)
                    ]
                    (directory / f"rank{rank}.jsonl").write_text(
                        "\n".join(json.dumps(row) for row in rows) + "\n"
                    )
            plan(self.config, root, "H1")
            selected = read_json(root / "partition.json")
            self.assertEqual(selected["topopp"]["partition"], [12, 12, 5, 11])
            self.assertAlmostEqual(selected["topopp"]["objective_ms"], 12)
            profile = read_json(root / "stage_profile.json")
            self.assertEqual(
                profile["scenarios"][0]["stages"][0]["communication_shapes"],
                [[4, 5120], [4, 5120]],
            )
            self.assertEqual(profile["scenarios"][0]["weight"], 6)
            self.assertIn("12,12,5,11", (root / "partition.env").read_text())


if __name__ == "__main__":
    unittest.main()
