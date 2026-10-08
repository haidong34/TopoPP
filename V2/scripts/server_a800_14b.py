"""Linux single-node 4xA800 official-vLLM Qwen3-14B experiment driver.

Actions: preflight, smoke, calibrate, plan, ab. Never rents/boots a machine,
downloads a model, installs dependencies or kills unrelated processes.
"""

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
from urllib.request import urlopen

from a800_support import (
    benchmark_command,
    checkpoint_bytes,
    geometry,
    memory_profile,
    server_command,
    validate_config,
)
from build_profile_from_traces import build_profile
from cost_model import CostModel, default_partition
from make_sweeps import make_sweeps
from partition import solve_v2
from profile_common import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def stop_owned(process):
    """Every process passed here was launched by this driver in its own session."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=15)


def launch_environment(instrumented=False):
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES="0,1,2,3", VLLM_USE_V1="1", VLLM_WORKER_MULTIPROC_METHOD="spawn"
    )
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    # Prevent inherited user settings from changing this experiment silently.
    for name in (
        "VLLM_PP_LAYER_PARTITION",
        "VLLM_TORCH_PROFILER_DIR",
        "TOPOPP_TRACE_DIR",
        "TOPOPP_ENABLE_FILE",
        "TOPOPP_RUN_ID",
        "TOPOPP_PROFILE_STATE",
    ):
        env.pop(name, None)
    if instrumented:
        env["PYTHONPATH"] = str(SCRIPTS) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
    return env


def preflight(config, run_directory, state, competitor=None, external=False):
    import torch

    if importlib.metadata.version("vllm") != config["vllm_version"]:
        raise RuntimeError(
            "Installed vLLM differs from pinned 0.10.1.1; keep your working environment or explicitly port the worker before proceeding"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise RuntimeError("Expected exactly four visible CUDA GPUs")
    model_config = read_json(Path(config["model"]) / "config.json")
    actual = geometry(model_config)
    if actual != config["expected_geometry"]:
        raise RuntimeError(f"Local model is not the expected Qwen3-14B: {actual}")
    if not (Path(config["model"]) / "tokenizer.json").exists():
        raise RuntimeError("Local tokenizer.json missing; model download is incomplete")
    gpu_memory = []
    for gpu_id in config["gpu_ids"]:
        torch.cuda.set_device(gpu_id)
        props = torch.cuda.get_device_properties(gpu_id)
        if (
            "A800" not in props.name
            or props.total_memory < 75 * 1024**3
            or not torch.cuda.is_bf16_supported(including_emulation=False)
        ):
            raise RuntimeError(
                f"GPU{gpu_id} does not match A800 80GB/native BF16: {props.name}"
            )
        free, total = torch.cuda.mem_get_info(gpu_id)
        gpu_memory.append(
            {"gpu_id": gpu_id, "name": props.name, "total_bytes": total, "free_bytes": free}
        )
    weights = checkpoint_bytes(config["model"], model_config)
    memory = memory_profile(config, model_config, weights, gpu_memory, state)
    topology = subprocess.run(
        ["nvidia-smi", "topo", "-m"], check=True, capture_output=True, text=True
    ).stdout
    if not re.search(r"\bNV\d+\b", topology):
        raise RuntimeError("NVLink not detected; inspect nvidia-smi topo -m")
    compute_apps = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    allowed = {os.getpid()}
    if competitor is not None:
        allowed.add(competitor.pid)
    active = {
        int(line.split(",", 1)[0].strip())
        for line in compute_apps.splitlines()
        if line.strip() and line.split(",", 1)[0].strip().isdigit()
    }
    if not external and active - allowed:
        raise RuntimeError(
            f"Other GPU compute processes active: {sorted(active - allowed)}; stop your own prior jobs before this experiment"
        )
    disk = shutil.disk_usage(config["model"])
    info = {
        "experiment_config": config,
        "profile_state": state,
        "model_geometry": actual,
        "model_config_sha256": hashlib.sha256(
            (Path(config["model"]) / "config.json").read_bytes()
        ).hexdigest(),
        "versions": {
            "vllm": importlib.metadata.version("vllm"),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "cuda_runtime": torch.version.cuda,
        },
        "gpus": gpu_memory,
        "topology": topology,
        "compute_apps": compute_apps,
        "contention_management": (
            "external" if external else "managed" if competitor is not None else "none"
        ),
        "cpu_count": os.cpu_count(),
        "disk_free_bytes": disk.free,
        "baseline_partition": default_partition(actual["layers"], 4),
    }
    target = run_directory / "environment"
    write_json(target / "environment.json", info)
    write_json(target / "memory_profile.json", memory)
    print("Confirmed: official vLLM, Qwen3-14B, 4xA800, TP=1, PP=4", flush=True)
    print(
        "Default partition:",
        info["baseline_partition"],
        "fixed cache tokens:",
        memory["memory"]["cache_tokens"],
        flush=True,
    )
    return info


@contextmanager
def contention(config, directory, enabled):
    if not enabled:
        yield None
        return
    ready = directory / "contention.ready.json"
    if ready.exists():
        raise ValueError("Contention readiness file already exists; use a fresh run directory")
    opts = config["contention"]
    command = [
        sys.executable,
        str(SCRIPTS / "contend_gpu.py"),
        "--gpu",
        str(opts["gpu"]),
        "--m",
        str(opts["m"]),
        "--n",
        str(opts["n"]),
        "--k",
        str(opts["k"]),
        "--ready-file",
        str(ready),
    ]
    with (directory / "contention.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            env=launch_environment(),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 180
            while not ready.exists():
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("Contention failed to start; inspect contention.log")
                time.sleep(1)
            deadline = time.monotonic() + opts["warmup_seconds"]
            while time.monotonic() < deadline:
                check_competitor(process)
                time.sleep(1)
            yield process
        finally:
            stop_owned(process)


def check_competitor(process):
    if process is not None and process.poll() is not None:
        raise RuntimeError("Managed GPU2 contention exited; experiment aborted")


def wait_ready(config, process, competitor):
    deadline = time.monotonic() + config["startup_timeout_seconds"]
    base = f"http://{config['host']}:{config['port']}"
    while time.monotonic() < deadline:
        check_competitor(competitor)
        if process.poll() is not None:
            raise RuntimeError("vLLM exited before readiness; inspect this run's server.log")
        try:
            with urlopen(base + "/health", timeout=3) as response:
                health = response.status == 200
            with urlopen(base + "/v1/models", timeout=3) as response:
                models = json.load(response)
            if health and config["served_model_name"] in {r["id"] for r in models["data"]}:
                return
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(1)
    raise TimeoutError("vLLM startup timeout; inspect server.log")


def run_benchmark(config, executable, output, prompts, competitor):
    command = benchmark_command(config, executable, output, prompts)
    write_json(output.with_suffix(".command.json"), command)
    with output.with_suffix(".client.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            env=launch_environment(),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + config["benchmark_timeout_seconds"]
            while process.poll() is None:
                check_competitor(competitor)
                if time.monotonic() > deadline:
                    raise TimeoutError("Benchmark timeout; inspect client.log/server.log")
                time.sleep(1)
            if process.returncode:
                raise RuntimeError(
                    f"Benchmark client failed; inspect {output.with_suffix('.client.log')}"
                )
        finally:
            stop_owned(process)
    check_competitor(competitor)
    metrics = read_json(output)
    if (
        metrics["completed"] != prompts
        or metrics["total_output_tokens"] != prompts * config["benchmark"]["output_len"]
    ):
        raise RuntimeError(
            "Benchmark has failed requests or unexpected output lengths; raw results retained"
        )
    return metrics


def session(
    config, executable, directory, run_id, partition, instrumented, prompts, state, competitor
):
    target = directory / "runs" / run_id
    target.mkdir(parents=True, exist_ok=False)
    with socket.socket() as probe:
        try:
            probe.bind((config["host"], config["port"]))
        except OSError as error:
            raise RuntimeError(
                "API port occupied; stop your own prior server or choose a free port in config"
            ) from error
    env = launch_environment(instrumented)
    if partition is not None:
        env["VLLM_PP_LAYER_PARTITION"] = ",".join(map(str, partition))
    if instrumented:
        trace_dir = directory / "traces" / run_id
        trace_dir.mkdir(parents=True, exist_ok=False)
        enable = target / "enable_profile"
        env.update(
            TOPOPP_TRACE_DIR=str(trace_dir),
            TOPOPP_ENABLE_FILE=str(enable),
            TOPOPP_RUN_ID=run_id,
            TOPOPP_PROFILE_STATE=state,
            TOPOPP_SAMPLE_EVERY=str(config["calibration"]["sample_every"]),
        )
    command = server_command(config, executable, instrumented)
    write_json(
        target / "launch.json",
        {
            "command": command,
            "partition": partition,
            "instrumented": instrumented,
            "profile_state": state,
        },
    )
    with (target / "server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            wait_ready(config, process, competitor)
            if instrumented:
                for rank in range(4):
                    admission = read_json(trace_dir / f"rank{rank}.admission.json")
                    if (
                        admission["kv_blocks"] != config["num_gpu_blocks_override"]
                        or admission["block_size"] != config["block_size"]
                    ):
                        raise RuntimeError("Actual vLLM KV page pool differs across partitions")
            run_benchmark(
                config,
                executable,
                target / "warmup.json",
                config["calibration"]["warmup_prompts"],
                competitor,
            )
            if instrumented:
                enable.touch()
            print("Measuring:", run_id, "partition:", partition or "vLLM default", flush=True)
            metrics = run_benchmark(
                config, executable, target / "benchmark.json", prompts, competitor
            )
        finally:
            stop_owned(process)
    # Workers belong to this process group; release device memory before the next startup.
    time.sleep(3)
    return metrics


def plan(config, directory, state, supplemental=None):
    environment = read_json(directory / "environment" / "environment.json")
    if environment["experiment_config"] != config or environment["profile_state"] != state:
        raise ValueError("Experiment config/state differs from the calibration session")
    records = []
    for path in sorted((directory / "traces").rglob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                records.append(json.loads(line))
    settings = config["calibration"]
    profile = build_profile(
        records,
        read_json(directory / "environment" / "memory_profile.json"),
        f"default_{state.lower()}",
        settings["context_bin"],
        settings["min_samples"],
        settings["min_counts"],
    )
    if profile["reference_iteration_coverage"] < settings["min_coverage"]:
        raise ValueError(
            "Insufficient workload coverage; repeat/expand calibration rather than silently dropping buckets"
        )
    write_json(directory / "stage_profile.json", profile)
    model = CostModel(profile, read_json(supplemental) if supplemental else None)
    ranked, search = solve_v2(model, top_k=5)
    baseline = model.evaluate(default_partition(model.layers, model.stages))
    result = {
        "schema_version": 2,
        "experiment_config": config,
        "profile_state": state,
        "timing_metric": profile["timing_metric"],
        "cost_model_revision": profile["cost_model_revision"],
        "model_config_sha256": environment["model_config_sha256"],
        "baseline": baseline,
        "topopp": ranked[0],
        "candidates": ranked,
        "search": search,
        "warnings": model.warnings,
        "predicted_surrogate_reduction_percent": (
            100 * (1 - ranked[0]["objective_ms"] / baseline["objective_ms"])
            if baseline
            else None
        ),
    }
    write_json(directory / "partition.json", result)
    (directory / "partition.env").write_text(
        'export VLLM_PP_LAYER_PARTITION="' + ",".join(map(str, ranked[0]["partition"])) + '"\n',
        encoding="utf-8",
    )
    print(
        "Selected:",
        ranked[0]["partition"],
        "coverage:",
        profile["reference_iteration_coverage"],
        flush=True,
    )


def run_calibration(config, executable, directory, state, competitor):
    sweeps = make_sweeps(
        config["expected_geometry"]["layers"], 4, config["calibration"]["deltas"]
    )
    write_json(directory / "sweeps.json", sweeps)
    for index, counts in enumerate(sweeps):
        run_id = f"default_{state.lower()}" if index == 0 else f"sweep{index}_{state.lower()}"
        session(
            config,
            executable,
            directory,
            run_id,
            counts if index else None,
            True,
            config["calibration"]["num_prompts"],
            state,
            competitor,
        )


def run_ab(config, executable, directory, state, partition_file, info, competitor):
    selection = read_json(partition_file)
    if (
        selection.get("experiment_config") != config
        or selection["profile_state"] != state
        or selection["model_config_sha256"] != info["model_config_sha256"]
    ):
        raise ValueError("A/B config, model or H0/H1 state differs from the selected partition")
    partition = selection["topopp"]["partition"]
    if (
        len(partition) != 4
        or sum(partition) != 40
        or any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in partition)
    ):
        raise ValueError("Invalid 40-layer/4-stage selected partition")
    measurements = []
    for index, scheme in enumerate(("A", "B", "B", "A", "A", "B")):
        metrics = session(
            config,
            executable,
            directory,
            f"ab{index + 1}_{scheme}",
            None if scheme == "A" else partition,
            False,
            config["benchmark"]["num_prompts"],
            state,
            competitor,
        )
        measurements.append({"scheme": scheme, "metrics": metrics})
        write_json(directory / "ab.partial.json", measurements)
    throughput = {
        scheme: [
            r["metrics"]["output_throughput"] for r in measurements if r["scheme"] == scheme
        ]
        for scheme in ("A", "B")
    }
    summary = {
        "order": "AB,BA,AB",
        "A": "vLLM default",
        "B": partition,
        "measurements": measurements,
        "output_throughput_samples": throughput,
        "median_output_throughput": {k: statistics.median(v) for k, v in throughput.items()},
        "median_output_throughput_change_percent": 100
        * (statistics.median(throughput["B"]) / statistics.median(throughput["A"]) - 1),
        "note": "Three samples per scheme; inspect paired changes and tails, not statistical significance.",
    }
    write_json(directory / "ab_summary.json", summary)
    print(
        "A/B median output throughput change:",
        summary["median_output_throughput_change_percent"],
        "%",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "smoke", "calibrate", "plan", "ab"))
    parser.add_argument("--config", default=str(ROOT / "configs" / "a800_qwen3_14b.json"))
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Fresh directory for each GPU experiment; plan reuses calibration directory",
    )
    parser.add_argument("--state", choices=("H0", "H1"), default="H0")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--with-contention",
        action="store_true",
        help="Driver owns continuous GPU2 GEMM for this session",
    )
    modes.add_argument(
        "--external-contention",
        action="store_true",
        help="Use your already-running GPU2 competitor; caller maintains it",
    )
    parser.add_argument("--partition-file", help="For ab: partition.json produced by plan")
    parser.add_argument(
        "--supplemental",
        help="For plan: matching scenario/GPU/state communication-only supplemental JSON",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print official-vLLM commands without GPU/model/SSH access",
    )
    args = parser.parse_args()
    config = validate_config(read_json(args.config))
    directory = Path(args.run_dir).resolve()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "server_calibration": server_command(config, "vllm", True),
                    "server_ab": server_command(config, "vllm", False),
                    "benchmark": benchmark_command(
                        config,
                        "vllm",
                        directory / "benchmark.json",
                        config["benchmark"]["num_prompts"],
                    ),
                    "sweeps": make_sweeps(
                        config["expected_geometry"]["layers"],
                        4,
                        config["calibration"]["deltas"],
                    ),
                },
                indent=2,
            )
        )
        return
    if args.command == "plan":
        plan(config, directory, args.state, args.supplemental)
        return
    if not sys.platform.startswith("linux"):
        parser.error("GPU actions run on your rented Linux server; use --dry-run locally")
    if args.state == "H1" and not (args.with_contention or args.external_contention):
        parser.error("H1 requires --with-contention or --external-contention")
    if args.state == "H0" and (args.with_contention or args.external_contention):
        parser.error("H0 is the no-contention condition")
    if directory.exists() and any(directory.iterdir()):
        parser.error(
            "GPU actions require a fresh --run-dir to avoid mixing trace/benchmark results"
        )
    if args.command == "ab" and not args.partition_file:
        parser.error("ab requires --partition-file from the plan action")
    directory.mkdir(parents=True, exist_ok=True)
    executable = shutil.which("vllm")
    if not executable:
        raise RuntimeError("Activate the official vLLM environment; vllm executable missing")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    with contention(config, directory, args.with_contention) as competitor:
        info = preflight(config, directory, args.state, competitor, args.external_contention)
        if args.command == "preflight":
            return
        if args.command == "smoke":
            session(
                config, executable, directory, "smoke", None, False, 8, args.state, competitor
            )
        elif args.command == "calibrate":
            run_calibration(config, executable, directory, args.state, competitor)
        else:
            run_ab(
                config, executable, directory, args.state, args.partition_file, info, competitor
            )


if __name__ == "__main__":
    main()
