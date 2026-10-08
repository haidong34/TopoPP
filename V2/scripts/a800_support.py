"""Pure-Python configuration, checkpoint metadata and official vLLM command builders."""

import json
import math
import re
import struct
from pathlib import Path

from cost_model import positive_int


def validate_config(config):
    if config["gpu_ids"] != [0, 1, 2, 3] or config["tp"] != 1 or config["pp"] != 4:
        raise ValueError("This experiment targets logical GPUs 0,1,2,3 with TP=1, PP=4")
    if config["vllm_version"] != "0.10.1.1" or config["dtype"] != "bfloat16":
        raise ValueError("Official vLLM 0.10.1.1 and BF16 required by the calibrated worker")
    if config["host"] != "127.0.0.1":
        raise ValueError("Experiment driver uses loopback API access")
    for key in (
        "port",
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "block_size",
        "num_gpu_blocks_override",
        "startup_timeout_seconds",
        "benchmark_timeout_seconds",
    ):
        positive_int(config[key], key)
    if config["port"] > 65535 or config["block_size"] != 16:
        raise ValueError("Valid TCP port and block_size=16 required")
    if (
        not 0 < config["gpu_memory_utilization"] <= 1
        or not math.isfinite(config["runtime_reserve_gib"])
        or config["runtime_reserve_gib"] < 0
    ):
        raise ValueError("Invalid memory utilization/reserve")
    bench = config["benchmark"]
    for key in ("input_len", "output_len", "concurrency", "num_prompts"):
        positive_int(bench[key], key)
    if bench["input_len"] + bench["output_len"] > config["max_model_len"]:
        raise ValueError("Benchmark exceeds max_model_len")
    if bench["concurrency"] > config["max_num_seqs"]:
        raise ValueError("max_num_seqs must support target concurrency")
    required_pages = bench["concurrency"] * math.ceil(
        (bench["input_len"] + bench["output_len"]) / config["block_size"]
    )
    if config["num_gpu_blocks_override"] < required_pages:
        raise ValueError("Fixed KV page pool cannot hold the benchmark live-token bound")
    calibration = config["calibration"]
    for key in (
        "warmup_prompts",
        "num_prompts",
        "sample_every",
        "context_bin",
        "min_samples",
        "min_counts",
    ):
        positive_int(calibration[key], key)
    if not 0 < calibration["min_coverage"] <= 1:
        raise ValueError("Invalid reference coverage threshold")
    contention = config["contention"]
    if contention["gpu"] != 2:
        raise ValueError("This experiment uses fixed GPU2 contention")
    for key in ("m", "n", "k", "warmup_seconds"):
        positive_int(contention[key], key)
    return config


def geometry(model_config):
    if (
        model_config.get("model_type") != "qwen3"
        or model_config.get("quantization_config")
        or model_config.get("use_sliding_window")
    ):
        raise ValueError("Unquantized dense full-attention Qwen3 required")
    return {
        "layers": model_config["num_hidden_layers"],
        "hidden_size": model_config["hidden_size"],
        "attention_heads": model_config["num_attention_heads"],
        "kv_heads": model_config["num_key_value_heads"],
        "head_dim": model_config.get("head_dim")
        or model_config["hidden_size"] // model_config["num_attention_heads"],
        "vocab_size": model_config["vocab_size"],
    }


def checkpoint_bytes(model_directory, model_config):
    """Read safetensors headers only; never instantiate a Hugging Face model."""
    root = Path(model_directory).resolve()
    index = root / "model.safetensors.index.json"
    if index.exists():
        mapping = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        shards = sorted(set(mapping.values()))
    else:
        mapping, shards = None, ["model.safetensors"]
    tensors = {}
    for name in shards:
        file = (root / name).resolve()
        if not file.is_relative_to(root):
            raise ValueError("Checkpoint shard escapes model directory")
        with file.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise ValueError(f"Truncated checkpoint: {file}")
            size = struct.unpack("<Q", prefix)[0]
            if size < 2 or size > 64 * 1024**2 or size + 8 > file.stat().st_size:
                raise ValueError(f"Invalid safetensors header: {file}")
            header = json.loads(handle.read(size))
        for key, metadata in header.items():
            if key == "__metadata__":
                continue
            if key in tensors:
                raise ValueError(f"Duplicate checkpoint tensor: {key}")
            if mapping is not None and mapping.get(key) != name:
                raise ValueError(f"Index/header mismatch: {key}")
            if metadata["dtype"] not in ("BF16", "F16", "F32"):
                raise ValueError(f"Nonfloating/quantized tensor unsupported: {key}")
            begin, end = metadata["data_offsets"]
            dtype_size = 4 if metadata["dtype"] == "F32" else 2
            numel = math.prod(metadata["shape"])
            if (
                begin < 0
                or end - begin != numel * dtype_size
                or end + size + 8 > file.stat().st_size
            ):
                raise ValueError(f"Tensor offsets/size invalid: {key}")
            tensors[key] = 2 * numel  # BF16 serving parameter footprint.
    if mapping is not None and set(mapping) != set(tensors):
        raise ValueError("Checkpoint shards incomplete")
    per_layer = [0] * model_config["num_hidden_layers"]
    for name, count in tensors.items():
        match = re.match(r"model\.layers\.(\d+)\.", name)
        if match:
            layer = int(match.group(1))
            if not 0 <= layer < len(per_layer):
                raise ValueError("Invalid checkpoint layer index")
            per_layer[layer] += count
    if not per_layer or min(per_layer) <= 0:
        raise ValueError("Missing Transformer layer weights")
    if "model.embed_tokens.weight" not in tensors or "model.norm.weight" not in tensors:
        raise ValueError("Missing embedding/final norm weights")
    head = tensors.get("lm_head.weight")
    if head is None:
        if not model_config.get("tie_word_embeddings"):
            raise ValueError("Missing LM head")
        head = tensors["model.embed_tokens.weight"]
    return {
        "per_layer": per_layer,
        "embedding": tensors["model.embed_tokens.weight"],
        "tail": tensors["model.norm.weight"] + head,
    }


def memory_profile(config, model_config, weights, gpu_memory, state):
    geom = geometry(model_config)
    cache_tokens = config["block_size"] * config["num_gpu_blocks_override"]
    layer_kv = 2 * geom["kv_heads"] * geom["head_dim"] * 2 * cache_tokens
    rows = []
    for i, gpu in enumerate(gpu_memory):
        rows.append(
            {
                "gpu_id": i,
                "budget_bytes": int(
                    min(
                        gpu["total_bytes"] * config["gpu_memory_utilization"], gpu["free_bytes"]
                    )
                ),
                "per_layer_weights_bytes": max(weights["per_layer"]),
                "per_layer_kv_bytes": layer_kv,
                "role_weights_bytes": (weights["embedding"] if i == 0 else 0)
                + (weights["tail"] if i == 3 else 0),
                "runtime_reserve_bytes": int(config["runtime_reserve_gib"] * 1024**3),
            }
        )
    return {
        "kind": "memory_only",
        "complete": True,
        "num_layers": geom["layers"],
        "gpu_ids": config["gpu_ids"],
        "profile_state": state,
        "model_geometry": geom,
        "memory": {
            "cache_tokens": cache_tokens,
            "fixed_gpu_blocks": config["num_gpu_blocks_override"],
            "block_size": config["block_size"],
            "stages": rows,
        },
        "warnings": [
            "KV footprint uses the identical fixed page pool in every vLLM run; runtime reserve remains an estimate."
        ],
    }


def server_command(config, executable, instrumented=False):
    args = [
        executable,
        "serve",
        config["model"],
        "--served-model-name",
        config["served_model_name"],
        "--tensor-parallel-size",
        "1",
        "--pipeline-parallel-size",
        "4",
        "--distributed-executor-backend",
        "mp",
        "--dtype",
        "bfloat16",
        "--host",
        config["host"],
        "--port",
        str(config["port"]),
        "--gpu-memory-utilization",
        str(config["gpu_memory_utilization"]),
        "--max-model-len",
        str(config["max_model_len"]),
        "--max-num-seqs",
        str(config["max_num_seqs"]),
        "--max-num-batched-tokens",
        str(config["max_num_batched_tokens"]),
        "--block-size",
        str(config["block_size"]),
        "--num-gpu-blocks-override",
        str(config["num_gpu_blocks_override"]),
        "--no-enable-prefix-caching",
        "--seed",
        str(config["seed"]),
    ]
    if config["enforce_eager"]:
        args.append("--enforce-eager")
    if instrumented:
        args.extend(["--worker-cls", "topopp_worker.TopoPPWorker"])
    return args


def benchmark_command(config, executable, output, prompts):
    bench = config["benchmark"]
    return [
        executable,
        "bench",
        "serve",
        "--endpoint-type",
        "openai",
        "--backend",
        "vllm",
        "--base-url",
        f"http://{config['host']}:{config['port']}",
        "--endpoint",
        "/v1/completions",
        "--model",
        config["model"],
        "--tokenizer",
        config["model"],
        "--served-model-name",
        config["served_model_name"],
        "--dataset-name",
        "random",
        "--random-input-len",
        str(bench["input_len"]),
        "--random-output-len",
        str(bench["output_len"]),
        "--random-range-ratio",
        "0",
        "--random-prefix-len",
        "0",
        "--num-prompts",
        str(prompts),
        "--max-concurrency",
        str(bench["concurrency"]),
        "--request-rate",
        "inf",
        "--ignore-eos",
        "--temperature",
        "0",
        "--seed",
        str(config["seed"]),
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(Path(output).parent),
        "--result-filename",
        Path(output).name,
        "--percentile-metrics",
        "ttft,tpot,itl",
        "--metric-percentiles",
        "50,99",
    ]
