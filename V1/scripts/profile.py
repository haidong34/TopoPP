"""Measure BF16 GEMM latency on each GPU in pipeline order."""

import argparse
import math
import statistics
import time

from common import write_json


def summarize(samples):
    if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError("Timing samples must be positive and finite")
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "std_ms": statistics.pstdev(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def measure_gpu(torch, gpu, shape, warmup, samples, operations):
    torch.cuda.set_device(gpu)
    device = f"cuda:{gpu}"
    rows, columns, inner = shape
    with torch.inference_mode():
        left = torch.randn(rows, inner, device=device, dtype=torch.bfloat16)
        right = torch.randn(inner, columns, device=device, dtype=torch.bfloat16)
        output = torch.empty(rows, columns, device=device, dtype=torch.bfloat16)
        for _ in range(warmup):
            torch.mm(left, right, out=output)
        torch.cuda.synchronize(gpu)
        cuda_samples = []
        wall_samples = []
        for _ in range(samples):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            wall_start = time.perf_counter()
            start.record()
            for _ in range(operations):
                torch.mm(left, right, out=output)
            end.record()
            end.synchronize()
            cuda_samples.append(start.elapsed_time(end) / operations)
            wall_samples.append((time.perf_counter() - wall_start) * 1000 / operations)
    result = {
        "gpu_id": gpu,
        "gpu_name": torch.cuda.get_device_name(gpu),
        **summarize(cuda_samples),
        "host_timing": summarize(wall_samples),
        "median_window_ms": statistics.median(cuda_samples) * operations,
    }
    del left, right, output
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2,3", help="Logical GPU IDs in pipeline order")
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--ops-per-sample", type=int, default=100)
    parser.add_argument("--state", choices=("H0", "H1"), default="H0")
    parser.add_argument("--output", default="results/raw/gemm_profile.json")
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",")]
    if not gpus or len(set(gpus)) != len(gpus) or min(gpus) < 0:
        parser.error("GPU IDs must be unique and nonnegative")
    if min(args.m, args.n, args.k, args.samples, args.ops_per_sample) < 1 or args.warmup < 0:
        parser.error(
            "Dimensions, samples and operations must be positive; warmup must be nonnegative"
        )
    import torch

    if not torch.cuda.is_available() or max(gpus) >= torch.cuda.device_count():
        raise RuntimeError("Requested CUDA GPUs are unavailable")
    torch.manual_seed(1234)
    results = []
    for gpu in gpus:
        torch.cuda.set_device(gpu)
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError(f"GPU {gpu} does not support native BF16")
        result = measure_gpu(
            torch, gpu, (args.m, args.n, args.k), args.warmup, args.samples, args.ops_per_sample
        )
        results.append(result)
        print(
            f"GPU {gpu}: {result['median_ms']:.6f} ms/GEMM, window {result['median_window_ms']:.2f} ms"
        )
    write_json(
        args.output,
        {
            "unit": "ms",
            "profile_state": args.state,
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "dtype": "bfloat16",
            "seed": 1234,
            "gemm_shape": {"m": args.m, "n": args.n, "k": args.k},
            "warmup": args.warmup,
            "samples": args.samples,
            "ops_per_sample": args.ops_per_sample,
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
