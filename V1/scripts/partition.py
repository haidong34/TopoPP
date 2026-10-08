"""Partition contiguous layers using a per-GPU GEMM latency proxy."""

import argparse
import math

from common import load_gpu_times, positive_int, positive_number, read_json, write_json


def solve_partition(num_layers, gpu_times):
    positive_int(num_layers, "num_layers")
    times = [positive_number(value, "GPU time") for value in gpu_times]
    stages = len(times)
    if stages == 0 or num_layers < stages:
        raise ValueError("Each stage requires at least one layer")

    cost = [[math.inf] * (num_layers + 1) for _ in range(stages + 1)]
    previous_cut = [[-1] * (num_layers + 1) for _ in range(stages + 1)]
    cost[0][0] = 0.0

    for stage_count in range(1, stages + 1):
        last_layer = num_layers - (stages - stage_count)
        for end in range(stage_count, last_layer + 1):
            for start in range(stage_count - 1, end):
                stage_cost = (end - start) * times[stage_count - 1]
                bottleneck = max(cost[stage_count - 1][start], stage_cost)
                if bottleneck < cost[stage_count][end]:
                    cost[stage_count][end] = bottleneck
                    previous_cut[stage_count][end] = start

    partition = []
    end = num_layers
    for stage_count in range(stages, 0, -1):
        start = previous_cut[stage_count][end]
        if start < 0:
            raise RuntimeError("No feasible partition")
        partition.append(end - start)
        end = start
    partition.reverse()

    ranges = []
    start = 0
    for count in partition:
        ranges.append([start, start + count])
        start += count
    stage_costs = [count * latency for count, latency in zip(partition, times)]
    return {
        "partition": partition,
        "ranges": ranges,
        "predicted_stage_costs": stage_costs,
        "bottleneck": max(stage_costs),
    }


def default_partition(layers, stages):
    positive_int(layers, "layers")
    positive_int(stages, "stages")
    if layers < stages:
        raise ValueError("Each stage requires at least one layer")
    counts = [layers // stages] * stages
    for offset in range(2, layers % stages + 2):
        counts[-offset] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--num-layers", type=int, default=28)
    parser.add_argument("--baseline", help="Layer counts in pipeline order")
    parser.add_argument("--output", default="results/raw/partition.json")
    args = parser.parse_args()

    gpu_ids, times, unit = load_gpu_times(read_json(args.profile))
    best = solve_partition(args.num_layers, times)
    baseline = (
        [int(value) for value in args.baseline.split(",")]
        if args.baseline
        else default_partition(args.num_layers, len(times))
    )
    if len(baseline) != len(times) or sum(baseline) != args.num_layers:
        raise ValueError("Baseline stage count or layer sum mismatch")
    for count in baseline:
        positive_int(count, "baseline count")
    baseline_costs = [count * latency for count, latency in zip(baseline, times)]
    result = {
        "num_layers": args.num_layers,
        "gpu_ids": gpu_ids,
        "unit": unit,
        "gpu_times": times,
        "default_partition": baseline,
        "default_stage_costs": baseline_costs,
        "default_bottleneck": max(baseline_costs),
        "topopp": best,
        "predicted_bottleneck_reduction_percent": 100
        * (1 - best["bottleneck"] / max(baseline_costs)),
        "objective": "minimize max_i layer_count_i * GEMM_latency_i",
    }
    write_json(args.output, result)
    print("Partition:", best["partition"])
    print(
        "Predicted bottleneck reduction (%):", result["predicted_bottleneck_reduction_percent"]
    )


if __name__ == "__main__":
    main()
