"""Check the dynamic program against exhaustive contiguous partitions."""

import argparse
from itertools import combinations
import math

from common import load_gpu_times, positive_int, read_json
from partition import solve_partition


def exhaustive_optimum(layers, times, max_candidates=2_000_000):
    positive_int(layers, "layers")
    positive_int(max_candidates, "max_candidates")
    if not times or layers < len(times):
        raise ValueError("Each stage requires at least one layer")
    candidates = math.comb(layers - 1, len(times) - 1)
    if candidates > max_candidates:
        raise ValueError(f"{candidates} candidates exceed limit {max_candidates}")
    best_cost = math.inf
    best_partition = None
    for cuts in combinations(range(1, layers), len(times) - 1):
        edges = (0, *cuts, layers)
        counts = [right - left for left, right in zip(edges, edges[1:])]
        cost = max(count * latency for count, latency in zip(counts, times))
        if cost < best_cost:
            best_cost = cost
            best_partition = counts
    return best_partition, best_cost, candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--num-layers", type=int, default=28)
    parser.add_argument("--max-candidates", type=int, default=2_000_000)
    args = parser.parse_args()
    _, times, _ = load_gpu_times(read_json(args.profile))
    dp_result = solve_partition(args.num_layers, times)
    counts, cost, candidates = exhaustive_optimum(args.num_layers, times, args.max_candidates)
    if not math.isclose(cost, dp_result["bottleneck"], rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError("Dynamic program disagrees with exhaustive search")
    print(f"Verified {candidates} candidates; optimum {counts}; cost {cost:.6f}")


if __name__ == "__main__":
    main()
