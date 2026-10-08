"""Search contiguous partitions using workload-weighted stage bottlenecks."""

import argparse
import heapq
import itertools
import math
from pathlib import Path

from cost_model import CostModel, default_partition, positive_int
from profile_common import read_json, write_json


def partitions(layers, stages):
    for cuts in itertools.combinations(range(1, layers), stages - 1):
        edges = (0, *cuts, layers)
        yield tuple(right - left for left, right in zip(edges, edges[1:]))


def stage_tables(model):
    tables = []
    for stage in range(model.stages):
        by_layer_count = {}
        for count in range(1, model.layers - model.stages + 2):
            values = [
                model.stage_cost(scenario, stage, count)
                for scenario in range(len(model.scenarios))
            ]
            if all(value is not None for value in values):
                by_layer_count[count] = values
        tables.append(by_layer_count)
    return tables


def solve_v2(model, top_k=5, max_candidates=2_000_000):
    positive_int(top_k, "top_k")
    positive_int(max_candidates, "max_candidates")
    candidate_count = math.comb(model.layers - 1, model.stages - 1)
    if candidate_count > max_candidates:
        raise ValueError(f"{candidate_count:,} candidates exceed limit {max_candidates:,}")
    tables = stage_tables(model)
    best = []
    feasible = 0
    for counts in partitions(model.layers, model.stages):
        if any(count not in tables[stage] for stage, count in enumerate(counts)):
            continue
        feasible += 1
        score = sum(
            weight * max(tables[stage][count][scenario] for stage, count in enumerate(counts))
            for scenario, weight in enumerate(model.weights)
        )
        # Negated keys keep the worst retained candidate at the heap root.
        entry = (-score, tuple(-count for count in counts), counts)
        if len(best) < top_k:
            heapq.heappush(best, entry)
        elif entry > best[0]:
            heapq.heapreplace(best, entry)
    if not best:
        raise ValueError("No feasible partition within the measured domain and memory budgets")
    ranked = sorted(
        (model.evaluate(entry[2]) for entry in best),
        key=lambda result: (result["objective_ms"], result["partition"]),
    )
    return ranked, {
        "enumerated": candidate_count,
        "feasible": feasible,
        "exact_within": (
            "fixed GPU order, measured/interpolated curve domain, "
            "configured memory constraints and surrogate objective"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--num-layers", type=int, help="Must match the profile")
    parser.add_argument("--supplemental", help="Matching state and scenario cost supplements")
    parser.add_argument("--baseline", help="Layer counts in pipeline order")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=2_000_000)
    parser.add_argument("--output", default="results/raw/partition.json")
    args = parser.parse_args()
    profile = read_json(args.profile)
    supplemental = read_json(args.supplemental) if args.supplemental else None
    model = CostModel(profile, supplemental)
    if args.num_layers is not None and args.num_layers != model.layers:
        parser.error("--num-layers does not match the profile")
    baseline_counts = (
        [int(value) for value in args.baseline.split(",")]
        if args.baseline
        else default_partition(model.layers, model.stages)
    )
    baseline = model.evaluate(baseline_counts)
    ranked, search = solve_v2(model, args.top_k, args.max_candidates)
    warnings = list(model.warnings)
    reduction = None
    if baseline is None:
        warnings.append("Baseline is outside the profile domain or memory budget")
    else:
        reduction = 100 * (1 - ranked[0]["objective_ms"] / baseline["objective_ms"])
    write_json(
        args.output,
        {
            "schema_version": 2,
            "num_layers": model.layers,
            "gpu_ids": model.gpu_ids,
            "profile_state": profile.get("profile_state"),
            "objective": "sum_s normalized_weight_s * max_i stage_cost_i_s",
            "baseline_partition": baseline_counts,
            "baseline": baseline,
            "topopp": ranked[0],
            "candidates": ranked,
            "search": search,
            "warnings": warnings,
            "predicted_surrogate_reduction_percent": reduction,
        },
    )
    export = (
        'export VLLM_PP_LAYER_PARTITION="' + ",".join(map(str, ranked[0]["partition"])) + '"\n'
    )
    Path(args.output).with_suffix(".env").write_text(export, encoding="utf-8")
    print("Partition:", ranked[0]["partition"])
    for warning in warnings:
        print("Warning:", warning)


if __name__ == "__main__":
    main()
