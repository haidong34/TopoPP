"""Build CUDA-event stage cost curves; keep host spans as diagnostics, never as local GPU cost."""

import argparse
from collections import defaultdict
import math
from pathlib import Path
import statistics

from cost_model import positive_int
from profile_common import read_json, write_json


def bucket(record, width):
    qs, contexts = record["query_tokens"], record["context_lengths"]
    return (
        record["phase"],
        record["batch_size"],
        sum(qs),
        min(qs),
        max(qs),
        min(contexts) // width,
        max(contexts) // width,
    )


def build_profile(
    records, memory_profile, weight_run_id, width=128, min_samples=5, min_counts=2
):
    for value in (width, min_samples, min_counts):
        positive_int(value, "bucket/sample/count parameter")
    if not records:
        raise ValueError("No trace records")
    first = records[0]
    invariants = (
        "num_layers",
        "pp_size",
        "config_sha256",
        "enforce_eager",
        "vllm",
        "torch",
        "profile_state",
        "CUDA_VISIBLE_DEVICES",
        "sample_every",
        "model_geometry",
        "execution_signature",
    )
    for row in records:
        if any(row[key] != first[key] for key in invariants):
            raise ValueError(
                "Do not mix model/state/backend/sampling settings across trace runs"
            )
        if (
            len(row["query_tokens"]) != row["batch_size"]
            or len(row["context_lengths"]) != row["batch_size"]
        ):
            raise ValueError("Invalid trace batch metadata")
    if (
        memory_profile["num_layers"] != first["num_layers"]
        or memory_profile["profile_state"] != first["profile_state"]
    ):
        raise ValueError("Memory profile model layer count/state mismatch")
    if (
        memory_profile.get("complete") is False
        or memory_profile.get("model_geometry") != first["model_geometry"]
    ):
        raise ValueError("Memory profile incomplete or model geometry differs")
    p = first["pp_size"]
    if len(memory_profile["gpu_ids"]) != p:
        raise ValueError("Memory profile stage count mismatch")
    reference_counts, groups = defaultdict(int), defaultdict(list)
    host_groups = defaultdict(list)
    transfer_shapes = defaultdict(set)
    reference_seen = False
    for row in records:
        rank = row["pp_rank"]
        if not 0 <= rank < p or row["gpu_id"] != memory_profile["gpu_ids"][rank]:
            raise ValueError("Trace GPU mapping differs from memory profile")
        key = bucket(row, width)
        # recv may return on the host before its queued GPU work completes.
        # Last-rank host spans then include upstream waits at GPU->CPU sync.
        # CUDA events on the ordered stream exclude the prefix before start.
        value = float(row["local_cuda_ms"])
        host_value = float(row["local_wall_ms"])
        if (
            value <= 0
            or not math.isfinite(value)
            or host_value < 0
            or not math.isfinite(host_value)
        ):
            raise ValueError("Invalid local-stage timing")
        groups[(key, rank, row["stage_layers"])].append(value)
        host_groups[(key, rank, row["stage_layers"])].append(host_value)
        if row.get("intermediate_shapes"):
            transfer_shapes[(key, rank)].add(
                tuple(tuple(shape) for shape in row["intermediate_shapes"])
            )
        if row["run_id"] == weight_run_id and rank == 0:
            reference_counts[key] += 1
            reference_seen = True
    if not reference_seen:
        raise ValueError("--weight-run-id has no rank-0 records")
    scenarios, covered = [], 0
    for key, weight in sorted(reference_counts.items()):
        rows = []
        for rank in range(p):
            points = {
                str(n): statistics.median(values)
                for (k, r, n), values in groups.items()
                if k == key and r == rank and len(values) >= min_samples
            }
            if len(points) < min_counts:
                break
            shapes = transfer_shapes[(key, rank)]
            host_points = {
                str(n): statistics.median(host_groups[(key, rank, n)]) for n in map(int, points)
            }
            rows.append(
                {
                    "gpu_id": memory_profile["gpu_ids"][rank],
                    "total_curve_ms": points,
                    "host_curve_ms_diagnostic_only": host_points,
                    "communication_shapes": (
                        list(next(iter(shapes))) if len(shapes) == 1 else None
                    ),
                    "communication_ms": 0,
                }
            )
        if len(rows) != p:
            continue
        phase, batch, tokens, qmin, qmax, cmin, cmax = key
        scenario_id = f"{phase}_b{batch}_q{tokens}_qrange{qmin}-{qmax}_ctxbins{cmin}-{cmax}"
        scenarios.append(
            {
                "id": scenario_id,
                "phase": phase,
                "batch_size": batch,
                "query_token_sum": tokens,
                "context_bin_width": width,
                "context_min_bin": cmin,
                "context_max_bin": cmax,
                "weight": weight,
                "stages": rows,
            }
        )
        covered += weight
    if not scenarios:
        raise ValueError(
            "No workload bucket has sufficient samples and layer counts on every rank; collect more partition sweeps or inspect --min-samples/--context-bin"
        )
    coverage = covered / sum(reference_counts.values())
    warnings = [
        "Stage costs use local CUDA Event spans, including timed-stream sampling and launch gaps; pure CPU costs are not fully modeled.",
        "PP receive/send and global scheduler latency are excluded; throughput objective remains a surrogate.",
        "Host spans may include upstream waits at GPU->CPU synchronization; retained as diagnostics, never maxed/added into GPU stage cost.",
        "Measurements from instrumented runs may perturb scheduling; final A/B must use the original worker.",
        "Memory comes from the provided geometric estimate; validate actual vLLM KV capacity.",
    ]
    if coverage < 1:
        warnings.append(
            f"Only {coverage:.1%} of reference sampled iterations are covered; remaining buckets were dropped and weights renormalized."
        )
    return {
        "schema_version": 2,
        "complete": True,
        "num_layers": first["num_layers"],
        "gpu_ids": memory_profile["gpu_ids"],
        "profile_state": first["profile_state"],
        "weight_source": "serving_trace_counts",
        "backend": "vllm_0.10.1.1_v1_local_runner",
        "timing_metric": "local_cuda_ms",
        "cost_model_revision": "cuda_event_cost_v2",
        "reference_run_id": weight_run_id,
        "reference_iteration_coverage": coverage,
        "trace_config_sha256": first["config_sha256"],
        "model_geometry": first["model_geometry"],
        "memory": memory_profile["memory"],
        "scenarios": scenarios,
        "warnings": warnings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-dir", required=True, help="Root directory of runs; scans *.jsonl recursively"
    )
    parser.add_argument(
        "--memory-from", required=True, help="Same model/H0-H1/GPU mapping block profile"
    )
    parser.add_argument(
        "--weight-run-id", required=True, help="One fixed default-partition benchmark run"
    )
    parser.add_argument("--context-bin", type=int, default=128)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--min-counts", type=int, default=2)
    parser.add_argument("--min-coverage", type=float, default=0.9)
    parser.add_argument("--output", default="results/raw/vllm_stage_h1.json")
    args = parser.parse_args()
    if not 0 < args.min_coverage <= 1:
        parser.error("--min-coverage must be in (0,1]")
    import json

    records = []
    for path in sorted(Path(args.trace_dir).rglob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Incomplete/corrupt trace {path}:{number}; stop serving before aggregation"
                    ) from error
    output = build_profile(
        records,
        read_json(args.memory_from),
        args.weight_run_id,
        args.context_bin,
        args.min_samples,
        args.min_counts,
    )
    if output["reference_iteration_coverage"] < args.min_coverage:
        raise ValueError(
            "Insufficient reference coverage; collect additional profiles or explicitly lower --min-coverage"
        )
    write_json(args.output, output)
    print(
        "Workloads:",
        len(output["scenarios"]),
        "coverage:",
        output["reference_iteration_coverage"],
    )
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
