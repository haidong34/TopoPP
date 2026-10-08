"""Validated, dependency-free stage cost tables for TopoPP V2."""

import bisect
import copy
import math


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def nonnegative(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite nonnegative number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return value


def default_partition(layers, stages):
    """Match vLLM 0.10.1.1 get_pp_indices (remaining layers from penultimate back)."""
    positive_int(layers, "layers")
    positive_int(stages, "stages")
    if layers < stages:
        raise ValueError("Each stage needs at least one layer")
    result = [layers // stages] * stages
    for offset in range(2, layers % stages + 2):
        result[-offset] += 1
    return result


def interpolate(points, n):
    """Interpolate only inside the measured domain; never extrapolate."""
    counts = sorted(points)
    if not counts or n < counts[0] or n > counts[-1]:
        return None
    if n in points:
        return points[n]
    upper = bisect.bisect_left(counts, n)
    lo, hi = counts[upper - 1], counts[upper]
    return points[lo] + (points[hi] - points[lo]) * (n - lo) / (hi - lo)


class CostModel:
    def __init__(self, profile, supplemental=None):
        if profile.get("schema_version") != 2:
            raise ValueError("Expected schema_version=2")
        if profile.get("complete") is False:
            raise ValueError("Incomplete profiling output cannot be used for planning")
        self.layers = positive_int(profile["num_layers"], "num_layers")
        self.gpu_ids = profile["gpu_ids"]
        if not self.gpu_ids or len(set(self.gpu_ids)) != len(self.gpu_ids):
            raise ValueError("gpu_ids must be nonempty and unique, in PP order")
        if any(isinstance(g, bool) or not isinstance(g, int) or g < 0 for g in self.gpu_ids):
            raise ValueError("gpu_ids must be nonnegative integers")
        self.stages = len(self.gpu_ids)
        if self.layers < self.stages:
            raise ValueError("num_layers must be >= stage count")
        self.warnings = list(profile.get("warnings", []))
        self.scenarios = profile["scenarios"]
        if not self.scenarios:
            raise ValueError("No workload scenarios")
        ids = [s["id"] for s in self.scenarios]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate workload ids")
        raw_weights = [nonnegative(s["weight"], "weight") for s in self.scenarios]
        if sum(raw_weights) <= 0 or not math.isfinite(sum(raw_weights)):
            raise ValueError("At least one scenario must have positive weight")
        self.weights = [w / sum(raw_weights) for w in raw_weights]
        self.tables = self._parse_tables()
        self._apply_supplemental(profile, supplemental)
        self._load_memory(profile)
        if profile.get("weight_source") != "serving_trace_counts":
            self.warnings.append(
                "Workload weights are manual/unverified; they are not inferred from concurrency or token lengths."
            )

    def _parse_tables(self):
        tables = []
        for scenario in self.scenarios:
            rows = scenario["stages"]
            if [r["gpu_id"] for r in rows] != self.gpu_ids:
                raise ValueError(f"{scenario['id']}: stage order must match gpu_ids")
            parsed = []
            for row in rows:
                if ("block_curve_ms" in row) == ("total_curve_ms" in row):
                    raise ValueError("Exactly one of block_curve_ms / total_curve_ms required")
                total_curve = "total_curve_ms" in row
                if total_curve and row.get("fixed_ms"):
                    raise ValueError("total_curve_ms already includes local stage overhead")
                curve = {}
                for key, value in row.get("block_curve_ms", row.get("total_curve_ms")).items():
                    n = positive_int(int(key), "curve layer count")
                    if n in curve:
                        raise ValueError("Duplicate curve layer counts")
                    curve[n] = nonnegative(value, "block cost")
                if not curve or any(value <= 0 for value in curve.values()):
                    raise ValueError("Block costs must be strictly positive")
                components = {
                    name: nonnegative(value, name)
                    for name, value in row.get("fixed_ms", {}).items()
                }
                parsed.append(
                    {
                        "curve": curve,
                        "fixed": components,
                        "total_curve": total_curve,
                        "communication_ms": nonnegative(
                            row.get("communication_ms", 0), "communication_ms"
                        ),
                    }
                )
            tables.append(parsed)
        return tables

    def _apply_supplemental(self, profile, supplemental):
        if supplemental is None:
            return
        ids = [scenario["id"] for scenario in self.scenarios]
        if supplemental.get("profile_state") != profile.get("profile_state"):
            raise ValueError("Supplemental profile_state must match H0/H1 profile")
        for scenario_id, rows in supplemental.get("scenarios", {}).items():
            if scenario_id not in ids or len(rows) != self.stages:
                raise ValueError("Supplemental scenario/stage count mismatch")
            table = self.tables[ids.index(scenario_id)]
            for i, row in enumerate(rows):
                if row.get("gpu_id") != self.gpu_ids[i]:
                    raise ValueError("Supplemental gpu_id order mismatch")
                if table[i]["total_curve"] and (
                    row.get("replace_fixed_ms") or row.get("runtime_extra_ms", 0)
                ):
                    raise ValueError(
                        "Do not add local overhead to total stage curves; communication only"
                    )
                # Explicit replacements prevent double-counting measured components.
                for name, value in row.get("replace_fixed_ms", {}).items():
                    table[i]["fixed"][name] = nonnegative(value, name)
                table[i]["fixed"]["runtime_extra"] = nonnegative(
                    row.get("runtime_extra_ms", 0), "runtime_extra_ms"
                )
                table[i]["communication_ms"] = nonnegative(
                    row.get("communication_ms", 0), "communication_ms"
                )
        self.warnings += list(supplemental.get("warnings", []))

    def _load_memory(self, profile):
        self.memory = copy.deepcopy(profile.get("memory"))
        if self.memory is None:
            self.warnings.append("No memory model: candidate admission is unchecked.")
        else:
            if len(self.memory["stages"]) != self.stages:
                raise ValueError("Memory stage count mismatch")
            for row, gpu_id in zip(self.memory["stages"], self.gpu_ids):
                if row["gpu_id"] != gpu_id:
                    raise ValueError("Memory gpu_id order mismatch")
                for key in (
                    "budget_bytes",
                    "per_layer_weights_bytes",
                    "per_layer_kv_bytes",
                    "role_weights_bytes",
                    "runtime_reserve_bytes",
                ):
                    row[key] = nonnegative(row[key], key)

    def memory_usage(self, stage, n):
        if self.memory is None:
            return None
        row = self.memory["stages"][stage]
        return (
            n * (row["per_layer_weights_bytes"] + row["per_layer_kv_bytes"])
            + row["role_weights_bytes"]
            + row["runtime_reserve_bytes"]
        )

    def stage_cost(self, scenario, stage, n):
        row = self.tables[scenario][stage]
        block = interpolate(row["curve"], n)
        if block is None:
            return None
        usage = self.memory_usage(stage, n)
        if usage is not None and usage > self.memory["stages"][stage]["budget_bytes"]:
            return None
        return block + sum(row["fixed"].values()) + row["communication_ms"]

    def evaluate(self, partition):
        if len(partition) != self.stages or sum(partition) != self.layers:
            raise ValueError("Partition length/sum mismatch")
        for n in partition:
            positive_int(n, "partition count")
        details = []
        objective = 0.0
        for s, (scenario, weight) in enumerate(zip(self.scenarios, self.weights)):
            costs = [self.stage_cost(s, i, n) for i, n in enumerate(partition)]
            if any(c is None for c in costs):
                return None
            bottleneck = max(costs)
            objective += weight * bottleneck
            details.append(
                {
                    "id": scenario["id"],
                    "weight": weight,
                    "stage_costs_ms": costs,
                    "bottleneck_ms": bottleneck,
                    "path_cost_ms": sum(costs),
                }
            )
        starts, cursor = [], 0
        for n in partition:
            starts.append([cursor, cursor + n])
            cursor += n
        return {
            "partition": list(partition),
            "ranges": starts,
            "objective_ms": objective,
            "scenarios": details,
            "estimated_stage_memory_bytes": [
                self.memory_usage(i, n) for i, n in enumerate(partition)
            ],
        }
