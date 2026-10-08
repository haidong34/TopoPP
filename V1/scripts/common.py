"""Profile storage and validation."""

import json
import math
from pathlib import Path


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def positive_number(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(target)


def load_gpu_times(profile):
    rows = profile["results"]
    gpu_ids = [row.get("gpu", row.get("gpu_id", rank)) for rank, row in enumerate(rows)]
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("GPU IDs must be nonempty and unique")
    if any(isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0 for gpu in gpu_ids):
        raise ValueError("GPU IDs must be nonnegative integers")
    unit = profile.get("unit", "ms")
    if unit not in ("ms", "relative"):
        raise ValueError("Profile unit must be ms or relative")
    field = "median_ms" if unit == "ms" else "relative_time"
    times = [positive_number(row[field], field) for row in rows]
    return gpu_ids, times, unit
