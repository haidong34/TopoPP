"""Generate balanced layer-count sweeps without launching or stopping any server."""

import argparse
from cost_model import default_partition, positive_int
from profile_common import read_json, write_json


def make_sweeps(layers, stages, deltas):
    baseline = default_partition(layers, stages)
    candidates = [baseline]
    if stages == 1:
        return candidates
    pattern = [1 if i % 2 == 0 else -1 for i in range(stages)]
    if stages % 2:
        pattern[-1] = 0
    for delta in deltas:
        positive_int(delta, "delta")
        for shift in range(stages):
            moved = [baseline[i] + delta * pattern[(i + shift) % stages] for i in range(stages)]
            if min(moved) < 1:
                raise ValueError(f"delta={delta} produces empty stages; use a smaller delta")
            if moved not in candidates:
                candidates.append(moved)
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--stages", type=int, default=4)
    parser.add_argument(
        "--deltas",
        default="4",
        help="Measured domain only; choose from model layer count, e.g. 4 or 2,4,6",
    )
    parser.add_argument("--output", default="configs/sweeps.json")
    args = parser.parse_args()
    import os

    layers = read_json(os.path.join(args.model, "config.json"))["num_hidden_layers"]
    sweeps = make_sweeps(layers, args.stages, [int(d) for d in args.deltas.split(",")])
    write_json(
        args.output,
        {
            "num_layers": layers,
            "partitions": sweeps,
            "per_stage_measured_counts": [
                sorted({p[i] for p in sweeps}) for i in range(args.stages)
            ],
        },
    )
    for i, counts in enumerate(sweeps):
        print(f"run{i}:", ",".join(map(str, counts)))
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
