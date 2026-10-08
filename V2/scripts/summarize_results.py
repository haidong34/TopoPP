"""Summarize the published, rounded per-run measurements."""

import argparse
import csv
from pathlib import Path
import statistics

from profile_common import write_json

METRICS = (
    "output_throughput_tok_s",
    "median_ttft_ms",
    "median_tpot_ms",
    "p99_itl_ms",
)


def summarize(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = {}
    for condition in sorted({row["condition"] for row in rows}):
        measurements = [row for row in rows if row["condition"] == condition]
        if [row["scheme"] for row in measurements] != ["A", "B", "B", "A", "A", "B"]:
            raise ValueError(f"{condition}: expected AB, BA, AB measurement order")
        metrics = {}
        for metric in METRICS:
            medians = {
                scheme: statistics.median(
                    float(row[metric]) for row in measurements if row["scheme"] == scheme
                )
                for scheme in ("A", "B")
            }
            metrics[metric] = {
                **medians,
                "change_percent": 100 * (medians["B"] / medians["A"] - 1),
            }
        paired_changes = []
        for start in range(0, len(measurements), 2):
            pair = {
                row["scheme"]: float(row["output_throughput_tok_s"])
                for row in measurements[start : start + 2]
            }
            paired_changes.append(100 * (pair["B"] / pair["A"] - 1))
        summary[condition] = {
            "metrics": metrics,
            "paired_throughput_change_percent": paired_changes,
            "source_precision": "Screenshot values rounded to two decimals",
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(Path(__file__).resolve().parents[1] / "results" / "reported_runs.csv"),
    )
    parser.add_argument("--output", help="Optional JSON output")
    args = parser.parse_args()
    result = summarize(args.input)
    for condition, values in result.items():
        print(condition)
        for name, metric in values["metrics"].items():
            print(
                f"  {name}: {metric['A']:.2f} -> {metric['B']:.2f} ({metric['change_percent']:+.2f}%)"
            )
        paired = ", ".join(
            f"{change:+.2f}%" for change in values["paired_throughput_change_percent"]
        )
        print("  Paired throughput changes:", paired)
    if args.output:
        write_json(args.output, result)


if __name__ == "__main__":
    main()
