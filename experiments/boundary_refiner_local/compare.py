"""Paired development comparison with source-video cluster bootstrap intervals."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from prepare_data import DEFAULT_ROOT, dump, jsonl, read_jsonl
from train import summarize


def paired_difference(first, second, repeats=2000):
    a = {(x["case_id"], x["kind"]): x for x in first}
    b = {(x["case_id"], x["kind"]): x for x in second}
    if a.keys() != b.keys():
        raise ValueError("Comparisons require identical development cases and coarse perturbations")
    grouped = defaultdict(list)
    for key in a:
        if a[key]["target"] != b[key]["target"] or a[key]["coarse"] != b[key]["coarse"]:
            raise ValueError("Changed target or observation protocol between experiments")
        grouped[a[key]["video_id"]].append(a[key]["metrics"]["iou"] - b[key]["metrics"]["iou"])
    sums = np.array([sum(x) for x in grouped.values()])
    counts = np.array([len(x) for x in grouped.values()])
    rng = np.random.default_rng(813)
    indices = rng.integers(0, len(sums), size=(repeats, len(sums)))
    boot = sums[indices].sum(1) / counts[indices].sum(1)
    return {"mean_iou_delta": float(sums.sum() / counts.sum()),
            "source_video_cluster_bootstrap_95pct": np.quantile(boot, [.025, .975]).tolist(),
            "source_videos": len(sums), "bootstrap_repeats": repeats,
            "warning": "development-set model selection; not independent final-test significance"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Comparison output already exists; use a new snapshot name")
    predictions = {"coarse": read_jsonl(args.baselines / "coarse_predictions.jsonl"),
                   "clip": read_jsonl(args.baselines / "clip_predictions.jsonl")}
    configurations = {}
    for run in args.runs:
        state = json.loads((run / "status.json").read_text(encoding="utf-8"))
        if state["status"] != "complete":
            raise ValueError(f"Training run is not terminal: {run}")
        config = json.loads((run / "configuration.json").read_text(encoding="utf-8"))
        if config["test_accessed"]:
            raise ValueError("Architecture search must not consume the final test split")
        predictions[run.name] = read_jsonl(run / "best_dev_predictions.jsonl")
        configurations[run.name] = config
    summaries = {name: summarize(rows) for name, rows in predictions.items()}
    deltas = {name: {reference: paired_difference(rows, predictions[reference]) for reference in ("coarse", "clip")}
              for name, rows in predictions.items() if name not in ("coarse", "clip")}
    # Also directly test the need for language conditioning and added complexity.
    for name, config in configurations.items():
        for ref, ref_config in configurations.items():
            if config["seed"] == ref_config["seed"] and config["variant"] != ref_config["variant"]:
                deltas[name][ref] = paired_difference(predictions[name], predictions[ref])
    best_name = max(configurations, key=lambda x: summaries[x]["iou"])
    reference = {(x["case_id"], x["kind"]): x for x in predictions["coarse"]}
    errors = []
    for row in predictions[best_name]:
        coarse = reference[row["case_id"], row["kind"]]
        errors.append({**row, "coarse_metrics": coarse["metrics"],
                       "iou_delta": row["metrics"]["iou"] - coarse["metrics"]["iou"]})
    args.output.mkdir(parents=True)
    jsonl(args.output / "worst_regressions.jsonl", sorted(errors, key=lambda x: x["iou_delta"])[:60])
    jsonl(args.output / "largest_remaining_errors.jsonl", sorted(errors, key=lambda x: x["metrics"]["iou"])[:60])
    report = {"scope": "synthetic-coarse development refinement only; not end-to-end editing",
              "final_test_accessed": False, "summaries": summaries, "paired_comparisons": deltas,
              "highest_dev_iou_run": best_name, "configurations": configurations}
    dump(args.output / "comparison.json", report)
    compact = {name: {k: round(value[k], 5) for k in ("iou", "boundary_mae_seconds", "event_coverage",
                                                    "redundancy_seconds", "coverage_at_90")}
               for name, value in summaries.items()}
    print(json.dumps({"development_comparison": compact, "highest_dev_iou_run": best_name,
                      "paired_comparisons": deltas}, indent=2), flush=True)


if __name__ == "__main__":
    main()
