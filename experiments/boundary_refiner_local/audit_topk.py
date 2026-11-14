"""Proposal recall/context ceilings, NOT achieved model or editing performance."""

import argparse
import json
from pathlib import Path

import numpy as np

from prepare_data import DEFAULT_ROOT, dump, read_jsonl
from train import interval_metrics


def merged_length(windows):
    total, lo, hi = 0., None, None
    for a, b in sorted(windows):
        if lo is None:
            lo, hi = a, b
        elif a <= hi:
            hi = max(hi, b)
        else:
            total += hi - lo
            lo, hi = a, b
    return total + (hi - lo if lo is not None else 0.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--old-top1", type=Path, required=True)
    args = parser.parse_args()
    truth = {x["case_id"]: x for x in read_jsonl(args.root / "manifests/dev.complete.jsonl")}
    proposals = read_jsonl(args.candidates / "candidates.jsonl")
    original = {x["case_id"]: x for x in read_jsonl(args.old_top1 / "candidates.jsonl")}
    result = {}
    for k in (1, 2, 3, 5, 10):
        best_iou, whole, potential, observed = [], [], [], []
        for row in proposals:
            old = original[row["case_id"]]
            if not np.allclose(row["coarse"], old["coarse"], atol=1e-5):
                raise ValueError("Recomputed coarse top1 changed; do not mix baseline snapshots")
            gt = truth[row["case_id"]]
            start, end = gt["target_interval_seconds"]
            selected = row["proposals"][:k]
            windows = [(max(0., x["start"] - 6), min(gt["duration_seconds"], x["end"] + 6)) for x in selected]
            best_iou.append(max(interval_metrics([x["start"], x["end"]], [start, end])["iou"] for x in selected))
            whole.append(any(a <= start < end <= b for a, b in windows))
            potential.append(max(max(0., min(b, end) - max(a, start)) / (end - start) for a, b in windows))
            observed.append(merged_length(windows))
        result[k] = {"oracle_best_candidate_mean_iou": float(np.mean(best_iou)),
                     "proposal_recall_iou50": float(np.mean(np.asarray(best_iou) >= .5)),
                     "target_fully_observable_in_one_window_fraction": float(np.mean(whole)),
                     "oracle_refinement_iou_ceiling_within_one_window": float(np.mean(potential)),
                     "mean_unique_observation_seconds": float(np.mean(observed))}
    report = {"scope": "development oracle diagnosis only; candidate selection has NOT achieved these ceilings",
              "test_accessed": False, "queries": len(proposals), "top1_numerically_matches_previous_snapshot": True,
              "topk": result, "source_video_ids": sorted({x["video_id"] for x in proposals})}
    dump(args.candidates / "topk_audit.json", report)
    print(json.dumps({"scope": report["scope"], "topk": result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
