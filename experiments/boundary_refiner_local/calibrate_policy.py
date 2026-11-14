"""Select policy on one dev-video half; audit it once on the other video half.

This is NOT an independent final test: the neural checkpoint was already
selected using development performance. Final held-out test remains sealed.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from interval_model import quality_decode
from prepare_data import dump, jsonl
from train import predictions_from_batch, summarize


def cached_batch(items):
    count, size = len(items), max(len(x["edges"]) for x in items)
    batch = {"edges": torch.zeros(count, size), "edge_mask": torch.zeros(count, size, dtype=torch.bool),
             "mask": torch.zeros(count, size - 1, dtype=torch.bool)}
    output = {"start": torch.full((count, size), -1e4), "end": torch.full((count, size), -1e4)}
    is_pair = "iou_quality" in items[0]
    if is_pair:
        for name in ("iou_quality", "coverage_quality"):
            output[name] = torch.full((count, size, size), -1e4)
        output["pair_mask"] = torch.zeros(count, size, size, dtype=torch.bool)
        output["outside"] = torch.stack([x["outside"] for x in items])
    for key in ("coarse", "target", "window"):
        batch[key] = torch.stack([x[key] for x in items])
    for key in ("case_id", "video_id", "kind"):
        batch[key] = [x[key] for x in items]
    for i, row in enumerate(items):
        n = len(row["edges"])
        batch["edges"][i, :n] = row["edges"]
        batch["edge_mask"][i, :n] = True
        batch["mask"][i, :n - 1] = True
        for key in ("start", "end"):
            output[key][i, :n] = row[key]
        if is_pair:
            for key in ("iou_quality", "coverage_quality", "pair_mask"):
                output[key][i, :n, :n] = row[key]
    return output, batch


def policy_grid(is_pair):
    yield {"mode": "coarse"}
    for q in (.05, .1, .15, .2, .25, .3, .4):
        yield {"mode": "quantile", "q": q}
    if is_pair:
        for weight in (0., .25, .5):
            yield {"mode": "quality", "coverage_weight": weight, "fallback": False}
            for margin in (0., .02, .05):
                for outside in (.25, .5, .75, 1.1):
                    yield {"mode": "quality", "coverage_weight": weight, "margin": margin,
                           "outside_threshold": outside, "fallback": True}


@torch.inference_mode()
def apply_policy(items, config):
    rows = []
    for offset in range(0, len(items), 64):
        output, batch = cached_batch(items[offset:offset + 64])
        mode = config["mode"]
        if mode == "coarse":
            prediction = batch["coarse"]
        elif mode == "quantile":
            size = batch["edges"].shape[-1]
            i = (output["start"].softmax(-1).cumsum(-1) < config["q"]).sum(-1).clamp(max=size - 1)
            j = (output["end"].softmax(-1).cumsum(-1) < 1 - config["q"]).sum(-1).clamp(max=size - 1)
            ids = torch.arange(len(i))
            proposal = torch.stack([batch["edges"][ids, i], batch["edges"][ids, j]], -1)
            prediction = torch.where((j > i)[:, None], proposal, batch["coarse"])
        else:
            prediction, _ = quality_decode(output, batch, **{k: v for k, v in config.items() if k != "mode"})
        result = predictions_from_batch(prediction, batch)
        for row in result:
            row["intervened"] = max(abs(a - b) for a, b in zip(row["prediction"], row["coarse"])) > .1
        rows.extend(result)
    summary = summarize(rows)
    summary["intervention_rate"] = float(np.mean([x["intervened"] for x in rows]))
    return summary, rows


def bootstrap_delta(first, second, metric, repeats=2000):
    a = {(x["case_id"], x["kind"]): x for x in first}
    b = {(x["case_id"], x["kind"]): x for x in second}
    if a.keys() != b.keys():
        raise ValueError("Unpaired calibration audit")
    grouped = {}
    for key, row in a.items():
        grouped.setdefault(row["video_id"], []).append(row["metrics"][metric] - b[key]["metrics"][metric])
    sums = np.array([sum(x) for x in grouped.values()])
    counts = np.array([len(x) for x in grouped.values()])
    indices = np.random.default_rng(187).integers(0, len(sums), (repeats, len(sums)))
    values = sums[indices].sum(1) / counts[indices].sum(1)
    return {"mean": float(sums.sum() / counts.sum()), "source_video_cluster_95pct": np.quantile(values, [.025, .975]).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(3)
    if args.output.exists():
        raise ValueError("Preserve earlier calibration snapshots")
    data = torch.load(args.cache / "outputs.pt", map_location="cpu", weights_only=True)
    videos = sorted({x["video_id"] for x in data}, key=lambda x: hashlib.sha256(("policy-calib-v1:" + x).encode()).hexdigest())
    calibration_ids = set(videos[:len(videos) // 2])
    calibration = [x for x in data if x["video_id"] in calibration_ids]
    validation = [x for x in data if x["video_id"] not in calibration_ids]
    baseline, _ = apply_policy(calibration, {"mode": "coarse"})
    best, trials = None, []
    for config in policy_grid("iou_quality" in data[0]):
        metrics, _ = apply_policy(calibration, config)
        feasible = (metrics["coverage_at_90"] >= baseline["coverage_at_90"]
                    and metrics["event_coverage"] >= baseline["event_coverage"]
                    and metrics["redundancy_seconds"] <= baseline["redundancy_seconds"]
                    and (metrics["intervention_rate"] >= .1 or config["mode"] == "coarse"))
        trial = {"policy": config, "calibration": metrics, "feasible": feasible}
        trials.append(trial)
        if feasible and (best is None or metrics["iou"] > best["calibration"]["iou"]):
            best = trial
    chosen, predictions = apply_policy(validation, best["policy"])
    original, coarse = apply_policy(validation, {"mode": "coarse"})
    delta = {key: bootstrap_delta(predictions, coarse, key) for key in ("iou", "coverage_at_90", "event_coverage", "redundancy_seconds")}
    report = {"scope": "policy-calibration video split inside previously used development set; not final test",
              "cache": str(args.cache), "test_accessed": False, "calibration_video_ids": sorted(calibration_ids),
              "validation_video_ids": sorted(set(videos) - calibration_ids), "selected_policy": best["policy"],
              "calibration_trials": trials, "validation": chosen, "coarse_validation": original, "paired_delta": delta,
              "passes_observed_guardrail": chosen["iou"] > original["iou"] and chosen["coverage_at_90"] >= original["coverage_at_90"]
                    and chosen["event_coverage"] >= original["event_coverage"] and chosen["redundancy_seconds"] <= original["redundancy_seconds"]
                    and chosen["intervention_rate"] >= .1}
    args.output.mkdir(parents=True)
    dump(args.output / "calibration.json", report)
    jsonl(args.output / "validation_predictions.jsonl", predictions)
    jsonl(args.output / "coarse_validation_predictions.jsonl", coarse)
    print(json.dumps({"policy": best["policy"], "validation": {k: chosen[k] for k in
         ("iou", "coverage_at_90", "event_coverage", "redundancy_seconds", "intervention_rate")},
         "paired_delta": delta, "observed_guardrail": report["passes_observed_guardrail"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
