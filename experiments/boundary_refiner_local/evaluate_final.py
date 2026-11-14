"""Evaluate a previously frozen three-seed pipeline exactly once on held-out data.

Never trains, tunes thresholds, selects seeds or changes the frozen protocol.
Results concern annotated temporal intervals, NOT Agent semantic/export success.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from calibrate_policy import bootstrap_delta
from candidate_model import CandidateRanker, collate_candidate_sets
from extract_features import PROFILE
from final_protocol import load_frozen_protocol
from prepare_data import DEFAULT_ROOT, dump, file_hash, jsonl, read_jsonl
from run_candidate_study import averaged_metrics, baseline_rows
from train import interval_metrics, summarize
from train_candidate import evaluate_candidate


def original_coarse(data):
    rows = []
    for item in data:
        pred, target = item["coarse"][0].tolist(), item["target"].tolist()
        rows.append({"case_id": item["case_id"], "video_id": item["video_id"], "kind": "proposal_set",
                     "prediction": pred, "target": target, "coarse": pred, "metrics": interval_metrics(pred, target)})
    return rows


def observation_audit(data, root):
    cases = {x["case_id"]: x for x in read_jsonl(root / "manifests/test.complete.jsonl")}
    cache, coarse_n, fine1, fine3, total = {}, [], [], [], []
    for item in data:
        vid = item["video_id"]
        if vid not in cache:
            with np.load(root / "features" / PROFILE / "videos" / f"{vid}.npz", allow_pickle=False) as arrays:
                cache[vid] = arrays["times"].copy()
        times = cache[vid]
        centers = np.arange(1., cases[item["case_id"]]["duration_seconds"], 2.)
        indices = np.abs(times[:, None] - centers[None]).argmin(0)
        coarse = np.zeros(len(times), dtype=bool)
        coarse[indices] = True
        observed = np.zeros(len(times), dtype=bool)
        first = None
        for lo, hi in item["window"].tolist():
            window = (times >= lo) & (times < hi)
            if first is None:
                first = window
            observed |= window
        coarse_n.append(int(coarse.sum()))
        fine1.append(int(first.sum()))
        fine3.append(int(observed.sum()))
        total.append(int((observed | coarse).sum()))
    return {"interpretation": "per-query unique observations used by inference; offline cache encoded full sources, not a latency/API-cost benchmark",
            "mean_coarse_frames": float(np.mean(coarse_n)), "mean_top1_local_frames": float(np.mean(fine1)),
            "mean_top3_union_local_frames": float(np.mean(fine3)), "mean_coarse_and_top3_union": float(np.mean(total))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_frozen_protocol(args.protocol, args.root)
    if args.output.exists():
        raise ValueError("Final evaluation output exists. Do not overwrite or retune after seeing it.")
    torch.set_num_threads(3)
    expected = {x["case_id"] for x in read_jsonl(args.root / "manifests/test.complete.jsonl")}
    methods = {name: [] for name in ("coarse_top1", "fine_top1", "frozen_quality_top3", "trained_top3")}
    per_seed, artifacts, observation = {}, {}, None
    args.output.mkdir(parents=True)
    for pipeline in protocol["pipelines"]:
        seed = pipeline["seed"]
        folder = args.features_root / f"features_s{seed}"
        config = json.loads((folder / "test_configuration.json").read_text())
        if config["encoder_checkpoint_sha256"] != pipeline["local_sha256"] or config["max_k"] != protocol["top_k"]:
            raise ValueError("Final local features differ from frozen pipeline")
        if config["metadata_fields"] != protocol["metadata_fields"] or config["decoder"] != "quantile_q0.3":
            raise ValueError("Final feature schema or decoder changed")
        data = torch.load(folder / "test.pt", map_location="cpu", weights_only=True)
        if {x["case_id"] for x in data} != expected or len(data) != len(expected):
            raise ValueError("Missing/duplicated final-test cases; no silent exclusion")
        saved = torch.load(pipeline["ranker_checkpoint"], map_location="cpu", weights_only=True)
        model = CandidateRanker(protocol["ranker_variant"])
        model.load_state_dict(saved["model"], strict=True)
        loader = DataLoader(data, batch_size=256, collate_fn=collate_candidate_sets)
        _, selected = evaluate_candidate(model, loader, "cpu")
        rows = {"coarse_top1": original_coarse(data), "fine_top1": baseline_rows(data, "top1"),
                "frozen_quality_top3": baseline_rows(data, "frozen_quality"), "trained_top3": selected}
        per_seed[seed] = {}
        for name, predictions in rows.items():
            methods[name].append(predictions)
            per_seed[seed][name] = summarize(predictions)
            jsonl(args.output / f"{name}_s{seed}.jsonl", predictions)
        artifacts[seed] = file_hash(folder / "test.pt")
        if observation is None:
            observation = observation_audit(data, args.root)
        print(f"Final frozen seed {seed} evaluated on {len(data)} positive test queries", flush=True)
    avg = {name: averaged_metrics(rows) for name, rows in methods.items()}
    mean = {name: {metric: {"mean": float(np.mean([p[name][metric] for p in per_seed.values()])),
                            "seed_std": float(np.std([p[name][metric] for p in per_seed.values()]))}
                   for metric in protocol["evaluation"]["metrics"]} for name in methods}
    compared = {ref: {metric: bootstrap_delta(avg["trained_top3"], avg[ref], metric)
                     for metric in ("iou", "coverage_at_90", "event_coverage", "redundancy_seconds")}
                for ref in ("coarse_top1", "fine_top1", "frozen_quality_top3")}
    base, fine = compared["coarse_top1"], compared["fine_top1"]
    audit = {"iou_vs_coarse_ci_lower_positive": base["iou"]["source_video_cluster_95pct"][0] > 0,
             "coverage90_vs_coarse_ci_lower_nonnegative": base["coverage_at_90"]["source_video_cluster_95pct"][0] >= 0,
             "redundancy_vs_coarse_mean_nonpositive": base["redundancy_seconds"]["mean"] <= 0,
             "iou_vs_fine_top1_mean_positive": fine["iou"]["mean"] > 0,
             "coverage90_vs_fine_top1_mean_nonnegative": fine["coverage_at_90"]["mean"] >= 0}
    if audit.keys() != protocol["acceptance_audit"].keys():
        raise ValueError("Final audit does not cover every frozen acceptance requirement")
    report = {"scope": protocol["scope"], "protocol_sha256": file_hash(args.protocol), "queries": len(expected),
              "test_accessed": True, "models_or_thresholds_changed_by_this_script": False,
              "per_seed": per_seed, "means": mean, "paired_deltas": compared, "acceptance_audit": audit,
              "all_predeclared_checks_pass": all(audit.values()), "feature_artifact_sha256": artifacts,
              "observation_budget": observation,
              "limits": ["positive interval queries only", "not exhaustive semantic verification", "not Agent/export evaluation",
                         "strict boundary subset, not the full original 3720-query test release", "not a state-of-the-art benchmark claim"]}
    dump(args.output / "results.json", report)
    print(json.dumps({"means": mean, "acceptance_audit": audit, "all_pass": all(audit.values())}, indent=2), flush=True)


if __name__ == "__main__":
    main()
