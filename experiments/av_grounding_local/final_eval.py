"""Freeze selection on dev, then evaluate new AV models once on held-out sources."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import AlignedMoments, collate
from model import AVGrounder
from train import evaluate


def paired_interval(baseline, candidate):
    # Same held-out source queries for every seed. Bootstrap queries, not frames.
    a = np.array([[row["top1_iou"] >= 0.5 for row in run] for run in baseline], dtype=np.float32)
    b = np.array([[row["top1_iou"] >= 0.5 for row in run] for run in candidate], dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError("Paired experiment shapes differ")
    differences = (b - a).mean(0)
    generator = np.random.default_rng(20261107)
    samples = differences[generator.integers(0, len(differences), size=(2000, len(differences)))].mean(1)
    return {"mean_R1_iou0.5_difference": float(differences.mean()),
            "query_bootstrap_95_interval": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
            "scope": "Query resampling conditional on three trained seeds; not deployment-domain uncertainty"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    folder = args.run_dir.resolve()
    if (folder / "heldout_results.json").exists():
        raise SystemExit("Held-out results already exist; do not retune or overwrite this evaluation")
    development = json.loads((folder / "development_report.json").read_text(encoding="utf-8"))
    config, runs = development["config"], development["runs"]
    expected = {(mode, seed) for mode in config["modes"] for seed in config["seeds"]}
    if {(run["mode"], run["seed"]) for run in runs} != expected:
        raise ValueError("Development runs are incomplete")
    by_mode = defaultdict(list)
    for run in runs:
        by_mode[run["mode"]].append(run)
    ranking = {mode: sum(run["selected_dev"]["R1_iou0.5"] for run in values) / len(values) for mode, values in by_mode.items()}
    multimodal = [mode for mode in ranking if mode in ("concat", "gated")]
    if "visual" not in by_mode or not multimodal:
        raise ValueError("Need visual and multimodal controls")
    selected = max(multimodal, key=lambda mode: ranking[mode])
    selection = {"selection_stage": "dev_only_before_loading_test", "selected_multimodal_mode": selected,
                 "mean_dev_R1_iou0.5": ranking, "config": config}
    selection_path = folder / "frozen_selection.json"
    if selection_path.exists():
        if json.loads(selection_path.read_text(encoding="utf-8")) != selection:
            raise ValueError("Frozen selection changed")
    else:
        with selection_path.open("x", encoding="utf-8") as stream:
            json.dump(selection, stream, indent=2)
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = AlignedMoments(Path(config["dataset"]) / "test.jsonl")
    # This subset was selected with one query per original video source.
    sources = [row["vid"].rsplit("_", 2)[0] for row in dataset.rows]
    if len(set(sources)) != len(sources):
        raise ValueError("Bootstrap assumes one query per source in this frozen subset")
    loader = DataLoader(dataset, batch_size=config["batch_size"], collate_fn=collate)
    results, predictions = [], defaultdict(list)
    for mode in config["modes"]:
        for seed in config["seeds"]:
            checkpoint = torch.load(folder / f"{mode}_seed{seed}.pt", map_location="cpu", weights_only=True)
            if checkpoint["split_plan_sha256"] != config["split_plan_sha256"]:
                raise ValueError("Checkpoint/data split mismatch")
            model = AVGrounder(mode, checkpoint["hidden"]).to(device)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            metrics, rows = evaluate(model, loader, dataset, device)
            predictions[mode].append(rows)
            controls = {}
            if mode == selected:
                for control in ("zero", "time_reverse"):
                    controls[control], _ = evaluate(model, loader, dataset, device, control)
            results.append({"mode": mode, "seed": seed, "metrics": metrics, "audio_controls": controls})
            print(json.dumps({"mode": mode, "seed": seed, "metrics": metrics}), flush=True)
    averages = {mode: {key: float(np.mean([row["metrics"][key] for row in results if row["mode"] == mode]))
                       for key in ("mean_iou", "R1_iou0.3", "R1_iou0.5", "R1_iou0.7", "R5_iou0.5")} for mode in by_mode}
    comparisons = {mode: {"vs_visual": paired_interval(predictions["visual"], predictions[mode]),
                          **({"vs_visual_wide": paired_interval(predictions["visual_wide"], predictions[mode])}
                             if "visual_wide" in predictions else {})} for mode in multimodal}
    report = {"selection": selection, "heldout_source_queries": len(dataset), "results": results,
              "averages": averages, "paired_comparisons": comparisons,
              "limitations": ["QVHighlights audiovisual temporal retrieval, not dialogue transcription or Chinese-query validation.",
                              "Audio zero/reverse controls are feature perturbations, not human-verified audio-dependent labels.",
                              "No claim of beating pretrained QD-DETR; this comparison isolates audio in newly trained small models."]}
    with (folder / "heldout_predictions.json").open("x", encoding="utf-8") as stream:
        json.dump(dict(predictions), stream)
    with (folder / "heldout_results.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({"averages": averages, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
