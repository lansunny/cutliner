"""One-shot held-out audio-adaptation comparison against the unchanged teacher."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import AlignedMoments, collate
from final_eval import paired_interval
from qd_audio import AdaptedGrounder, evaluate_adapter, load_teacher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    folder = args.run_dir.resolve()
    output_path = folder / "heldout_results.json"
    if output_path.exists():
        raise SystemExit("Held-out evaluation already exists; preserving it")
    report = json.loads((folder / "development_report.json").read_text(encoding="utf-8"))
    config, runs = report["config"], report["runs"]
    if {(row["mode"], row["seed"]) for row in runs} != {(mode, seed) for mode in config["modes"] for seed in config["seeds"]}:
        raise ValueError("Adapter development runs not complete")
    ranks = {mode: float(np.mean([row["selected_dev"]["R1_iou0.5"] + 0.25*row["selected_dev"]["mean_iou"]
                                for row in runs if row["mode"] == mode])) for mode in config["modes"]}
    selected = max(ranks, key=ranks.get)
    selection = {"selected_mode_dev_only": selected, "dev_selection_scores": ranks, "config": config,
                 "note": "Selection frozen before any heldout model evaluation"}
    selection_path = folder / "frozen_selection.json"
    if selection_path.exists():
        if json.loads(selection_path.read_text(encoding="utf-8")) != selection:
            raise ValueError("Frozen adapter selection changed")
    else:
        with selection_path.open("x", encoding="utf-8") as stream:
            json.dump(selection, stream, indent=2)
    plan = json.loads((Path(config["dataset"]) / "split_plan.json").read_text(encoding="utf-8"))
    dataset = AlignedMoments(Path(plan["heldout_manifest"]))
    if len({row["vid"].rsplit("_", 2)[0] for row in dataset.rows}) != len(dataset):
        raise ValueError("Expected one heldout query per source")
    loader = DataLoader(dataset, batch_size=config["batch_size"], collate_fn=collate)
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher, _criterion = load_teacher(device)
    unchanged = AdaptedGrounder(teacher, config["modes"][0]).to(device)
    baseline_metrics, baseline_rows = evaluate_adapter(unchanged, loader, dataset, device, disable_audio=True)
    results, predictions = [], defaultdict(list)
    print(json.dumps({"unchanged_teacher": baseline_metrics}), flush=True)
    for mode in config["modes"]:
        for seed in config["seeds"]:
            checkpoint = torch.load(folder / f"{mode}_seed{seed}.pt", map_location="cpu", weights_only=True)
            if checkpoint["config"] != config:
                raise ValueError("Checkpoint belongs to another adapter study")
            model = AdaptedGrounder(teacher, mode).to(device)
            model.fusion.load_state_dict(checkpoint["fusion_state_dict"])
            metrics, rows = evaluate_adapter(model, loader, dataset, device)
            predictions[mode].append(rows)
            controls = {}
            if mode == selected:
                controls["audio_time_reverse"], _ = evaluate_adapter(model, loader, dataset, device, reverse_audio=True)
            corrected = sum(a["top1_iou"] < 0.5 <= b["top1_iou"] for a, b in zip(baseline_rows, rows))
            harmed = sum(b["top1_iou"] < 0.5 <= a["top1_iou"] for a, b in zip(baseline_rows, rows))
            results.append({"mode": mode, "seed": seed, "selected_epoch": checkpoint["epoch"], "metrics": metrics,
                            "corrected_vs_teacher": corrected, "harmed_vs_teacher": harmed, "controls": controls})
            print(json.dumps(results[-1]), flush=True)
    comparisons = {mode: paired_interval([baseline_rows] * len(config["seeds"]), predictions[mode]) for mode in config["modes"]}
    averages = {mode: {metric: float(np.mean([row["metrics"][metric] for row in results if row["mode"] == mode]))
                       for metric in ("R1_iou0.3", "R1_iou0.5", "R1_iou0.7", "R5_iou0.5", "mean_iou")} for mode in config["modes"]}
    result = {"selection": selection, "baseline": baseline_metrics, "runs": results, "averages": averages,
              "paired_comparisons": comparisons,
              "limitations": ["No ASR/dialogue recognition branch yet; this tests learned use of PANN audio features.",
                              "Public QD-DETR checkpoint may have been selected on official val; incremental comparison only.",
                              "No Chinese-language, no-target rejection, or entire-long-video end-to-end result."]}
    with (folder / "heldout_predictions.json").open("x", encoding="utf-8") as stream:
        json.dump({"baseline": baseline_rows, **dict(predictions)}, stream)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({"baseline": baseline_metrics, "averages": averages, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
