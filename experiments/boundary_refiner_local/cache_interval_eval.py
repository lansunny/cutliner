"""Cache terminal models on dev proposals for policy calibration, never test."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import BoundaryDataset, collate, to_device
from interval_model import IntervalQualityRefiner
from model import BoundaryRefiner, joint_decode
from prepare_data import DEFAULT_ROOT, dump, file_hash, jsonl
from retrieval_dev import RetrievedDataset
from train import predictions_from_batch, summarize
from train_interval import evaluate_interval


@torch.inference_mode()
def cache_v1(model, loader):
    cached, predictions = [], []
    model.eval()
    for batch in loader:
        output = model(batch)
        predictions.extend(predictions_from_batch(joint_decode(output, batch), batch))
        for i in range(len(batch["case_id"])):
            n = int(batch["mask"][i].sum())
            item = {key: batch[key][i] for key in ("case_id", "video_id", "kind")}
            for key in ("target", "coarse", "window"):
                item[key] = batch[key][i].clone()
            item["edges"] = batch["edges"][i, :n + 1].clone()
            for key in ("start", "end"):
                item[key] = output[key][i, :n + 1].clone()
            cached.append(item)
    return cached, summarize(predictions), predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--v1", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(3)
    state = json.loads((args.run / "status.json").read_text(encoding="utf-8"))
    if state["status"] != "complete":
        raise ValueError("Only terminal checkpoints may be audited")
    if args.output.exists():
        raise ValueError("New output snapshot required")
    data = RetrievedDataset(args.root, args.candidates) if args.candidates else BoundaryDataset(args.root, "dev")
    loader = DataLoader(data, batch_size=64, collate_fn=collate, num_workers=0)
    saved = torch.load(args.run / "best.pt", map_location="cpu", weights_only=True)
    if args.v1:
        config = saved["configuration"]
        model = BoundaryRefiner(config.get("backbone_variant", config["variant"]), config.get("hidden", 128))
        model.load_state_dict(saved["model"], strict=True)
        cached, summary, predictions = cache_v1(model, loader)
        report = {"default": summary}
    else:
        model = IntervalQualityRefiner(backbone_variant=saved["configuration"].get("backbone_variant", "film"))
        model.load_state_dict(saved["model"], strict=True)
        summary, predictions, marginal, cached = evaluate_interval(model, loader, "cpu", cache=True)
        report = {"quality_unconstrained": summary, "marginal_head": marginal}
    args.output.mkdir(parents=True)
    torch.save(cached, args.output / "outputs.pt")
    jsonl(args.output / "default_predictions.jsonl", predictions)
    report.update(source_run=str(args.run), checkpoint_sha256=file_hash(args.run / "best.pt"),
                  protocol="actual_top1_dev" if args.candidates else "synthetic_dev",
                  test_accessed=False, calibration_only=True)
    dump(args.output / "results.json", report)
    compact = {key: {k: v[k] for k in ("iou", "coverage_at_90", "event_coverage", "redundancy_seconds")}
               for key, v in report.items() if isinstance(v, dict) and "iou" in v}
    print(json.dumps({"run": str(args.run), "results": compact}, indent=2), flush=True)


if __name__ == "__main__":
    main()
