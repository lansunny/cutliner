"""Develop AV retrieval models on train/dev only. Never reads test manifest."""

import argparse
import hashlib
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import AlignedMoments, collate
from model import AVGrounder, objective
from public_zip import ROOT


def to_device(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def iou(span, truth):
    intersection = max(0.0, min(span[1], truth[1]) - max(span[0], truth[0]))
    return intersection / max(1e-9, span[1] - span[0] + truth[1] - truth[0] - intersection)


def proposals(output, batch, index):
    length, duration = int(batch["mask"][index].sum()), float(batch["duration"][index])
    scores = output["logits"][index, :length].sigmoid().tolist()
    distances = output["distances"][index, :length].tolist()
    centers = batch["centers"][index, :length].tolist()
    predictions = []
    for score, distance, center in sorted(zip(scores, distances, centers), reverse=True):
        start, end = max(0.0, center - distance[0] * duration), min(duration, center + distance[1] * duration)
        if end - start < 0.5:
            continue
        if any(iou((start, end), old["span"]) > 0.6 for old in predictions):
            continue
        predictions.append({"span": [start, end], "score": score})
        if len(predictions) == 5:
            break
    return predictions


@torch.inference_mode()
def evaluate(model, loader, dataset, device, audio_mode="aligned"):
    model.eval()
    results = []
    for raw in loader:
        batch = to_device(raw, device)
        override = None
        if audio_mode == "zero":
            override = torch.zeros_like(batch["audio"])
        elif audio_mode == "time_reverse":
            override = batch["audio"].clone()
            for i, mask in enumerate(batch["audio_mask"]):
                length = int(mask.sum())
                override[i, :length] = override[i, :length].flip(0)
        output = model(batch, audio_override=override)
        for i, row_index in enumerate(raw["indices"]):
            row = dataset.rows[row_index]
            predicted = proposals(output, batch, i)
            top1 = max((iou(predicted[0]["span"], truth) for truth in row["relevant_windows"]), default=0) if predicted else 0
            top5 = max((iou(item["span"], truth) for item in predicted for truth in row["relevant_windows"]), default=0)
            results.append({"qid": row["qid"], "source_group": row["vid"].rsplit("_", 2)[0],
                            "top1_iou": top1, "top5_iou": top5, "predictions": predicted})
    count = max(1, len(results))
    metrics = {"n_queries": len(results), "mean_iou": sum(row["top1_iou"] for row in results) / count,
               **{f"R1_iou{threshold}": sum(row["top1_iou"] >= threshold for row in results) / count for threshold in (0.3, 0.5, 0.7)},
               "R5_iou0.5": sum(row["top5_iou"] >= 0.5 for row in results) / count}
    return metrics, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/av_v1")
    parser.add_argument("--modes", nargs="+", choices=["visual", "visual_wide", "audio", "concat", "gated"],
                        default=["visual", "visual_wide", "concat", "gated"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0005)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.hidden) <= 0 or args.hidden % 4:
        parser.error("Invalid epochs/batch/hidden")
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.perf_counter()
    train, dev = AlignedMoments(args.dataset / "train.jsonl"), AlignedMoments(args.dataset / "dev.jsonl")
    train_sources = {row["vid"].rsplit("_", 2)[0] for row in train.rows}
    if train_sources & {row["vid"].rsplit("_", 2)[0] for row in dev.rows}:
        raise ValueError("Development source-video leakage")
    dev_loader = DataLoader(dev, batch_size=args.batch_size, collate_fn=collate)
    folder = ROOT / "training_runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True)
    config = {"dataset": str(args.dataset.resolve()), "modes": args.modes, "seeds": args.seeds,
              "epochs": args.epochs, "batch_size": args.batch_size, "hidden": args.hidden, "lr": args.lr,
              "device": device, "split_plan_sha256": hashlib.sha256((args.dataset / "split_plan.json").read_bytes()).hexdigest(),
              "test_accessed": False, "encoders_frozen": True,
              "scope": "English query-conditioned interval retrieval from visual+PANN features; no ASR yet"}
    (folder / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    summary = []
    for mode in args.modes:
      for seed in args.seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = AVGrounder(mode, args.hidden).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                            generator=torch.Generator().manual_seed(seed))
        best, history, checkpoint_path = -1.0, [], folder / f"{mode}_seed{seed}.pt"
        for epoch in range(args.epochs):
            model.train()
            total, seen = 0.0, 0
            for raw in loader:
                batch = to_device(raw, device)
                optimizer.zero_grad(set_to_none=True)
                output = model(batch)
                loss, _ = objective(output, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += float(loss.detach()) * len(raw["indices"])
                seen += len(raw["indices"])
            metrics, _ = evaluate(model, dev_loader, dev, device)
            selection = metrics["R1_iou0.5"] + 0.25 * metrics["mean_iou"]
            history.append({"epoch": epoch + 1, "train_loss": total / seen, "dev": metrics})
            print(f"{mode} seed{seed} epoch{epoch+1}: loss={total/seen:.4f}, devR1@.5={metrics['R1_iou0.5']:.4f}, mIoU={metrics['mean_iou']:.4f}", flush=True)
            if selection > best:
                best = selection
                torch.save({"state_dict": model.state_dict(), "mode": mode, "hidden": model.hidden,
                            "seed": seed, "epoch": epoch + 1, "development": metrics,
                            "split_plan_sha256": config["split_plan_sha256"], "config": config}, checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        controls = {}
        if mode in ("concat", "gated"):
            for control in ("zero", "time_reverse"):
                controls[control], _ = evaluate(model, dev_loader, dev, device, control)
        entry = {"mode": mode, "seed": seed, "checkpoint": str(checkpoint_path), "epoch": checkpoint["epoch"],
                 "parameters": sum(parameter.numel() for parameter in model.parameters()),
                 "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
                 "selected_dev": checkpoint["development"], "audio_controls_dev": controls, "history": history}
        summary.append(entry)
        (folder / "development_report.json").write_text(json.dumps({"config": config, "runs": summary,
                                                                    "elapsed_seconds": time.perf_counter()-started}, indent=2), encoding="utf-8")
    print(f"Development runs saved: {folder}; test remains untouched", flush=True)


if __name__ == "__main__":
    main()
