"""Train/evaluate local boundary architectures on train/dev ONLY by default.

All reported development numbers use synthetic coarse proposals. The official
test split is deliberately inaccessible from this architecture-search entrypoint.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import BoundaryDataset, PROTOCOL, collate, to_device
from extract_features import PROFILE
from model import BoundaryRefiner, joint_decode, objective
from prepare_data import DEFAULT_ROOT, dump, file_hash, jsonl


def interval_metrics(prediction, target):
    start, end = map(float, prediction)
    a, b = map(float, target)
    intersection = max(0., min(end, b) - max(start, a))
    union = max(end, b) - min(start, a)
    return {"iou": intersection / max(union, 1e-6),
            "start_error_seconds": abs(start - a), "end_error_seconds": abs(end - b),
            "boundary_mae_seconds": (abs(start - a) + abs(end - b)) / 2,
            "event_coverage": intersection / (b - a),
            "redundancy_seconds": (end - start) - intersection,
            "missed_event_seconds": (b - a) - intersection,
            "duration_seconds": end - start,
            "iou_at_50": float(intersection / max(union, 1e-6) >= .5),
            "iou_at_70": float(intersection / max(union, 1e-6) >= .7),
            "coverage_at_90": float(intersection / (b - a) >= .9),
            "both_boundaries_within_1s": float(abs(start - a) <= 1 and abs(end - b) <= 1)}


def summarize(rows):
    metric_names = list(rows[0]["metrics"])
    result = {name: float(np.mean([x["metrics"][name] for x in rows])) for name in metric_names}
    result["samples"] = len(rows)
    result["independent_queries"] = len({x["case_id"] for x in rows})
    result["source_videos"] = len({x["video_id"] for x in rows})
    result["by_perturbation"] = {}
    for kind in sorted({x["kind"] for x in rows}):
        selected = [x for x in rows if x["kind"] == kind]
        result["by_perturbation"][kind] = {name: float(np.mean([x["metrics"][name] for x in selected])) for name in metric_names}
    return result


def predictions_from_batch(predictions, batch):
    predictions = predictions.detach().cpu().numpy() if torch.is_tensor(predictions) else predictions
    rows = []
    for i, prediction in enumerate(predictions):
        target, coarse = batch["target"][i].cpu().tolist(), batch["coarse"][i].cpu().tolist()
        rows.append({"case_id": batch["case_id"][i], "video_id": batch["video_id"][i], "kind": batch["kind"][i],
                     "prediction": list(map(float, prediction)), "target": target, "coarse": coarse,
                     "metrics": interval_metrics(prediction, target)})
    return rows


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    rows = []
    for cpu in loader:
        batch = to_device(cpu, device)
        output = model(batch)
        result = joint_decode(output, batch)
        rows.extend(predictions_from_batch(result, cpu))
    return summarize(rows), rows


def clip_prediction(sample, threshold_fraction, smooth_radius):
    v = sample["visual"]
    q = sample["query"]
    similarity = (v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)) @ (q / max(np.linalg.norm(q), 1e-8))
    if smooth_radius:
        similarity = np.convolve(np.pad(similarity, smooth_radius, mode="edge"),
                                 np.ones(2 * smooth_radius + 1) / (2 * smooth_radius + 1), mode="valid")
    threshold = similarity.min() + threshold_fraction * (similarity.max() - similarity.min())
    active = similarity >= threshold
    changes = np.diff(np.pad(active.astype(np.int32), (1, 1)))
    intervals = zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))
    standardized = (similarity - similarity.mean()) / max(similarity.std(), 1e-6)
    best, best_score = None, -math.inf
    for start, end in intervals:
        candidate = [sample["edges"][start], sample["edges"][end]]
        anchor_iou = interval_metrics(candidate, sample["coarse"])["iou"]
        score = standardized[start:end].mean() + anchor_iou
        if score > best_score:
            best, best_score = candidate, score
    return best


def baselines(dev, directory):
    directory.mkdir(parents=True, exist_ok=False)
    coarse_rows = []
    samples = [dev[i] for i in range(len(dev))]
    for sample in samples:
        coarse_rows.append({"case_id": sample["case_id"], "video_id": sample["video_id"], "kind": sample["kind"],
                            "prediction": sample["coarse"].tolist(), "target": sample["target"].tolist(),
                            "coarse": sample["coarse"].tolist(),
                            "metrics": interval_metrics(sample["coarse"], sample["target"])})
    jsonl(directory / "coarse_predictions.jsonl", coarse_rows)
    best, trials = None, []
    for threshold in (.0, .25, .5, .75):
        for radius in (0, 1):
            rows = []
            for sample in samples:
                prediction = clip_prediction(sample, threshold, radius)
                rows.append({"case_id": sample["case_id"], "video_id": sample["video_id"], "kind": sample["kind"],
                             "prediction": list(map(float, prediction)), "target": sample["target"].tolist(),
                             "coarse": sample["coarse"].tolist(), "metrics": interval_metrics(prediction, sample["target"])})
            summary = summarize(rows)
            trial = {"threshold_fraction": threshold, "smooth_radius": radius, "dev": summary}
            trials.append(trial)
            if best is None or summary["iou"] > best[0]["dev"]["iou"]:
                best = trial, rows
    jsonl(directory / "clip_predictions.jsonl", best[1])
    report = {"protocol": PROTOCOL, "scope": "dev synthetic-positive refinement; not end-to-end",
              "test_accessed": False, "coarse": summarize(coarse_rows), "clip": best[0],
              "clip_tuning_trials": trials}
    dump(directory / "results.json", report)
    print(json.dumps({"coarse": report["coarse"], "clip": report["clip"]}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--variant", choices=["film", "cross", "cross_delta", "no_query"], default="film")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baselines-only", action="store_true")
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-dev", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new experiment directory; never overwrite an earlier run")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(3)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    print("Loading independent development sources and their frozen feature caches...", flush=True)
    dev = BoundaryDataset(args.root, "dev", seed=args.seed, limit_cases=args.limit_dev)
    if args.baselines_only:
        baselines(dev, args.output)
        return
    print("Loading training sources...", flush=True)
    train = BoundaryDataset(args.root, "train", seed=args.seed, limit_cases=args.limit_train)
    args.output.mkdir(parents=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(protocol=PROTOCOL, feature_profile=PROFILE, scope="positive synthetic-coarse refinement",
                  test_accessed=False, encoder_trained=False, null_head_trained=False,
                  train_queries=len(train.rows), dev_queries=len(dev.rows),
                  train_manifest_sha256=file_hash(args.root / "manifests/train.complete.jsonl"),
                  dev_manifest_sha256=file_hash(args.root / "manifests/dev.complete.jsonl"),
                  model_source_sha256=file_hash(Path(__file__).with_name("model.py")),
                  dataset_source_sha256=file_hash(Path(__file__).with_name("dataset.py")),
                  train_source_sha256=file_hash(Path(__file__)),
                  torch_version=str(torch.__version__))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BoundaryRefiner(args.variant, hidden=args.hidden).to(device)
    config["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dump(args.output / "configuration.json", config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    shuffle_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                              num_workers=0, generator=shuffle_generator)
    dev_loader = DataLoader(dev, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    best_iou, best_epoch, history = -1, 0, []
    started = time.monotonic()
    print(f"Training {args.variant}: {config['trainable_parameters']} parameters; train={len(train)}, dev={len(dev)}; {device}", flush=True)
    for epoch in range(1, args.epochs + 1):
        train.epoch = epoch - 1
        model.train()
        learning_rate = args.lr * (.1 + .9 * (1 + math.cos(math.pi * (epoch - 1) / max(args.epochs - 1, 1))) / 2)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        losses = []
        for cpu in train_loader:
            batch = to_device(cpu, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss, _ = objective(output, batch)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics, predictions = evaluate(model, dev_loader, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "learning_rate": learning_rate,
               "dev": metrics, "elapsed_seconds": round(time.monotonic() - started, 1)}
        history.append(row)
        if metrics["iou"] > best_iou:
            best_iou, best_epoch = metrics["iou"], epoch
            torch.save({"model": model.state_dict(), "configuration": config, "epoch": epoch, "dev": metrics}, args.output / "best.pt")
            jsonl(args.output / "best_dev_predictions.jsonl", predictions)
            dump(args.output / "best_dev.json", row)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "configuration": config,
                    "epoch": epoch, "best_epoch": best_epoch}, args.output / "last.pt")
        dump(args.output / "history.json", history)
        dump(args.output / "status.json", {"status": "running", "epoch": epoch, "best_epoch": best_epoch,
                                           "best_dev_iou": best_iou, "pid": os.getpid(), "elapsed_seconds": row["elapsed_seconds"]})
        print(json.dumps({"epoch": epoch, "train_loss": row["train_loss"], "dev_iou": round(metrics["iou"], 5),
                          "boundary_mae": round(metrics["boundary_mae_seconds"], 4),
                          "coverage": round(metrics["event_coverage"], 5), "best_epoch": best_epoch,
                          "elapsed_seconds": row["elapsed_seconds"]}), flush=True)
        if epoch - best_epoch >= args.patience:
            break
    dump(args.output / "status.json", {"status": "complete", "epochs": len(history), "best_epoch": best_epoch,
                                       "best_dev_iou": best_iou, "test_accessed": False,
                                       "elapsed_seconds": round(time.monotonic() - started, 1)})
    print(json.dumps({"run": str(args.output), "best_epoch": best_epoch, "best_dev_iou": best_iou,
                      "final_test_used": False}), flush=True)


if __name__ == "__main__":
    main()
