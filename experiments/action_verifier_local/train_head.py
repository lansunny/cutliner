"""Train the small conditional head on real labeled clips, never on dummy data."""

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from actions import ACTIONS, OPPOSITES
from encoder import FrozenVideoEncoder
from prepare_model import ROOT, sha256_file
from rejection_policy import rejection_metrics, select_rejection_policy
from verifier import ACTION_ORDER, ActionMatchingHead


def load_manifest(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("No real labeled examples; training cannot start")
    memberships, sample_ids = {}, set()
    for row in rows:
        for key in ("sample_id", "group_id", "video_path", "action", "label", "split", "annotation_source"):
            if key not in row:
                raise ValueError(f"Missing {key} in manifest row")
        if row["sample_id"] in sample_ids:
            raise ValueError("Duplicate sample ID")
        sample_ids.add(row["sample_id"])
        if row["action"] not in ACTIONS or type(row["label"]) is not int or row["label"] not in (0, 1):
            raise ValueError("Unknown action or non-binary label")
        if row["split"] not in ("train", "val", "test"):
            raise ValueError("Expected explicit train/val/test split")
        source = Path(row["video_path"])
        if not source.is_absolute():
            source = (path.parent / source).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        row["video_path"] = str(source)
        # Check both declared groups and literal source files against leakage.
        digest = sha256_file(source)
        if row.get("video_sha256") and row["video_sha256"] != digest:
            raise ValueError("Source video differs from frozen manifest")
        for identity in (row["group_id"], str(source.resolve()).casefold(), digest):
            memberships.setdefault(identity, set()).add(row["split"])
    if any(len(splits) > 1 for splits in memberships.values()):
        raise ValueError("Source/group leakage across data splits")
    for split in ("train", "val"):
        selected = [row for row in rows if row["split"] == split]
        if {row["label"] for row in selected} != {0, 1}:
            raise ValueError(f"{split} requires actual positive and negative examples")
    return rows


def metrics(scores, labels, threshold):
    predictions = scores >= threshold
    truth = labels.bool()
    tp = int((predictions & truth).sum())
    fp = int((predictions & ~truth).sum())
    fn = int((~predictions & truth).sum())
    tn = int((~predictions & ~truth).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {"n": len(labels), "accuracy": (tp + tn) / max(1, len(labels)),
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(1e-12, precision + recall),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def choose_threshold(scores, labels):
    # Operating-point selection on validation data, not probability calibration.
    candidates = sorted({0.5, *torch.quantile(scores.float(), torch.linspace(0, 1, 101)).tolist()})
    threshold = max(candidates, key=lambda value: (metrics(scores, labels, value)["f1"], -abs(value - 0.5)))
    return float(threshold), metrics(scores, labels, threshold)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--hwaccel", choices=["auto", "cuda", "none"], default="auto")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pilot", action="store_true", help="Mark this run as execution validation, not an effectiveness experiment")
    args = parser.parse_args()
    started = time.perf_counter()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("Epochs, batch size and learning rate must be positive")
    rows = load_manifest(args.manifest)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    encoder = FrozenVideoEncoder(args.device, args.hwaccel)
    fingerprint = encoder.fingerprint
    features, original_scores, opposite_scores, opposite_top1 = [], [], [], []
    for index, row in enumerate(rows, 1):
        feature, logits, _, _ = encoder.encode(row["video_path"], float(row.get("start", 0)), row.get("end"))
        features.append(feature)
        probabilities = torch.from_numpy(logits).softmax(-1)
        original_scores.append(float(probabilities[encoder.action_labels[row["action"]]]))
        opposite_id = encoder.action_labels[OPPOSITES[row["action"]]]
        opposite_scores.append(float(probabilities[opposite_id]))
        opposite_top1.append(int(probabilities.argmax()) == opposite_id)
        if index % 20 == 0 or index == len(rows):
            print(f"Prepared features for {index}/{len(rows)} pairs (shared videos are cached)", flush=True)
    feature_dim = features[0].shape[-1]
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Head training on cached features is intentionally CPU-light; no backbone gradients.
    x = torch.from_numpy(np.stack(features))
    a = torch.tensor([ACTION_ORDER.index(row["action"]) for row in rows])
    y = torch.tensor([row["label"] for row in rows], dtype=torch.float32)
    train_idx = torch.tensor([i for i, row in enumerate(rows) if row["split"] == "train"])
    val_idx = torch.tensor([i for i, row in enumerate(rows) if row["split"] == "val"])
    dataset = TensorDataset(x[train_idx], a[train_idx], y[train_idx])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    model = ActionMatchingHead(feature_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    baseline = torch.tensor(original_scores)
    opposite_scores = torch.tensor(opposite_scores)
    opposite_top1 = torch.tensor(opposite_top1)
    baseline_threshold, baseline_metrics = choose_threshold(baseline[val_idx], y[val_idx])
    run_dir = ROOT / "training_runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True)
    best_f1 = -1.0
    history = []
    print(f"Frozen encoder; trainable head parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for bx, ba, by in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(bx, ba), by)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(by)
        model.eval()
        with torch.inference_mode():
            val_scores = model(x[val_idx], a[val_idx]).sigmoid()
        threshold, val_metrics = choose_threshold(val_scores, y[val_idx])
        history.append({"epoch": epoch + 1, "train_loss": total_loss / len(dataset),
                        "threshold": threshold, "validation": val_metrics})
        print(f"Epoch {epoch + 1}: loss={history[-1]['train_loss']:.4f}, val_F1={val_metrics['f1']:.4f}", flush=True)
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save({"schema_version": 1, "state_dict": model.state_dict(),
                        "feature_dim": feature_dim, "encoder_fingerprint": fingerprint,
                        "actions": ACTION_ORDER, "threshold": threshold,
                        "validation_metrics": val_metrics,
                        "manifest_sha256": sha256_file(args.manifest),
                        "objective": "primary_action_match_for_selected_classes",
                        "purpose": "execution_pilot" if args.pilot else "experimental_training",
                        "epoch": epoch + 1}, run_dir / "best_head.pt")
    # Test labels, when supplied, are evaluated once after selection, never used above.
    checkpoint = torch.load(run_dir / "best_head.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    with torch.inference_mode():
        all_scores = model(x, a).sigmoid()
    policy = select_rejection_policy(all_scores[val_idx], y[val_idx], opposite_scores[val_idx], opposite_top1[val_idx])
    checkpoint["rejection_policy"] = policy
    checkpoint["deployment_status"] = "experimental_closed_domain_opt_in_only"
    torch.save(checkpoint, run_dir / "best_head.pt")
    validation_rejection = rejection_metrics(all_scores[val_idx], y[val_idx], opposite_scores[val_idx], opposite_top1[val_idx], policy)
    test_idx = [i for i, row in enumerate(rows) if row["split"] == "test"]
    test_metrics = None
    baseline_test = None
    test_rejection, per_action_test, test_predictions = None, {}, []
    if test_idx:
        with torch.inference_mode():
            scores = model(x[test_idx], a[test_idx]).sigmoid()
        test_metrics = metrics(scores, y[test_idx], checkpoint["threshold"])
        baseline_test = metrics(baseline[test_idx], y[test_idx], baseline_threshold)
        test_rejection = rejection_metrics(all_scores[test_idx], y[test_idx], opposite_scores[test_idx], opposite_top1[test_idx], policy)
        for action in ACTIONS:
            selected = [i for i in test_idx if rows[i]["action"] == action]
            per_action_test[action] = {"head": metrics(all_scores[selected], y[selected], checkpoint["threshold"]),
                                       "baseline": metrics(baseline[selected], y[selected], baseline_threshold)}
        for i in test_idx:
            test_predictions.append({"sample_id": rows[i]["sample_id"], "group_id": rows[i]["group_id"],
                                     "action": rows[i]["action"], "label": rows[i]["label"],
                                     "head_score": float(all_scores[i]), "baseline_score": float(baseline[i]),
                                     "opposite_score": float(opposite_scores[i]), "opposite_top1": bool(opposite_top1[i])})
    report = {"manifest": str(args.manifest.resolve()), "encoder_fingerprint": fingerprint,
              "purpose": "execution_pilot" if args.pilot else "experimental_training",
              "trainable_parameters": sum(p.numel() for p in model.parameters()),
              "source_videos": len({row["group_id"] for row in rows}),
              "pairs": len(rows), "history": history,
              "config": {"epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "seed": args.seed},
              "manifest_sha256": sha256_file(args.manifest),
              "selected_epoch": checkpoint["epoch"], "head_threshold": checkpoint["threshold"],
              "selected_validation": checkpoint["validation_metrics"],
              "video_counts": {split: len({r["group_id"] for r in rows if r["split"] == split}) for split in ("train", "val", "test")},
              "baseline_validation": baseline_metrics, "baseline_threshold": baseline_threshold,
              "head_test": test_metrics, "baseline_test": baseline_test,
              "rejection_policy": policy, "validation_rejection": validation_rejection,
              "test_rejection": test_rejection, "per_action_test": per_action_test,
              "elapsed_seconds": time.perf_counter() - started,
              "limitations": ["Validation scores select weights and thresholds; they are not held-out test results.",
                              "Template-derived negatives supervise primary action, not exhaustive absence.",
                              "Test is held out from our head; pretrained backbone was fine-tuned on SSV2, not an external-domain test.",
                              "Small bounded-prefix convenience sample; pairs from the same video are correlated.",
                              "No arbitrary language, state explanation, identity tracking or OOD guarantee."]}
    (run_dir / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "test_predictions.json").write_text(json.dumps(test_predictions, indent=2), encoding="utf-8")
    print(f"Saved: {run_dir / 'best_head.pt'}")


if __name__ == "__main__":
    main()
