"""Second architecture round: structured pair quality with preserved v1 controls."""

from __future__ import annotations

import argparse
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
from interval_dataset import IntervalTrainingDataset, RepresentativeTrainingDataset
from interval_model import IntervalQualityRefiner, interval_objective, masked_boundary_objective, quality_decode
from model import BoundaryRefiner, joint_decode
from prepare_data import DEFAULT_ROOT, dump, file_hash, jsonl
from train import predictions_from_batch, summarize


@torch.inference_mode()
def evaluate_interval(model, loader, device, cache=False, marginal_only=False):
    model.eval()
    rows, marginal, cached = [], [], []
    for cpu in loader:
        batch = to_device(cpu, device)
        output = model(batch)
        if marginal_only:
            prediction = joint_decode(output, batch)
            diagnostics = {"outside_probability": torch.zeros(len(cpu["case_id"]), device=device)}
        else:
            prediction, diagnostics = quality_decode(output, batch, fallback=False)
        decoded = predictions_from_batch(prediction, cpu)
        for i, row in enumerate(decoded):
            row["outside_probability"] = float(diagnostics["outside_probability"][i])
        rows.extend(decoded)
        marginal.extend(predictions_from_batch(joint_decode(output, batch), cpu))
        if cache:
            for i in range(len(decoded)):
                n = int(batch["mask"][i].sum())
                item = {"case_id": cpu["case_id"][i], "video_id": cpu["video_id"][i], "kind": cpu["kind"][i]}
                for key in ("target", "coarse", "window"):
                    item[key] = cpu[key][i].clone()
                item["edges"] = cpu["edges"][i, :n + 1].clone()
                for key in ("start", "end"):
                    item[key] = output[key][i, :n + 1].detach().cpu().clone()
                if not marginal_only:
                    for key in ("iou_quality", "coverage_quality", "pair_mask"):
                        item[key] = output[key][i, :n + 1, :n + 1].detach().cpu().clone()
                    item["outside"] = output["outside"][i].detach().cpu().clone()
                cached.append(item)
    return summarize(rows), rows, summarize(marginal), cached


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partial-context", action="store_true")
    parser.add_argument("--marginal-only", action="store_true", help="Ablate all interval-quality/completeness heads")
    parser.add_argument("--backbone-variant", choices=["film", "no_query"], default="film")
    parser.add_argument("--actual-candidates", type=Path,
                        help="Independent train-only top1 proposals; mixes real, nearly-correct and perturbed anchors")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--effective-batch", type=int, default=96)
    parser.add_argument("--microbatch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    if args.actual_candidates and args.partial_context:
        parser.error("Keep the representative-proposal and partial-context interventions separate")
    if args.output.exists():
        raise ValueError("New output directory required; preserve previous experiments")
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
    print("Loading unchanged train/dev feature sources...", flush=True)
    dev = BoundaryDataset(args.root, "dev", seed=args.seed)
    train = (RepresentativeTrainingDataset(args.root, "train", seed=args.seed, candidates=args.actual_candidates)
             if args.actual_candidates else
             IntervalTrainingDataset(args.root, "train", seed=args.seed, partial_context=args.partial_context))
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(protocol=PROTOCOL, feature_profile=PROFILE, test_accessed=False, encoder_trained=False,
                  train_queries=len(train.rows), dev_queries=len(dev.rows),
                  variant="pair_representative" if args.actual_candidates else ("pair_partial" if args.partial_context else "pair_full"),
                  target_semantics="annotated target interval; not exhaustive scene absence",
                  source_hashes={name: file_hash(Path(__file__).with_name(name)) for name in
                                 ("model.py", "dataset.py", "interval_model.py", "interval_dataset.py", "train_interval.py")},
                  train_manifest_sha256=file_hash(args.root / "manifests/train.complete.jsonl"),
                  dev_manifest_sha256=file_hash(args.root / "manifests/dev.complete.jsonl"))
    if args.actual_candidates:
        config["actual_training_candidates_sha256"] = file_hash(args.actual_candidates / "candidates.jsonl")
        config["proposal_mixture"] = {"original_perturbations": 1 / 3, "nearly_correct": 1 / 3, "real_top1": 1 / 3}
    config["model_kind"] = "marginal" if args.marginal_only else "interval_quality"
    config["hidden"], config["rank"] = 128, 16
    if args.marginal_only:
        config["variant"] += "_marginal_only"
    if args.backbone_variant == "no_query":
        config["variant"] += "_no_query"
    args.output.mkdir(parents=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = (BoundaryRefiner(args.backbone_variant) if args.marginal_only else
             IntervalQualityRefiner(backbone_variant=args.backbone_variant)).to(device)
    config["trainable_parameters"] = sum(p.numel() for p in model.parameters())
    dump(args.output / "configuration.json", config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train, batch_size=args.effective_batch, shuffle=True, generator=generator,
                        collate_fn=list, num_workers=0)
    dev_loader = DataLoader(dev, batch_size=64, collate_fn=collate, num_workers=0)
    best_iou, best_epoch, history = -1., 0, []
    started = time.monotonic()
    print(f"Training {config['variant']}: {config['trainable_parameters']} params; effective batch {args.effective_batch}, microbatch {args.microbatch}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train.epoch = epoch - 1
        lr = args.lr * (.1 + .9 * (1 + math.cos(math.pi * (epoch - 1) / max(args.epochs - 1, 1))) / 2)
        for group in optimizer.param_groups:
            group["lr"] = lr
        losses, partials, seen = [], 0, 0
        for raw in loader:
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0.
            for offset in range(0, len(raw), args.microbatch):
                samples = raw[offset:offset + args.microbatch]
                batch = to_device(collate(samples), device)
                output = model(batch)
                loss, _ = (masked_boundary_objective(output, batch) if args.marginal_only else interval_objective(output, batch))
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite interval objective")
                (loss * (len(samples) / len(raw))).backward()
                accumulated += float(loss.detach()) * len(samples) / len(raw)
                missing = (batch["target"][:, 0] < batch["window"][:, 0]) | (batch["target"][:, 1] > batch["window"][:, 1])
                partials += int(missing.sum())
                seen += len(samples)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(accumulated)
        metrics, predictions, marginal, _ = evaluate_interval(model, dev_loader, device, marginal_only=args.marginal_only)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "quality_dev": metrics,
               "marginal_dev": marginal, "partial_training_fraction": partials / seen,
               "elapsed_seconds": round(time.monotonic() - started, 1)}
        history.append(row)
        if metrics["iou"] > best_iou:
            best_iou, best_epoch = metrics["iou"], epoch
            torch.save({"model": model.state_dict(), "configuration": config, "epoch": epoch}, args.output / "best.pt")
            jsonl(args.output / "best_dev_predictions.jsonl", predictions)
            dump(args.output / "best_dev.json", row)
        dump(args.output / "history.json", history)
        dump(args.output / "status.json", {"status": "running", "epoch": epoch, "best_epoch": best_epoch,
             "best_dev_iou": best_iou, "pid": os.getpid(), "elapsed_seconds": row["elapsed_seconds"]})
        print(json.dumps({"epoch": epoch, "loss": row["train_loss"], "dev_iou": metrics["iou"],
                          "dev_coverage90": metrics["coverage_at_90"], "marginal_iou": marginal["iou"],
                          "partial_fraction": row["partial_training_fraction"], "best_epoch": best_epoch,
                          "elapsed_seconds": row["elapsed_seconds"]}), flush=True)
        if epoch - best_epoch >= args.patience:
            break
    saved = torch.load(args.output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(saved["model"])
    _, _, _, cached = evaluate_interval(model, dev_loader, device, cache=True, marginal_only=args.marginal_only)
    torch.save(cached, args.output / "synthetic_dev_outputs.pt")
    dump(args.output / "status.json", {"status": "complete", "epochs": len(history), "best_epoch": best_epoch,
         "best_dev_iou": best_iou, "test_accessed": False, "elapsed_seconds": round(time.monotonic() - started, 1),
         "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else None})


if __name__ == "__main__":
    main()
