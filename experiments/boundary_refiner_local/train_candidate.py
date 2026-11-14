"""Train small proposal-set networks on frozen local features; development only."""

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

from candidate_model import CandidateRanker, collate_candidate_sets, ranking_objective
from dataset import to_device
from prepare_data import dump, file_hash, jsonl
from train import interval_metrics, summarize


def selected_rows(indices, batch):
    rows = []
    for i, j in enumerate(indices.cpu().tolist()):
        prediction, target = batch["prediction"][i, j].cpu().tolist(), batch["target"][i].cpu().tolist()
        rows.append({"case_id": batch["case_id"][i], "video_id": batch["video_id"][i], "kind": "proposal_set",
                     "prediction": prediction, "target": target, "coarse": batch["coarse"][i, j].cpu().tolist(),
                     "candidate_rank": int(batch["rank"][i, j]), "metrics": interval_metrics(prediction, target)})
    return rows


@torch.inference_mode()
def evaluate_candidate(model, loader, device):
    model.eval()
    rows = []
    for cpu in loader:
        out = model(to_device(cpu, device))
        rows.extend(selected_rows(out["score"].argmax(-1), cpu))
    return summarize(rows), rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=["independent", "independent_deep", "joint", "metadata_only"], default="joint")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve previous runs")
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
    train = torch.load(args.features / "train.pt", map_location="cpu", weights_only=True)
    dev = torch.load(args.features / "dev.pt", map_location="cpu", weights_only=True)
    if {r["video_id"] for r in train} & {r["video_id"] for r in dev}:
        raise ValueError("Source video leakage")
    config = {"variant": args.variant, "seed": args.seed, "epochs": args.epochs,
              "train_queries": len(train), "dev_queries": len(dev), "local_encoder_frozen": True,
              "test_accessed": False, "encoder_selected_on_development": True,
              "label_scope": "relative quality of the selected annotated interval, not semantic absence",
              "selection_objective": "dev mean IoU + 0.25 * fraction covering at least 90 percent of annotated duration",
              "encoder_config": json.loads((args.features / "encoder_configuration.json").read_text()),
              "train_features_sha256": file_hash(args.features / "train.pt"),
              "dev_features_sha256": file_hash(args.features / "dev.pt"),
              "source_hashes": {name: file_hash(Path(__file__).with_name(name)) for name in ("candidate_model.py", "train_candidate.py")}}
    model = CandidateRanker(args.variant)
    config["trainable_parameters"] = sum(p.numel() for p in model.parameters())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    train_loader = DataLoader(train, batch_size=128, shuffle=True, num_workers=0, collate_fn=collate_candidate_sets,
                              generator=torch.Generator().manual_seed(args.seed))
    dev_loader = DataLoader(dev, batch_size=256, num_workers=0, collate_fn=collate_candidate_sets)
    args.output.mkdir(parents=True)
    dump(args.output / "configuration.json", config)
    best, best_epoch, history = -1., 0, []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for group in optimizer.param_groups:
            group["lr"] = 5e-4 * (.1 + .9 * (1 + math.cos(math.pi * (epoch - 1) / max(1, args.epochs - 1))) / 2)
        losses = []
        for cpu in train_loader:
            batch = to_device(cpu, device)
            optimizer.zero_grad(set_to_none=True)
            loss = ranking_objective(model(batch), batch)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite ranking objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics, predictions = evaluate_candidate(model, dev_loader, device)
        score = metrics["iou"] + .25 * metrics["coverage_at_90"]
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "dev": metrics, "score": score,
               "elapsed_seconds": time.monotonic() - started}
        history.append(row)
        if score > best:
            best, best_epoch = score, epoch
            torch.save({"model": model.state_dict(), "configuration": config, "epoch": epoch}, args.output / "best.pt")
            jsonl(args.output / "best_dev_predictions.jsonl", predictions)
            dump(args.output / "best_dev.json", row)
        dump(args.output / "status.json", {"status": "running", "epoch": epoch, "best_epoch": best_epoch,
             "best_score": best, "pid": os.getpid(), "test_accessed": False})
        dump(args.output / "history.json", history)
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps({"variant": args.variant, "epoch": epoch, "iou": metrics["iou"],
                              "complete90": metrics["coverage_at_90"], "redundancy": metrics["redundancy_seconds"],
                              "best_epoch": best_epoch, "seconds": round(row["elapsed_seconds"], 1)}), flush=True)
        if epoch - best_epoch >= args.patience:
            break
    dump(args.output / "status.json", {"status": "complete", "epochs": len(history), "best_epoch": best_epoch,
         "best_score": best, "test_accessed": False, "elapsed_seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()
