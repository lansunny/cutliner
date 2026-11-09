"""Develop semantic-curve temporal models; the heldout split stays sealed."""

import argparse
import hashlib
import json
import random
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from curve_data import VERSION, EvidenceCurves, collate_curves, reduce_metrics
from model_curves import CurveGrounder, curve_objective
from prepare_speech import DATA
from public_zip import ROOT
from train import iou, to_device


@torch.inference_mode()
def evaluate_curves(model, loader, dataset, device, disable=None):
    model.eval()
    predictions = []
    for raw in loader:
        batch = to_device(raw, device)
        output = model(batch, disable)
        scores = output["logits"].cpu()
        distance = (output["distances"].cpu()*30).double()
        times = raw["times"].double()
        durations = torch.tensor([dataset.rows[i]["duration"] for i in raw["indices"]], dtype=torch.float64)
        starts = (times-distance[:, :, 0]).clamp(min=0)
        ends = torch.minimum(durations[:, None], times+distance[:, :, 1])
        valid = raw["mask"] & (starts < ends)
        scores = scores.masked_fill(~valid, -float("inf"))
        selected = scores.argmax(-1)
        for i, index in enumerate(raw["indices"]):
            row, j = dataset.rows[index], int(selected[i])
            span = [float(starts[i, j]), float(ends[i, j])] if valid[i, j] else None
            overlap = max((iou(span, target) for target in row["relevant_windows"]), default=0.) if span else 0.
            predictions.append({"qid": row["qid"], "video": row["video"], "span": span, "iou": overlap})
    by_id = {row["qid"]: row for row in predictions}
    predictions = [by_id[row["qid"]] for row in dataset.rows]
    return reduce_metrics(dataset.rows, [row["iou"] for row in predictions]), predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", choices=["visual", "speech", "fusion"], default=["visual", "speech", "fusion"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--lr", type=float, default=0.0005)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, dev = EvidenceCurves("train"), EvidenceCurves("dev")
    if {row["video"] for row in train.rows} & {row["video"] for row in dev.rows}:
        raise ValueError("Source-video leakage")
    priors = json.loads((DATA/"curve_baselines_dev.json").read_text(encoding="utf-8"))["modes"]
    folder = ROOT/"curve_runs"/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True)
    config = {**vars(args), "curve_version": VERSION, "heldout_accessed": False,
              "selection": "dev source-macro R1@0.5 + 0.25 mean IoU; epoch zero eligible",
              "task": "HiREST derived step-query localization, not official benchmark task",
              "inputs": "Full author 1fps visual and timed speech features; paired frozen text encoders",
              "supervision": "Human temporal intervals, never speech boundaries as action truth",
              "hashes": {name: hashlib.sha256((DATA/name).read_bytes()).hexdigest()
                         for name in ("train.jsonl", "dev.jsonl", "split_plan.json", "curve_baselines_dev.json")},
              "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                for name in ("model_curves.py", "train_curves.py", "curve_data.py")}}
    (folder/"config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    dev_loader = DataLoader(dev, batch_size=args.batch_size, collate_fn=collate_curves)
    runs = []
    for mode in args.modes:
      for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        prior = {key: priors[mode][key] for key in ("alpha", "left_seconds", "right_seconds")}
        model = CurveGrounder(mode, prior, args.hidden).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.03)
        loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_curves,
                            generator=torch.Generator().manual_seed(seed))
        checkpoint_path = folder/f"{mode}_seed{seed}.pt"
        metrics, _ = evaluate_curves(model, dev_loader, dev, device)
        if abs(metrics["R1_iou0.5"]-priors[mode]["dev"]["R1_iou0.5"]) > 1e-8:
            raise ValueError("Epoch-zero decoder does not reproduce static control")
        best = metrics["source_macro_R1_iou0.5"]+0.25*metrics["mean_iou"]
        history = [{"epoch": 0, "dev": metrics}]

        def save(epoch, values):
            torch.save({"state_dict": model.state_dict(), "mode": mode, "prior": prior, "seed": seed,
                        "epoch": epoch, "development": values, "config": config}, checkpoint_path)

        save(0, metrics)
        print(f"{mode} seed{seed} epoch0 R1@.5={metrics['R1_iou0.5']:.4f}", flush=True)
        for epoch in range(1, args.epochs+1):
            model.train()
            total = 0.
            for raw in loader:
                batch = to_device(raw, device)
                optimizer.zero_grad(set_to_none=True)
                output = model(batch)
                loss = curve_objective(output, batch)
                if not loss.isfinite():
                    raise ValueError("Non-finite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                total += float(loss.detach())*len(raw["indices"])
            metrics, _ = evaluate_curves(model, dev_loader, dev, device)
            score = metrics["source_macro_R1_iou0.5"]+0.25*metrics["mean_iou"]
            history.append({"epoch": epoch, "loss": total/len(train), "dev": metrics})
            print(f"{mode} seed{seed} epoch{epoch}: loss={total/len(train):.4f}, R1@.5={metrics['R1_iou0.5']:.4f}, sourceMacro={metrics['source_macro_R1_iou0.5']:.4f}, mIoU={metrics['mean_iou']:.4f}", flush=True)
            if score > best:
                best = score
                save(epoch, metrics)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        controls = {}
        for disable in (["visual", "speech"] if mode == "fusion" else []):
            controls[f"{disable}_disabled"], _ = evaluate_curves(model, dev_loader, dev, device, disable)
        runs.append({"mode": mode, "seed": seed, "selected_epoch": checkpoint["epoch"],
                     "selected_dev": checkpoint["development"], "controls_dev": controls,
                     "trained_parameters": sum(p.numel() for p in model.parameters()), "history": history})
        (folder/"development_report.json").write_text(json.dumps({"config": config, "runs": runs}, indent=2), encoding="utf-8")
    print(f"Curve development run finished: {folder}; heldout untouched", flush=True)


if __name__ == "__main__":
    main()
