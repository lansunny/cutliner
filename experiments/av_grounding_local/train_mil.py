"""Evaluate low-capacity interval-bag metric learning; heldout remains sealed."""

import argparse
import hashlib
import json
import random
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_paired import PairedEvidence, collate_paired
from model_mil import MetricMILGrounder, mil_loss
from prepare_speech import DATA
from public_zip import ROOT
from train import to_device
from train_curves import evaluate_curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", choices=["visual", "speech", "fusion"], default=["fusion", "speech", "visual"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=.01)
    args = parser.parse_args()
    torch.set_num_threads(3)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, dev = PairedEvidence("train"), PairedEvidence("dev")
    if {r["video"] for r in train.rows} & {r["video"] for r in dev.rows}:
        raise ValueError("Original-video split leakage")
    priors = json.loads((DATA/"curve_baselines_dev.json").read_text(encoding="utf-8"))["modes"]
    folder = ROOT/"mil_runs"/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True)
    config = {**vars(args), "heldout_accessed": False,
              "task": "HiREST derived step-query localization, not official benchmark",
              "supervision": "MIL probability mass of at least some evidence within human intervals; no uniform-positive-frame assumption",
              "trainable": "Diagonal positive metric in each paired pretrained space, global modality gate and signed narration lag kernel",
              "decoder": "Unchanged train-only fitted global window; isolates retrieval training from decoder changes",
              "selection": "Epoch0 and multiples of5: dev source-macro R1@0.5 + 0.25 mIoU",
              "manifest_sha256": {s: hashlib.sha256((DATA/f"{s}.jsonl").read_bytes()).hexdigest() for s in ("train", "dev")},
              "source_sha256": {s: hashlib.sha256((ROOT/s).read_bytes()).hexdigest()
                                for s in ("dataset_paired.py", "model_mil.py", "train_mil.py", "train_curves.py")}}
    (folder/"config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    dev_loader = DataLoader(dev, batch_size=args.batch_size, collate_fn=collate_paired)
    runs = []
    for mode in args.modes:
      for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        prior = {key: priors[mode][key] for key in ("alpha", "left_seconds", "right_seconds")}
        model = MetricMILGrounder(mode, prior).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
        loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_paired,
                            generator=torch.Generator().manual_seed(seed))
        path = folder/f"{mode}_seed{seed}.pt"
        metrics, _ = evaluate_curves(model, dev_loader, dev, device)
        best = metrics["source_macro_R1_iou0.5"]+.25*metrics["mean_iou"]
        history = [{"epoch": 0, "dev": metrics}]

        def save(epoch, values):
            torch.save({"state_dict": model.state_dict(), "mode": mode, "seed": seed, "epoch": epoch,
                        "prior": prior, "development": values, "config": config}, path)

        save(0, metrics)
        print(f"{mode} seed{seed} epoch0 R1@.5={metrics['R1_iou0.5']:.4f}", flush=True)
        for epoch in range(1, args.epochs+1):
            model.train()
            total = 0.
            for raw in loader:
                batch = to_device(raw, device)
                optimizer.zero_grad(set_to_none=True)
                loss = mil_loss(model, model(batch), batch)
                if not loss.isfinite():
                    raise ValueError("Non-finite metric adaptation loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                total += float(loss.detach())*len(raw["indices"])
            if epoch%5:
                print(f"{mode} seed{seed} epoch{epoch}: loss={total/len(train):.4f}", flush=True)
                continue
            metrics, _ = evaluate_curves(model, dev_loader, dev, device)
            history.append({"epoch": epoch, "loss": total/len(train), "dev": metrics})
            print(f"{mode} seed{seed} epoch{epoch}: loss={total/len(train):.4f}, R1@.5={metrics['R1_iou0.5']:.4f}, sourceMacro={metrics['source_macro_R1_iou0.5']:.4f}, mIoU={metrics['mean_iou']:.4f}", flush=True)
            score = metrics["source_macro_R1_iou0.5"]+.25*metrics["mean_iou"]
            if score > best:
                best = score
                save(epoch, metrics)
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        controls = {}
        for disable in (["visual", "speech"] if mode == "fusion" else []):
            controls[f"{disable}_disabled"], _ = evaluate_curves(model, dev_loader, dev, device, disable)
        runs.append({"mode": mode, "seed": seed, "selected_epoch": ckpt["epoch"], "selected_dev": ckpt["development"],
                     "controls_dev": controls, "trained_parameters": sum(p.numel() for p in model.parameters()), "history": history})
        (folder/"development_report.json").write_text(json.dumps({"config": config, "runs": runs}, indent=2), encoding="utf-8")
    print(f"MIL development run finished: {folder}; heldout untouched", flush=True)


if __name__ == "__main__":
    main()
