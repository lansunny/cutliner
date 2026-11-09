"""Train audio fusion while preserving the existing pretrained visual localizer."""

import argparse
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import AlignedMoments, collate
from public_zip import ROOT
from qd_audio import AdaptedGrounder, evaluate_adapter, load_teacher, targets
from train import to_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/adapter_v1")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--modes", nargs="+", choices=["audio_residual", "query_gated"], default=["audio_residual", "query_gated"])
    parser.add_argument("--lr", type=float, default=0.0003)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) <= 0 or args.lr <= 0:
        parser.error("Invalid training config")
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, dev = AlignedMoments(args.dataset / "train.jsonl"), AlignedMoments(args.dataset / "dev.jsonl")
    if {row["vid"].rsplit("_", 2)[0] for row in train.rows} & {row["vid"].rsplit("_", 2)[0] for row in dev.rows}:
        raise ValueError("Adapter source split leakage")
    dev_loader = DataLoader(dev, batch_size=args.batch_size, collate_fn=collate)
    teacher, criterion = load_teacher(device)
    folder = ROOT / "adapter_runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True)
    config = {"dataset": str(args.dataset.resolve()), "modes": args.modes, "seeds": args.seeds,
              "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
              "plan_sha256": hashlib.sha256((args.dataset / "split_plan.json").read_bytes()).hexdigest(),
              "teacher": "Existing CLIP-only QD-DETR frozen; dropout disabled",
              "teacher_sha256": hashlib.sha256((ROOT.parent / "video_grounding_local/weights/clip_qd_detr_qvhighlight.ckpt").read_bytes()).hexdigest(),
              "test_accessed": False}
    (folder / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    runs = []
    for mode in args.modes:
      for seed in args.seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = AdaptedGrounder(teacher, mode).to(device)
        optimizer = torch.optim.AdamW([p for p in model.fusion.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
        loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                            generator=torch.Generator().manual_seed(seed))
        baseline, _ = evaluate_adapter(model, dev_loader, dev, device)
        best = baseline["R1_iou0.5"] + 0.25 * baseline["mean_iou"]
        checkpoint_path = folder / f"{mode}_seed{seed}.pt"

        def save(epoch, metrics):
            torch.save({"fusion_state_dict": model.fusion.state_dict(), "mode": mode, "seed": seed,
                        "epoch": epoch, "development": metrics, "config": config}, checkpoint_path)

        save(0, baseline)
        history = [{"epoch": 0, "dev": baseline, "note": "Zero residual, pretrained starting point"}]
        print(f"{mode} seed{seed} initial dev: {baseline}", flush=True)
        for epoch in range(args.epochs):
            model.train()
            total = 0.0
            for raw in loader:
                batch = to_device(raw, device)
                optimizer.zero_grad(set_to_none=True)
                output, delta, _ = model(batch)
                loss_dict = criterion(output, targets(batch, train))
                loss = sum(value * criterion.weight_dict[key] for key, value in loss_dict.items() if key in criterion.weight_dict)
                loss = loss + 0.1 * delta.square().sum(-1)[batch["mask"]].mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.fusion.parameters(), 1.0)
                optimizer.step()
                total += float(loss.detach()) * len(raw["indices"])
            metrics, _ = evaluate_adapter(model, dev_loader, dev, device)
            score = metrics["R1_iou0.5"] + 0.25 * metrics["mean_iou"]
            history.append({"epoch": epoch + 1, "train_loss": total/len(train), "dev": metrics})
            print(f"{mode} seed{seed} epoch{epoch+1}: loss={total/len(train):.4f}, devR1@.5={metrics['R1_iou0.5']:.4f}, mIoU={metrics['mean_iou']:.4f}", flush=True)
            if score > best:
                best = score
                save(epoch + 1, metrics)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.fusion.load_state_dict(checkpoint["fusion_state_dict"])
        controls = {}
        for name, kwargs in (("audio_disabled", {"disable_audio": True}), ("audio_time_reverse", {"reverse_audio": True})):
            controls[name], _ = evaluate_adapter(model, dev_loader, dev, device, **kwargs)
        runs.append({"mode": mode, "seed": seed, "selected_epoch": checkpoint["epoch"], "baseline_dev": baseline,
                     "selected_dev": checkpoint["development"], "audio_controls_dev": controls, "history": history,
                     "trained_parameters": sum(p.numel() for p in model.fusion.parameters() if p.requires_grad)})
        (folder / "development_report.json").write_text(json.dumps({"config": config, "runs": runs}, indent=2), encoding="utf-8")
    print(f"Audio adaptation development results: {folder}; heldout not accessed", flush=True)


if __name__ == "__main__":
    main()
