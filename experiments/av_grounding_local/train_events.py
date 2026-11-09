"""Train audiovisual event tools on actual paired media, without opening heldout features."""

import argparse
import copy
import hashlib
import json
import random
from datetime import datetime

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from event_dataset import EventFeatures, event_metrics, prepare_assets, select_threshold
from model_events import EventGrounder, ScoreCalibration
from prepare_ave import DATA
from public_zip import ROOT


def fit_calibration(values, labels, positive_weight, device):
    model = ScoreCalibration(values.mean((0, 1)), values.std((0, 1))).to(device)
    values, labels = values.to(device), labels.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.03)
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(values), labels, pos_weight=positive_weight)
        loss.backward()
        optimizer.step()
    return model.eval().cpu()


def prepare_calibrators(train, dev, positive_weight, device):
    path = DATA/"calibration_train_only.pt"
    if path.exists():
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if ckpt["plan_sha256"] != hashlib.sha256((DATA/"split_plan.json").read_bytes()).hexdigest():
            raise ValueError("Source split changed after score calibration")
        if ckpt["native_assets_sha256"] != hashlib.sha256((DATA/"native_classifiers.npz").read_bytes()).hexdigest():
            raise ValueError("Native classifier assets changed after calibration")
        models = {}
        for key in ("visual", "audio"):
            state = ckpt[key]
            models[key] = ScoreCalibration(state["mean"], state["std"])
            models[key].load_state_dict(state)
        return models, ckpt["alpha"], ckpt["static_dev"]
    models = {mode: fit_calibration(getattr(train, f"{mode}_prior"), train.labels, positive_weight, device) for mode in ("visual", "audio")}
    with torch.inference_mode():
        logits = {mode: models[mode](getattr(train, f"{mode}_prior")) for mode in models}
        best = None
        for alpha in (0., .25, .5, .75, 1.):
            metrics = select_threshold((alpha*logits["visual"]+(1-alpha)*logits["audio"]).sigmoid().numpy(), train.labels.numpy())
            if best is None or metrics["frame_macro_f1"] > best[0]:
                best = (metrics["frame_macro_f1"], alpha)
        alpha = best[1]
        dv, da = [models[mode](getattr(dev, f"{mode}_prior")) for mode in ("visual", "audio")]
        static = {"visual": select_threshold(dv.sigmoid().numpy(), dev.labels.numpy()),
                  "audio": select_threshold(da.sigmoid().numpy(), dev.labels.numpy()),
                  "late_fusion": select_threshold((alpha*dv+(1-alpha)*da).sigmoid().numpy(), dev.labels.numpy())}
    torch.save({**{mode: model.state_dict() for mode, model in models.items()}, "alpha": alpha, "static_dev": static,
                "fit_split": "train_only", "dev_used_only_for_final_global_thresholds": True,
                "plan_sha256": hashlib.sha256((DATA/"split_plan.json").read_bytes()).hexdigest(),
                "native_assets_sha256": hashlib.sha256((DATA/"native_classifiers.npz").read_bytes()).hexdigest()}, path)
    print(f"Train-fit prior fusion alpha={alpha}; calibrated static dev: {static}", flush=True)
    return models, alpha, static


@torch.inference_mode()
def probabilities_for(model, loader, device, disable=None):
    model.eval()
    return torch.cat([model({key: value.to(device) for key, value in batch.items()}, disable).sigmoid().cpu()
                      for batch in loader]).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", choices=["visual", "visual_wide", "audio", "concat", "gated"],
                        default=["visual", "audio", "visual_wide", "concat", "gated"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=.0005)
    args = parser.parse_args()
    torch.set_num_threads(3)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prepare_assets()
    train, dev = EventFeatures("train"), EventFeatures("dev")
    if {r["video"] for r in train.rows} & {r["video"] for r in dev.rows}:
        raise ValueError("Original-source leakage")
    positives = train.labels.sum((0, 1))
    positive_weight = ((len(train)*10-positives)/positives.clamp(min=1)).sqrt().clamp(1, 8).to(device)
    models, alpha, static = prepare_calibrators(train, dev, positive_weight, device)
    folder = ROOT/"event_runs"/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True)
    config = {**vars(args), "heldout_accessed": False, "pretrained_encoders_frozen": True,
              "task": "AVE-derived 28-class multi-label 1-second audible-and-visible events; not original query-grounding or official AVE benchmark",
              "split": "Author source splits group-cleaned test>dev>train; hash-selected class caps32/8/8",
              "selection": "dev frame-macro F1, with global threshold0.1..0.9 on dev; epoch0 and every5epochs",
              "decoder": "Contiguous thresholded seconds; highest sum(prob-threshold) interval per event query",
              "priors": "Native CLIP/PANN class evidence, train-only 56-parameter affine calibrations; alpha fit train only",
              "pretraining_limit": "PANN is pretrained on AudioSet, which is the source collection of AVE; not unseen-source or zero-shot audio evidence",
              "head_training_sources": len(train), "dev_sources": len(dev), "prior_visual_alpha": alpha,
              "plan_sha256": hashlib.sha256((DATA/"split_plan.json").read_bytes()).hexdigest(),
              "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                for name in ("event_dataset.py", "model_events.py", "train_events.py", "extract_ave.py", "audio_encoder.py")}}
    (folder/"config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    dev_loader = DataLoader(dev, batch_size=args.batch_size)
    runs = []
    for mode in args.modes:
      for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        model = EventGrounder(mode, copy.deepcopy(models["visual"]), copy.deepcopy(models["audio"]), alpha).to(device)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=.03)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr*.1)
        loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
        path = folder/f"{mode}_seed{seed}.pt"
        metrics = select_threshold(probabilities_for(model, dev_loader, device), dev.labels.numpy())
        best = metrics["frame_macro_f1"]
        history = [{"epoch": 0, "dev": metrics}]

        def save(epoch, values):
            torch.save({"state_dict": model.state_dict(), "mode": mode, "seed": seed, "epoch": epoch,
                        "alpha": alpha, "development": values, "config": config}, path)

        save(0, metrics)
        print(f"{mode} seed{seed} epoch0 F1={best:.4f}, queryR1@.5={metrics['event_query_R1_iou0.5']:.4f}", flush=True)
        for epoch in range(1, args.epochs+1):
            model.train()
            total = 0.
            for batch in loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(model(batch), batch["labels"], pos_weight=positive_weight)
                if not loss.isfinite():
                    raise ValueError("Non-finite event supervision loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                total += float(loss.detach())*len(batch["labels"])
            schedule.step()
            if epoch%5:
                continue
            metrics = select_threshold(probabilities_for(model, dev_loader, device), dev.labels.numpy())
            history.append({"epoch": epoch, "loss": total/len(train), "dev": metrics})
            print(f"{mode} seed{seed} epoch{epoch}: loss={total/len(train):.4f}, F1={metrics['frame_macro_f1']:.4f}, queryR1@.5={metrics['event_query_R1_iou0.5']:.4f}, alarm={metrics['unannotated_query_alarm_rate']:.4f}", flush=True)
            if metrics["frame_macro_f1"] > best:
                best = metrics["frame_macro_f1"]
                save(epoch, metrics)
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        controls = {}
        for disable in (["audio", "visual"] if mode in ("concat", "gated") else []):
            probability = probabilities_for(model, dev_loader, device, disable)
            controls[f"{disable}_disabled_same_threshold"] = event_metrics(probability, dev.labels.numpy(), ckpt["development"]["threshold"])
            controls[f"{disable}_disabled_recalibrated"] = select_threshold(probability, dev.labels.numpy())
        runs.append({"mode": mode, "seed": seed, "selected_epoch": ckpt["epoch"], "selected_dev": ckpt["development"],
                     "controls_dev": controls, "trained_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad), "history": history})
        (folder/"development_report.json").write_text(json.dumps({"config": config, "static_dev": static, "runs": runs}, indent=2), encoding="utf-8")
    print(f"Event development run finished: {folder}; heldout untouched", flush=True)


if __name__ == "__main__":
    main()
