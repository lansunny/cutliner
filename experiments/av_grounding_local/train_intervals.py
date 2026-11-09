"""Learn query-conditional multiscale audiovisual interval ranking on human time labels."""

import argparse
import hashlib
import json
import random
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from curve_data import reduce_metrics
from prepare_speech import DATA
from proposal_data import VERSION, IntervalEvidence, SPEECH, VISUAL, collate_intervals
from public_zip import ROOT
from train import to_device


def static_scores(features, prior):
    visual = (1-prior["peak_mix"])*features[..., 1]+prior["peak_mix"]*features[..., 2]
    speech = (1-prior["peak_mix"])*features[..., 12]+prior["peak_mix"]*features[..., 13]
    # Train-only global duration preference, not a hard duration filter.
    seconds = torch.expm1(features[..., 22]*4).clamp(min=0.01) if isinstance(features, torch.Tensor) else np.maximum(0.01, np.expm1(features[..., 22]*4))
    log = torch.log if isinstance(features, torch.Tensor) else np.log
    return prior["alpha"]*visual+(1-prior["alpha"])*speech - 0.5*log(seconds/prior["length_seconds"])**2


def static_evaluate(dataset, prior):
    scores = static_scores(dataset.features, prior)
    values = []
    for a, b in zip(dataset.offsets[:-1], dataset.offsets[1:]):
        values.append(float(dataset.ious[a+int(scores[a:b].argmax())]))
    return reduce_metrics(dataset.rows, values)


def fit_priors(train, dev):
    path = DATA/"interval_baselines_dev.json"
    if path.exists():
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["version"] != VERSION:
            raise ValueError("Static interval report version differs")
        return report["modes"]
    modes = {}
    for mode, alphas in (("visual", [1.]), ("speech", [0.]), ("fusion", [0., .25, .5, .75, 1.])):
        best = None
        for alpha in alphas:
            for peak_mix in (0., .5, 1.):
                for length in (4, 8, 12, 20, 32, 48, 80):
                    prior = {"alpha": alpha, "peak_mix": peak_mix, "length_seconds": length}
                    metrics = static_evaluate(train, prior)
                    score = metrics["source_macro_R1_iou0.5"]+0.25*metrics["mean_iou"]
                    if best is None or score > best[0]:
                        best = (score, prior, metrics)
        modes[mode] = {"prior": best[1], "train": best[2], "dev": static_evaluate(dev, best[1])}
        print(f"Train-fitted static {mode}: {modes[mode]}", flush=True)
    report = {"version": VERSION, "selection": "train only", "heldout_accessed": False, "modes": modes}
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    return modes


class IntervalRanker(nn.Module):
    def __init__(self, mode, prior, mean, std, hidden=64):
        super().__init__()
        self.mode, self.prior = mode, dict(prior)
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).clone())
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32).clone().clamp(min=.01))
        self.network = nn.Sequential(nn.Linear(29, hidden), nn.GELU(), nn.Dropout(.2),
                                     nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, batch, disable=None):
        raw = batch["features"].clone()
        normalized = (raw-self.mean)/self.std
        for condition, columns in ((self.mode == "speech" or disable == "visual", VISUAL),
                                   (self.mode == "visual" or disable == "speech", SPEECH)):
            if condition:
                raw[:, :, columns] = 0
                normalized[:, :, columns] = 0
        logits = static_scores(raw, self.prior) + self.network(normalized).squeeze(-1)
        return logits.masked_fill(~batch["mask"], -1e4)


def ranking_loss(logits, batch):
    target = batch["ious"]
    weights = target**4*batch["mask"]
    weights = weights/weights.sum(-1, keepdim=True).clamp(min=1e-8)
    # Target distribution is built only for the loss, never included in inputs.
    listwise = -(weights*logits.log_softmax(-1)).sum(-1)
    return listwise.mean()


@torch.inference_mode()
def evaluate_intervals(model, loader, dataset, device, disable=None):
    model.eval()
    predictions = []
    for raw in loader:
        output = model(to_device(raw, device), disable).cpu()
        selected = output.argmax(-1)
        for i, index in enumerate(raw["indices"]):
            row, j = dataset.rows[index], int(selected[i])
            predictions.append({"qid": row["qid"], "video": row["video"],
                                "span": raw["spans"][i, j].tolist(), "iou": float(raw["ious"][i, j])})
    by_id = {row["qid"]: row for row in predictions}
    predictions = [by_id[row["qid"]] for row in dataset.rows]
    return reduce_metrics(dataset.rows, [row["iou"] for row in predictions]), predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", choices=["visual", "speech", "fusion"], default=["visual", "speech", "fusion"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=.001)
    args = parser.parse_args()
    torch.set_num_threads(3)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, dev = IntervalEvidence("train"), IntervalEvidence("dev")
    if {r["video"] for r in train.rows} & {r["video"] for r in dev.rows}:
        raise ValueError("Source-video split leakage")
    priors = fit_priors(train, dev)
    # Statistics are fit on train, before masking each modality explicitly.
    mean, std = train.features.mean(0), train.features.std(0)
    folder = ROOT/"interval_runs"/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True)
    config = {**vars(args), "version": VERSION, "heldout_accessed": False,
              "task": "HiREST derived step-query localization, not original benchmark",
              "proposal_policy": "All modality-independent grid intervals at 4..256 seconds, 2-frame stride; no GT insertions",
              "loss": "Listwise ranking against human-interval IoU^4 distribution",
              "selection": "dev source-macro R1@0.5 + 0.25 mIoU; epoch0 eligible",
              "manifest_sha256": {s: hashlib.sha256((DATA/f"{s}.jsonl").read_bytes()).hexdigest() for s in ("train", "dev")},
              "source_sha256": {s: hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in ("proposal_data.py", "train_intervals.py")}}
    (folder/"config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    dev_loader = DataLoader(dev, batch_size=args.batch_size, collate_fn=collate_intervals)
    runs = []
    for mode in args.modes:
      for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        model = IntervalRanker(mode, priors[mode]["prior"], mean, std, args.hidden).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.05)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr*.1)
        loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_intervals,
                            generator=torch.Generator().manual_seed(seed))
        path = folder/f"{mode}_seed{seed}.pt"
        metrics, _ = evaluate_intervals(model, dev_loader, dev, device)
        best = metrics["source_macro_R1_iou0.5"]+0.25*metrics["mean_iou"]
        history = [{"epoch": 0, "dev": metrics}]

        def save(epoch, values):
            torch.save({"state_dict": model.state_dict(), "mode": mode, "seed": seed, "epoch": epoch,
                        "prior": model.prior, "development": values, "config": config}, path)

        save(0, metrics)
        print(f"{mode} seed{seed} epoch0 R1@.5={metrics['R1_iou0.5']:.4f}", flush=True)
        for epoch in range(1, args.epochs+1):
            model.train()
            total = 0.
            for raw in loader:
                batch = to_device(raw, device)
                optimizer.zero_grad(set_to_none=True)
                loss = ranking_loss(model(batch), batch)
                if not loss.isfinite():
                    raise ValueError("Non-finite ranking loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
                optimizer.step()
                total += float(loss.detach())*len(raw["indices"])
            schedule.step()
            metrics, _ = evaluate_intervals(model, dev_loader, dev, device)
            history.append({"epoch": epoch, "loss": total/len(train), "dev": metrics})
            print(f"{mode} seed{seed} epoch{epoch}: loss={total/len(train):.4f}, R1@.5={metrics['R1_iou0.5']:.4f}, sourceMacro={metrics['source_macro_R1_iou0.5']:.4f}, mIoU={metrics['mean_iou']:.4f}", flush=True)
            score = metrics["source_macro_R1_iou0.5"]+0.25*metrics["mean_iou"]
            if score > best:
                best = score
                save(epoch, metrics)
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        controls = {}
        for disable in (["visual", "speech"] if mode == "fusion" else []):
            controls[f"{disable}_disabled"], _ = evaluate_intervals(model, dev_loader, dev, device, disable)
        runs.append({"mode": mode, "seed": seed, "selected_epoch": ckpt["epoch"], "selected_dev": ckpt["development"],
                     "controls_dev": controls, "trained_parameters": sum(p.numel() for p in model.parameters()), "history": history})
        (folder/"development_report.json").write_text(json.dumps({"config": config, "runs": runs}, indent=2), encoding="utf-8")
    print(f"Interval development run finished: {folder}; heldout untouched", flush=True)


if __name__ == "__main__":
    main()
