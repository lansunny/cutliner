"""Executable invariants for the second architecture; not performance evidence."""

import importlib.util
from pathlib import Path

import numpy as np
import torch

from dataset import BoundaryDataset, collate, to_device
from interval_dataset import IntervalTrainingDataset
from interval_model import IntervalQualityRefiner, interval_objective, quality_decode
from model import BoundaryRefiner
from prepare_data import DEFAULT_ROOT


def main():
    torch.set_num_threads(2)
    root = DEFAULT_ROOT
    data = BoundaryDataset(root, "dev", limit_cases=8)
    batch = collate([data[i] for i in range(24)])
    spec = importlib.util.spec_from_file_location("frozen_v1", root / "experiments/round_01/source_snapshot/model.py")
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    state = torch.load(root / "experiments/round_01/film_s17/best.pt", map_location="cpu", weights_only=True)["model"]
    frozen, current = old.BoundaryRefiner("film").eval(), BoundaryRefiner("film").eval()
    frozen.load_state_dict(state)
    current.load_state_dict(state)
    with torch.no_grad():
        a, b = frozen(batch), current(batch)
        for key in ("start", "end", "membership"):
            assert torch.equal(a[key], b[key])
    print("Frozen v1 forward compatibility: exact match")
    model = IntervalQualityRefiner()
    loss, _ = interval_objective(model(batch), batch)
    loss.backward()
    assert torch.isfinite(loss) and model.left.weight.grad.abs().sum() > 0
    assert model.outside[-1].weight.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        out = model(batch)
        prediction, _ = quality_decode(out, batch)
        assert torch.all(prediction[:, 1] > prediction[:, 0])
        other = model({**batch, "target": batch["target"] + 123})
        assert torch.equal(out["iou_quality"], other["iou_quality"])
        out["outside"] = torch.full_like(out["outside"], 20)
        fallback, _ = quality_decode(out, batch)
        assert torch.equal(fallback, batch["coarse"])
    print("Quality gradients, legal spans, no GT leakage and exact fallback: pass")
    if torch.cuda.is_available():
        model = model.to("cuda").train()
        gpu = to_device(batch, "cuda")
        loss, _ = interval_objective(model(gpu), gpu)
        loss.backward()
        print(f"GPU gradient pass; peak allocated {torch.cuda.max_memory_allocated() / 1024**2:.0f} MiB")
    partial = IntervalTrainingDataset(root, "train", partial_context=True)
    count = 0
    for i in range(len(partial)):
        sample = partial[i]
        row = partial.rows[i // 3]
        assert np.allclose(sample["target"], row["target_interval_seconds"])
        a, b = sample["coarse"]
        expected = [max(0., a - 6), min(row["duration_seconds"], b + 6)]
        assert np.allclose(sample["window"], expected, atol=1e-5)
        lo, hi = sample["window"]
        start, end = sample["target"]
        coverage = max(0., min(hi, end) - max(lo, start)) / (end - start)
        assert coverage >= .64999
        count += int(not (lo <= start < end <= hi))
    print(f"All {len(partial)} augmented windows preserve original GT and coarse-derived context; {count} correctly marked partial")


if __name__ == "__main__":
    main()
