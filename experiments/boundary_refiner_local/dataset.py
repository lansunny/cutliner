"""Fixed, source-disjoint positive refinement benchmark on cached real frames.

The coarse intervals are SYNTHETIC ground-truth perturbations, not retrieval
predictions. Results on this protocol must not be described as end-to-end editing.
Only training perturbations vary by epoch; dev/test perturbations are frozen.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from extract_features import PROFILE
from prepare_data import read_jsonl

PROTOCOL = "synthetic_coarse_anchor_context6_v1"


def coarse_window(row, variant: int, seed: str):
    rng = random.Random(hashlib.sha256(f"{seed}:{row['case_id']}:{variant}".encode()).hexdigest())
    start, end = row["target_interval_seconds"]
    duration = row["duration_seconds"]
    if variant == 0:
        a, b = start - rng.uniform(1.5, 5), end + rng.uniform(1.5, 5)
        kind = "redundant"
    elif variant == 1:
        shift = rng.choice([-1, 1]) * rng.uniform(2, 4)
        a = min(max(0, start + shift), duration - (end - start))
        b = a + (end - start)
        kind = "shifted"
    else:
        length = end - start
        a = start + min(5., rng.uniform(.1, .25) * length)
        b = end - min(5., rng.uniform(.1, .25) * length)
        kind = "truncated"
    a, b = max(0, a), min(duration, b)
    if b <= a:
        raise ValueError("Invalid synthetic coarse interval")
    # Crucially, the OBSERVATION WINDOW is derived from the coarse anchor, not
    # from true boundaries plus symmetric GT-centered context at evaluation time.
    lower, upper = max(0, a - 6), min(duration, b + 6)
    if not lower <= start < end <= upper:
        raise ValueError("Positive protocol unexpectedly truncated its target")
    return (a, b), (lower, upper), kind


class BoundaryDataset(Dataset):
    def __init__(self, root: Path, split: str, seed: int = 17, limit_cases=None):
        self.rows = read_jsonl(root / "manifests" / f"{split}.complete.jsonl")
        if limit_cases is not None:
            self.rows = self.rows[:limit_cases]
        self.split, self.seed, self.epoch = split, seed, 0
        folder = root / "features" / PROFILE
        self.videos, self.texts = {}, {}
        for row in self.rows:
            vid = row["video_id"]
            if vid not in self.videos:
                with np.load(folder / "videos" / f"{vid}.npz", allow_pickle=False) as data:
                    if str(data["profile"]) != PROFILE or str(data["media_sha256"]) != row["video_sha256"]:
                        raise ValueError(f"Stale visual cache: {vid}")
                    self.videos[vid] = (data["visual"].astype(np.float32), data["times"].copy())
            with np.load(folder / "texts" / f"{row['case_id']}.npz", allow_pickle=False) as data:
                if str(data["query"]) != row["query"] or str(data["profile"]) != PROFILE:
                    raise ValueError("Stale query cache")
                self.texts[row["case_id"]] = (data["pooled"].astype(np.float32), data["tokens"].astype(np.float32))

    def __len__(self):
        return len(self.rows) * 3

    def __getitem__(self, index):
        row, variant = self.rows[index // 3], index % 3
        seed = f"train:{self.seed}:epoch:{self.epoch}" if self.split == "train" else "frozen_dev_and_test_v1"
        coarse, window, kind = coarse_window(row, variant, seed)
        return self.observe(row, coarse, window, kind)

    def observe(self, row, coarse, window, kind):
        lower, upper = window
        visual, times = self.videos[row["video_id"]]
        mask = (times >= lower) & (times < upper)
        visual, times = visual[mask], times[mask]
        if len(times) < 2:
            raise ValueError("Insufficient real observations in a local window")
        edges = np.concatenate(([lower], (times[:-1] + times[1:]) / 2, [upper])).astype(np.float32)
        if not np.all(np.diff(edges) > 0):
            raise ValueError("Invalid real-frame cell boundaries")
        start, end = row["target_interval_seconds"]
        membership = np.maximum(0, np.minimum(edges[1:], end) - np.maximum(edges[:-1], start)) / np.diff(edges)
        query, tokens = self.texts[row["case_id"]]
        return {"visual": visual, "times": times, "edges": edges, "query": query, "tokens": tokens,
                "target": np.asarray([start, end], np.float32), "membership": membership,
                "coarse": np.asarray(coarse, np.float32), "window": np.asarray(window, np.float32),
                "case_id": row["case_id"], "video_id": row["video_id"], "kind": kind}


def collate(rows):
    batch, max_t, max_q = len(rows), max(len(x["times"]) for x in rows), max(len(x["tokens"]) for x in rows)
    output = {"visual": torch.zeros(batch, max_t, 512), "times": torch.zeros(batch, max_t),
              "edges": torch.zeros(batch, max_t + 1), "membership": torch.zeros(batch, max_t),
              "tokens": torch.zeros(batch, max_q, 512), "mask": torch.zeros(batch, max_t, dtype=torch.bool),
              "edge_mask": torch.zeros(batch, max_t + 1, dtype=torch.bool),
              "token_mask": torch.zeros(batch, max_q, dtype=torch.bool)}
    for key in ("query", "target", "coarse", "window"):
        output[key] = torch.from_numpy(np.stack([x[key] for x in rows]))
    for i, row in enumerate(rows):
        t, q = len(row["times"]), len(row["tokens"])
        for key in ("visual", "times", "membership"):
            output[key][i, :t] = torch.from_numpy(row[key])
        output["edges"][i, :t + 1] = torch.from_numpy(row["edges"])
        output["tokens"][i, :q] = torch.from_numpy(row["tokens"])
        output["mask"][i, :t] = True
        output["edge_mask"][i, :t + 1] = True
        output["token_mask"][i, :q] = True
    for key in ("case_id", "video_id", "kind"):
        output[key] = [x[key] for x in rows]
    return output


def to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
