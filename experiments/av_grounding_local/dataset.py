"""Aligned visual/audio/query tensors with real QVHighlights interval labels."""

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from public_zip import ROOT


class AlignedMoments(Dataset):
    def __init__(self, manifest):
        self.manifest = Path(manifest)
        self.rows = [json.loads(line) for line in self.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.tensors = []
        features = ROOT / "data/features/qvhighlights"
        for row in self.rows:
            with np.load(features / f"clip_features/{row['vid']}.npz", allow_pickle=False) as data:
                visual = data["features"].astype(np.float32)
            audio = np.load(features / f"pann_features/{row['vid']}.npy", allow_pickle=False).astype(np.float32)
            with np.load(features / f"clip_text_features/qid{row['qid']}.npz", allow_pickle=False) as data:
                query = data["pooler_output"].astype(np.float32)
                query_tokens = data["last_hidden_state"].astype(np.float32)
            if visual.ndim != 2 or visual.shape[1] != 512 or audio.ndim != 2 or audio.shape[1] != 2048 or query.shape != (512,):
                raise ValueError(f"Unexpected feature dimensions for qid {row['qid']}")
            duration = float(row["duration"])
            n = min(len(visual), math.ceil(duration / 2))
            if n < 2 or n > 150:
                raise ValueError("Unexpected QVHighlights context length")
            # Keep visual tail if audio is shorter. Missing audio is explicitly masked.
            aligned_audio, audio_mask = np.zeros((n, 2048), np.float32), np.zeros(n, np.bool_)
            available = min(n, len(audio))
            aligned_audio[:available], audio_mask[:available] = audio[:available], True
            centers = np.minimum(np.arange(n, dtype=np.float32) * 2 + 1, duration - 1e-4)
            inside, distances = np.zeros(n, np.float32), np.zeros((n, 2), np.float32)
            # Assign each positive anchor to the shortest containing ground-truth span.
            for start, end in sorted(row["relevant_windows"], key=lambda span: span[1] - span[0], reverse=True):
                mask = (centers >= start) & (centers < end)
                inside[mask] = 1
                distances[mask, 0] = (centers[mask] - start) / duration
                distances[mask, 1] = (end - centers[mask]) / duration
            if not inside.any():
                raise ValueError(f"No observed positive anchor for qid {row['qid']}")
            self.tensors.append({"visual": torch.from_numpy(visual[:n]), "audio": torch.from_numpy(aligned_audio),
                                 "query": torch.from_numpy(query), "audio_mask": torch.from_numpy(audio_mask),
                                 "query_tokens": torch.from_numpy(query_tokens),
                                 "inside": torch.from_numpy(inside), "distances": torch.from_numpy(distances),
                                 "centers": torch.from_numpy(centers), "duration": duration})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.tensors[index], index


def collate(batch):
    examples, indices = zip(*batch)
    n = max(len(item["visual"]) for item in examples)
    nq = max(len(item["query_tokens"]) for item in examples)
    result = {"visual": torch.zeros(len(batch), n, 512), "audio": torch.zeros(len(batch), n, 2048),
              "query": torch.stack([item["query"] for item in examples]), "mask": torch.zeros(len(batch), n, dtype=torch.bool),
              "audio_mask": torch.zeros(len(batch), n, dtype=torch.bool), "inside": torch.zeros(len(batch), n),
              "distances": torch.zeros(len(batch), n, 2), "centers": torch.zeros(len(batch), n),
              "duration": torch.tensor([item["duration"] for item in examples]), "indices": list(indices)}
    result["query_tokens"] = torch.zeros(len(batch), nq, 512)
    result["query_mask"] = torch.zeros(len(batch), nq, dtype=torch.bool)
    for i, item in enumerate(examples):
        length = len(item["visual"])
        result["mask"][i, :length] = True
        query_length = len(item["query_tokens"])
        result["query_tokens"][i, :query_length] = item["query_tokens"]
        result["query_mask"][i, :query_length] = True
        for key in ("visual", "audio", "audio_mask", "inside", "distances", "centers"):
            result[key][i, :length] = item[key]
    return result
