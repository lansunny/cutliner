"""Modality-independent multiscale intervals, with query-dependent evidence statistics."""

import hashlib
import json

import numpy as np
import torch
from torch.utils.data import Dataset

from curve_data import EvidenceCurves
from prepare_speech import DATA

VERSION = "dense_multiscale_evidence_4to256s_v1"
LENGTHS = (4, 8, 12, 20, 32, 48, 80, 128, 256)
VISUAL = list(range(11)) + [24, 25]
SPEECH = list(range(11, 22)) + [26]


def proposal_features(curves, times, duration):
    n = len(times)
    all_starts, all_ends = [], []
    for seconds in LENGTHS:
        length = min(n, max(1, round(seconds*n/duration)))
        starts = np.unique(np.append(np.arange(0, n-length+1, 2), n-length))
        all_starts.append(starts)
        all_ends.append(starts+length)
    start, end = np.concatenate(all_starts), np.concatenate(all_ends)
    width = end-start
    prefix = np.pad(curves.cumsum(0), ((1, 0), (0, 0)))
    squares = np.pad((curves**2).cumsum(0), ((1, 0), (0, 0)))
    features = []
    for raw_col, z_col in ((0, 2), (1, 3)):
        def avg(a, b, col=z_col):
            return (prefix[b, col]-prefix[a, col])/np.maximum(b-a, 1)
        mean = avg(start, end)
        std = np.sqrt(np.maximum((squares[end, z_col]-squares[start, z_col])/width - mean**2, 0))
        quarter = np.maximum(1, width//4)
        head, tail = avg(start, start+quarter), avg(end-quarter, end)
        left, right = avg(np.maximum(0, start-quarter), start), avg(end, np.minimum(n, end+quarter))
        maxima = []
        for a, b in zip(all_starts, all_ends):
            windows = np.lib.stride_tricks.sliding_window_view(curves[:, z_col], int(b[0]-a[0]))
            maxima.append(windows[a].max(-1))
        active = np.pad((curves[:, z_col] > 0.5).cumsum(), (1, 0))
        features += [avg(start, end, raw_col), mean, np.concatenate(maxima), std, head, tail,
                     left, right, head-left, tail-right, (active[end]-active[start])/width]
    novelty = (prefix[end, 6]-prefix[start, 6])/width
    # All grid intervals are independent of modalities and human annotations.
    # This matters: disabling speech must not retain speech-chosen proposals.
    features += [np.log1p(width*duration/n)/4, width/n, novelty,
                 curves[start, 6]+curves[np.minimum(n-1, end), 6],
                 (prefix[end, 7]-prefix[start, 7])/width,
                 start/n, end/n]
    spans = np.stack([start*duration/n, end*duration/n], axis=-1).astype(np.float32)
    values = np.stack(features, axis=-1).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite interval features")
    return values, spans


def prepare_proposals(split):
    if split not in ("train", "dev"):
        raise ValueError("Heldout preparation requires frozen selection")
    source = EvidenceCurves(split)
    path = DATA/f"{split}_interval_evidence.npz"
    if path.exists():
        print(f"Preserving interval cache {split}", flush=True)
        return
    features, spans, targets, offsets = [], [], [], [0]
    for i, row in enumerate(source.rows):
        a, b = source.offsets[i:i+2]
        values, intervals = proposal_features(source.curves[a:b], source.times[a:b], row["duration"])
        overlap = np.zeros(len(intervals), np.float32)
        for start, end in row["relevant_windows"]:
            intersection = np.maximum(0, np.minimum(end, intervals[:, 1])-np.maximum(start, intervals[:, 0]))
            union = np.maximum(1e-6, intervals[:, 1]-intervals[:, 0]+end-start-intersection)
            overlap = np.maximum(overlap, intersection/union)
        features.append(values)
        spans.append(intervals)
        targets.append(overlap)
        offsets.append(offsets[-1]+len(values))
        if (i+1)%500 == 0:
            print(f"Prepared interval evidence {split} {i+1}/{len(source)}", flush=True)
    np.savez_compressed(path, features=np.concatenate(features), spans=np.concatenate(spans),
                        ious=np.concatenate(targets), offsets=np.array(offsets),
                        ids=np.array([r["qid"] for r in source.rows]), version=VERSION,
                        manifest_sha256=hashlib.sha256((DATA/f"{split}.jsonl").read_bytes()).hexdigest())
    print(f"Interval evidence ready {split}: {offsets[-1]} candidates", flush=True)


class IntervalEvidence(Dataset):
    def __init__(self, split):
        self.rows = [json.loads(line) for line in (DATA/f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        with np.load(DATA/f"{split}_interval_evidence.npz", allow_pickle=False) as data:
            if str(data["version"]) != VERSION or data["ids"].tolist() != [r["qid"] for r in self.rows]:
                raise ValueError("Interval cache identity differs")
            if str(data["manifest_sha256"]) != hashlib.sha256((DATA/f"{split}.jsonl").read_bytes()).hexdigest():
                raise ValueError("Interval manifest changed")
            for name in ("features", "spans", "ious", "offsets"):
                setattr(self, name, data[name].copy())

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        a, b = self.offsets[index:index+2]
        return {"features": torch.from_numpy(self.features[a:b]), "spans": torch.from_numpy(self.spans[a:b]),
                "ious": torch.from_numpy(self.ious[a:b]), "index": index}


def collate_intervals(items):
    b, n = len(items), max(len(item["features"]) for item in items)
    result = {"features": torch.zeros(b, n, 29), "spans": torch.zeros(b, n, 2),
              "ious": torch.zeros(b, n), "mask": torch.zeros(b, n, dtype=torch.bool),
              "indices": [item["index"] for item in items]}
    for i, item in enumerate(items):
        size = len(item["features"])
        for key in ("features", "spans", "ious"):
            result[key][i, :size] = item[key]
        result["mask"][i, :size] = True
    return result


if __name__ == "__main__":
    torch.set_num_threads(2)
    for split in ("train", "dev"):
        prepare_proposals(split)
