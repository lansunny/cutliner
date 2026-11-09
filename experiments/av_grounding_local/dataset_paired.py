"""Aligned query/frame and query/speech interactions, retaining semantic dimensions."""

import numpy as np
import torch
from torch.nn import functional as F

from curve_data import EvidenceCurves, collate_curves
from dataset_speech_candidates import video_features
from prepare_speech import DATA


class PairedEvidence(EvidenceCurves):
    def __init__(self, split):
        super().__init__(split)
        for suffix, attribute in (("query_features", "speech_queries"), ("eva_query_features", "visual_queries")):
            with np.load(DATA/f"{split}_{suffix}.npz", allow_pickle=False) as cache:
                if cache["ids"].tolist() != [r["qid"] for r in self.rows]:
                    raise ValueError("Paired query order differs")
                setattr(self, attribute, torch.from_numpy(cache["features"].copy()))

    def __getitem__(self, index):
        result = super().__getitem__(index)
        row = self.rows[index]
        visual, speech, available, times = video_features(row["video"], row["duration"])
        if len(times) != len(result["times"]) or not torch.equal(times, result["times"]):
            raise ValueError("Semantic feature timeline differs")
        result.update(visual=F.normalize(visual, dim=-1), speech=speech,
                      visual_query=self.visual_queries[index], speech_query=self.speech_queries[index])
        return result


def collate_paired(items):
    batch = collate_curves(items)
    b, n = batch["mask"].shape
    batch["visual"] = torch.zeros(b, n, 1024)
    batch["speech"] = torch.zeros(b, n, 384)
    batch["visual_query"] = torch.stack([item["visual_query"] for item in items])
    batch["speech_query"] = torch.stack([item["speech_query"] for item in items])
    for i, item in enumerate(items):
        length = len(item["visual"])
        batch["visual"][i, :length] = item["visual"]
        batch["speech"][i, :length] = item["speech"]
    return batch
