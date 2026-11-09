"""Speech proposals plus complete 1-fps visual timelines; no ground truth in inputs."""

import json
import math
from functools import lru_cache

import numpy as np
import torch
from torch.utils.data import Dataset

from prepare_speech import DATA, feature_name
from speech_baselines import speech_for
from train import iou


@lru_cache(maxsize=600)
def video_features(video, duration):
    visual = torch.load(DATA / "features" / feature_name("visual", video), map_location="cpu", weights_only=True).float()
    speech, intervals = speech_for(video)
    n = len(visual)
    times = (torch.arange(n).float() + 0.5) * duration/n
    dense, available = torch.zeros(n, 384), torch.zeros(n, dtype=torch.bool)
    coverage = torch.zeros(n)
    half = duration/n/2
    for vector, (start, end) in zip(speech, intervals):
        overlap = (torch.minimum(times + half, torch.tensor(end)) - torch.maximum(times - half, torch.tensor(start))).clamp(min=0)
        chosen = (overlap > coverage) & (overlap > 0)
        dense[chosen], available[chosen], coverage[chosen] = vector, True, overlap[chosen]
    return visual, dense, available, times


def make_proposals(row, query, left, right):
    speech, intervals = speech_for(row["video"])
    duration = row["duration"]
    count = min(12, max(4, math.ceil(duration/120) + 3))
    proposals = []
    if len(speech):
        scores = (speech @ query).tolist()
        for index in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True):
            a, b = intervals[index]
            context = [max(0.0, a-45), min(duration, b+45)]
            if context[1] <= context[0] or any(iou(context, old["context"]) > 0.8 for old in proposals):
                continue
            base = [max(0.0, a+left), min(duration, b+right)]
            if base[1] <= base[0]:
                continue
            proposals.append({"base": base, "context": context, "semantic": scores[index], "global": False})
            if len(proposals) == count:
                break
    if not proposals:
        proposals.append({"base": [0.0, min(10.0, duration)], "context": [0.0, duration], "semantic": 0.0, "global": True})
    # A full-timeline recovery view shares the same encoded frames. It is not an
    # oracle window and never uses annotation times; initial score is lower.
    top = proposals[0]
    proposals.append({"base": top["base"], "context": [0.0, duration], "semantic": top["semantic"]-0.01, "global": True})
    return proposals


class SpeechCandidates(Dataset):
    def __init__(self, split):
        if split not in ("train", "dev", "test"):
            raise ValueError("Unknown speech split")
        self.split = split
        self.rows = [json.loads(line) for line in (DATA / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        with np.load(DATA / f"{split}_query_features.npz", allow_pickle=False) as cache:
            if cache["ids"].tolist() != [row["qid"] for row in self.rows]:
                raise ValueError("Query feature IDs differ from immutable manifest")
            queries = cache["features"].copy()
        prior = json.loads((DATA / "baselines_dev.json").read_text(encoding="utf-8"))["global_shift_seconds"]
        self.examples = []
        for row, query in zip(self.rows, queries):
            query = torch.from_numpy(query)
            props = make_proposals(row, query, prior["left"], prior["right"])
            # Targets below are training/evaluation supervision only, never model inputs.
            matched, full, coarse_ious = [], [], []
            for proposal in props:
                eligible = [span for span in row["relevant_windows"]
                            if span[0] >= proposal["context"][0]-1e-5 and span[1] <= proposal["context"][1]+1e-5]
                target = max(eligible or row["relevant_windows"], key=lambda span: iou(proposal["base"], span))
                matched.append(target)
                full.append(bool(eligible))
                coarse_ious.append(iou(proposal["base"], target))
            local = [i for i, proposal in enumerate(props) if full[i] and not proposal["global"]]
            oracle = max(local, key=lambda i: (coarse_ious[i], props[i]["semantic"])) if local else len(props)-1
            self.examples.append({"query": query, "proposals": props, "targets": torch.tensor(matched), "oracle": oracle,
                                  "video": row["video"], "duration": row["duration"]})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        example = self.examples[index]
        visual, speech, available, times = video_features(example["video"], example["duration"])
        return example, visual, speech, available, times, index


def collate_candidates(items):
    n, k, b = max(len(item[1]) for item in items), max(len(item[0]["proposals"]) for item in items), len(items)
    batch = {"visual": torch.zeros(b,n,1024), "speech": torch.zeros(b,n,384), "speech_available": torch.zeros(b,n,dtype=torch.bool),
             "times": torch.zeros(b,n), "mask": torch.zeros(b,n,dtype=torch.bool), "query": torch.stack([item[0]["query"] for item in items]),
             "duration": torch.tensor([item[0]["duration"] for item in items]), "context_mask": torch.zeros(b,k,n,dtype=torch.bool),
             "candidate_mask": torch.zeros(b,k,dtype=torch.bool), "base": torch.zeros(b,k,2), "semantic": torch.zeros(b,k),
             "metadata": torch.zeros(b,k,5), "target": torch.zeros(b,k,2), "oracle": torch.tensor([item[0]["oracle"] for item in items]),
             "indices": [item[-1] for item in items]}
    for i,(example,visual,speech,available,times,index) in enumerate(items):
        length = len(visual)
        batch["visual"][i,:length], batch["speech"][i,:length], batch["speech_available"][i,:length] = visual,speech,available
        batch["times"][i,:length], batch["mask"][i,:length] = times,True
        batch["target"][i,:len(example["proposals"])] = example["targets"]
        for j,proposal in enumerate(example["proposals"]):
            batch["context_mask"][i,j,:length] = (times >= proposal["context"][0]) & (times <= proposal["context"][1])
            if not batch["context_mask"][i,j].any():
                raise ValueError("Proposal context contains no observed source feature")
            batch["candidate_mask"][i,j] = True
            batch["base"][i,j] = torch.tensor(proposal["base"])
            batch["semantic"][i,j] = proposal["semantic"]
            batch["metadata"][i,j] = torch.tensor([proposal["base"][0]/example["duration"], proposal["base"][1]/example["duration"],
                                                    (proposal["context"][1]-proposal["context"][0])/example["duration"],
                                                    float(proposal["global"]), j/12])
    return batch
