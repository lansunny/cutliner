"""Strong speech-retrieval controls before training a multimodal temporal network."""

import hashlib
import json
from functools import lru_cache

import numpy as np
import torch
from torch.nn import functional as F

from prepare_speech import DATA, feature_name, srt_intervals
from text_encoder import FrozenSentenceEncoder
from train import iou


def rows_for(split):
    if split not in ("train", "dev"):
        raise ValueError("Development preparation never opens heldout query labels")
    return [json.loads(line) for line in (DATA / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]


@lru_cache(maxsize=1024)
def speech_for(video):
    value = torch.load(DATA / "features" / feature_name("speech", video), map_location="cpu", weights_only=True).float()
    intervals = srt_intervals((DATA / f"ASR/{video.removesuffix('.mp4')}.srt").read_text(encoding="utf-8-sig"))
    if len(value) != len(intervals):
        raise ValueError("ASR timestamps and source embeddings differ")
    return F.normalize(value, dim=-1), intervals


def prepare_queries(encoder, split, rows):
    path = DATA / f"{split}_query_features.npz"
    manifest_hash = hashlib.sha256((DATA / f"{split}.jsonl").read_bytes()).hexdigest()
    ids = [row["qid"] for row in rows]
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if data["ids"].tolist() != ids or str(data["encoder"]) != encoder.fingerprint or str(data["manifest_sha256"]) != manifest_hash:
                raise ValueError("Existing speech-query cache differs")
            return data["features"]
    features = encoder.encode([row["query"] for row in rows])
    np.savez_compressed(path, features=features, ids=np.array(ids), encoder=encoder.fingerprint, manifest_sha256=manifest_hash)
    return features


def retrieve(rows, queries):
    result = []
    for row, query in zip(rows, queries):
        speech, intervals = speech_for(row["video"])
        if len(speech):
            scores = speech @ torch.from_numpy(query)
            index = int(scores.argmax())
            start, end = intervals[index]
            span = [max(0.0, start), min(row["duration"], end)]
            if span[1] <= span[0]:
                span = None
        else:
            span = None
        result.append({"qid": row["qid"], "video": row["video"], "span": span})
    return result


def measure(rows, retrieved, left=0.0, right=0.0):
    values = []
    for row, item in zip(rows, retrieved):
        if item["span"] is None:
            values.append(0.0)
            continue
        a, b = max(0.0, item["span"][0] + left), min(row["duration"], item["span"][1] + right)
        values.append(max((iou((a, b), target) for target in row["relevant_windows"]), default=0.0) if a < b else 0.0)
    return {"queries": len(values), "source_videos": len({row["video"] for row in rows}),
            "mean_iou": float(np.mean(values)),
            **{f"R1_iou{threshold}": float(np.mean(np.array(values) >= threshold)) for threshold in (0.3, 0.5, 0.7)}}


def main():
    torch.set_num_threads(4)
    output = DATA / "baselines_dev.json"
    if output.exists():
        raise SystemExit("Speech baseline report already exists; preserving it")
    encoder = FrozenSentenceEncoder()
    rows = {split: rows_for(split) for split in ("train", "dev")}
    features = {split: prepare_queries(encoder, split, values) for split, values in rows.items()}
    retrieved = {split: retrieve(values, features[split]) for split, values in rows.items()}
    candidates = []
    for left in (-20, -10, -5, 0, 5, 10):
        for right in (-10, -5, 0, 5, 10, 20, 40):
            metrics = measure(rows["train"], retrieved["train"], left, right)
            candidates.append((metrics["R1_iou0.5"] + 0.25*metrics["mean_iou"], -abs(left)-abs(right), left, right))
    _, _, left, right = max(candidates)
    report = {"task": "HiREST-derived step-caption query localization, not official step-caption generation",
              "encoder_fingerprint": encoder.fingerprint, "global_shift_selected_on": "train_only",
              "global_shift_seconds": {"left": left, "right": right}, "heldout_not_evaluated": True,
              "results": {split: {"raw_speech_semantic_retrieval": measure(values, retrieved[split]),
                                  "speech_with_global_window_adjustment": measure(values, retrieved[split], left, right)}
                          for split, values in rows.items()},
              "purpose": "Any trained fusion network must beat speech retrieval and a simple timing prior, not just a weak visual model"}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
