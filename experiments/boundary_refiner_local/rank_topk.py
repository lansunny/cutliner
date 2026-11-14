"""Actual neural top-K selection diagnostic, with identical observations per ranker.

Ranks by learned scores, raw QD-DETR scores, or frozen CLIP similarity. Ground
truth is used only after prediction for metrics; oracle recall is never reported
as achieved neural performance. Development only; no final-test access.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from calibrate_policy import bootstrap_delta
from dataset import BoundaryDataset, collate
from interval_model import IntervalQualityRefiner, quality_decode
from prepare_data import DEFAULT_ROOT, dump, file_hash, jsonl, read_jsonl
from train import predictions_from_batch, summarize


class CandidatePool(BoundaryDataset):
    def __init__(self, root, candidates, max_k, split="dev"):
        super().__init__(root, split)
        configuration = json.loads((candidates / "configuration.json").read_text(encoding="utf-8"))
        if configuration["split"] != split or configuration["ground_truth_used_for_candidate_selection"]:
            raise ValueError("Candidate split or selection provenance mismatch")
        self.proposals = {x["case_id"]: x for x in read_jsonl(candidates / "candidates.jsonl")}
        if self.proposals.keys() != {x["case_id"] for x in self.rows}:
            raise ValueError("Candidate/query mismatch")
        self.indices = [(i, j) for i, row in enumerate(self.rows)
                        for j in range(min(max_k, len(self.proposals[row["case_id"]]["proposals"])))]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i, j = self.indices[index]
        row = self.rows[i]
        proposal = self.proposals[row["case_id"]]["proposals"][j]
        a, b = proposal["start"], proposal["end"]
        sample = self.observe(row, [a, b], [max(0., a - 6), min(row["duration_seconds"], b + 6)], "topk_rerank")
        return {**sample, "candidate_rank": j + 1, "raw_coarse_score": proposal["score"],
                "source_duration": row["duration_seconds"]}

    def observation_budget(self, k):
        counts = []
        for row in self.rows:
            _, times = self.videos[row["video_id"]]
            observed = np.zeros(len(times), dtype=bool)
            for proposal in self.proposals[row["case_id"]]["proposals"][:k]:
                a, b = max(0., proposal["start"] - 6), min(row["duration_seconds"], proposal["end"] + 6)
                observed |= (times >= a) & (times < b)
            counts.append(observed.sum())
        return float(np.mean(counts))


def pool_collate(samples):
    batch = collate(samples)
    batch["candidate_rank"] = [x["candidate_rank"] for x in samples]
    batch["raw_coarse_score"] = [x["raw_coarse_score"] for x in samples]
    batch["source_duration"] = torch.tensor([x["source_duration"] for x in samples], dtype=torch.float32)
    return batch


def choose(records, field, k):
    groups = defaultdict(list)
    for row in records:
        if row["candidate_rank"] <= k:
            groups[row["case_id"]].append(row)
    # No target/metric access in candidate selection.
    return [max(rows, key=lambda x: (x[field], -x["candidate_rank"])) for rows in groups.values()]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-k", type=int, default=5)
    parser.add_argument("--decoder", choices=["quality", "quantile"], default="quantile")
    parser.add_argument("--q", type=float, default=.3)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("New diagnostic output required")
    state = json.loads((args.run / "status.json").read_text(encoding="utf-8"))
    if state["status"] != "complete":
        raise ValueError("Use terminal checkpoints only")
    torch.set_num_threads(3)
    saved = torch.load(args.run / "best.pt", map_location="cpu", weights_only=True)
    model = IntervalQualityRefiner(backbone_variant=saved["configuration"].get("backbone_variant", "film"))
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    data = CandidatePool(args.root, args.candidates, args.max_k)
    records = []
    for batch in DataLoader(data, batch_size=64, collate_fn=pool_collate, num_workers=0):
        output = model(batch)
        if args.decoder == "quality":
            prediction, _ = quality_decode(output, batch, fallback=False)
            i = (batch["edges"] - prediction[:, :1]).abs().masked_fill(~batch["edge_mask"], 1e6).argmin(-1)
            j = (batch["edges"] - prediction[:, 1:]).abs().masked_fill(~batch["edge_mask"], 1e6).argmin(-1)
        else:
            size = batch["edges"].shape[-1]
            i = (output["start"].softmax(-1).cumsum(-1) < args.q).sum(-1).clamp(max=size - 1)
            j = (output["end"].softmax(-1).cumsum(-1) < 1 - args.q).sum(-1).clamp(max=size - 1)
            # Keep a legal interval without consulting true boundaries.
            j = torch.maximum(j, i + 1).minimum(batch["mask"].sum(-1))
            i = torch.minimum(i, j - 1)
            ids = torch.arange(len(i))
            prediction = torch.stack([batch["edges"][ids, i], batch["edges"][ids, j]], -1)
        ids = torch.arange(len(i))
        learned = (output["iou_quality"][ids, i, j].sigmoid()
                   + .25 * output["coverage_quality"][ids, i, j].sigmoid()
                   - .25 * output["outside"].sigmoid().max(-1).values)
        similarity = (F.normalize(batch["visual"], dim=-1) * F.normalize(batch["query"], dim=-1)[:, None]).sum(-1)
        inside = batch["mask"] & (batch["times"] >= prediction[:, :1]) & (batch["times"] <= prediction[:, 1:])
        clip = (similarity * inside).sum(-1) / inside.sum(-1).clamp(min=1)
        rows = predictions_from_batch(prediction, batch)
        for n, row in enumerate(rows):
            row.update(candidate_rank=batch["candidate_rank"][n], learned_score=float(learned[n]),
                       clip_score=float(clip[n]), raw_coarse_score=batch["raw_coarse_score"][n])
        records.extend(rows)
    args.output.mkdir(parents=True)
    jsonl(args.output / "all_scored_candidates.jsonl", records)
    reference = choose(records, "raw_coarse_score", 1)
    results = {}
    for k in (1, 2, 3, 5):
        if k > args.max_k:
            continue
        results[k] = {"mean_unique_real_frames_observed": data.observation_budget(k), "rankers": {}}
        for field in ("raw_coarse_score", "clip_score", "learned_score"):
            selected = choose(records, field, k)
            metrics = summarize(selected)
            metrics["mean_selected_original_rank"] = float(np.mean([x["candidate_rank"] for x in selected]))
            results[k]["rankers"][field] = metrics
            jsonl(args.output / f"k{k}_{field}.jsonl", selected)
        results[k]["neural_vs_top1_delta"] = {m: bootstrap_delta(choose(records, "learned_score", k), reference, m)
                                               for m in ("iou", "coverage_at_90", "redundancy_seconds")}
    report = {"scope": "actual neural selection/refinement on development candidates; not Agent/export accuracy",
              "ground_truth_used_for_selection": False, "test_accessed": False, "checkpoint_sha256": file_hash(args.run / "best.pt"),
              "candidate_sha256": file_hash(args.candidates / "candidates.jsonl"),
              "decoder": args.decoder, "q": args.q, "results": results}
    dump(args.output / "results.json", report)
    compact = {k: {name: {m: stats[m] for m in ("iou", "coverage_at_90", "redundancy_seconds")}
                    for name, stats in value["rankers"].items()} for k, value in results.items()}
    print(json.dumps(compact, indent=2), flush=True)


if __name__ == "__main__":
    main()
