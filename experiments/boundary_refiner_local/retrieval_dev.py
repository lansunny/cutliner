"""Development evaluation on real QD-DETR top-1 proposals, without GT selection.

The coarse checkpoint is the existing QVHighlights-trained QD-DETR. This is a
cross-dataset transfer diagnostic, not a Charades-trained SOTA comparison. Only
train/dev splits are accessible by default; final test requires a frozen protocol.
Source content review is not evaluated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset import BoundaryDataset, collate
from extract_features import CLIP_ROOT, PROFILE
from model import BoundaryRefiner
from prepare_data import DEFAULT_ROOT, dump, file_hash, jsonl, read_jsonl
from train import evaluate, interval_metrics, summarize
from final_protocol import load_frozen_protocol


def build_candidates(root: Path, output: Path, split="dev", keep_all=False, final_protocol=None):
    if output.exists():
        raise ValueError("Candidate snapshot already exists; preserve it")
    sys.path.insert(0, str(CLIP_ROOT))
    from run_local import load_checkpoint, locate

    torch.set_num_threads(3)
    model, config = load_checkpoint("cpu")
    folder = root / "features" / PROFILE
    if split == "test":
        load_frozen_protocol(final_protocol, root)
    elif split not in ("train", "dev"):
        raise ValueError("Unknown candidate split")
    rows = read_jsonl(root / "manifests" / f"{split}.complete.jsonl")
    records = []
    started = time.monotonic()
    with torch.inference_mode():
        for i, row in enumerate(rows):
            with np.load(folder / "videos" / f"{row['video_id']}.npz", allow_pickle=False) as data:
                times, visual = data["times"], data["visual"].astype(np.float32)
            centers = np.arange(1., row["duration_seconds"], 2.)
            chosen = np.abs(times[:, None] - centers[None]).argmin(0)
            if len(np.unique(chosen)) != len(chosen):
                raise ValueError("Sparse coarse observations unexpectedly duplicated")
            with np.load(folder / "texts" / f"{row['case_id']}.npz", allow_pickle=False) as data:
                tokens = torch.from_numpy(data["tokens"].astype(np.float32))[None]
            text = F.normalize(tokens, dim=-1, eps=1e-5)
            proposals, _ = locate(model, torch.from_numpy(visual[chosen]), text,
                                  torch.ones(1, tokens.shape[1]), config, 0., row["duration_seconds"], "cpu")
            if not proposals:
                raise ValueError(f"No valid coarse proposals: {row['case_id']}; do not silently drop it")
            # Ranking uses only model score. Ground truth is never consulted.
            best = proposals[0]
            records.append({"case_id": row["case_id"], "video_id": row["video_id"],
                            "coarse": [best["start"], best["end"]], "score": best["score"],
                            "coarse_observed_pts": times[chosen].tolist(), "proposal_count": len(proposals)})
            if keep_all:
                records[-1]["proposals"] = proposals
            if (i + 1) % 50 == 0:
                print(f"Real coarse retrieval {i + 1}/{len(rows)}; {time.monotonic() - started:.1f}s", flush=True)
    output.mkdir(parents=True)
    jsonl(output / "candidates.jsonl", records)
    dump(output / "configuration.json", {"split": split, "ground_truth_used_for_candidate_selection": False,
         "source": "existing QVHighlights-trained QD-DETR; cross-dataset transfer diagnostic",
         "checkpoint_sha256": file_hash(CLIP_ROOT / "weights/clip_qd_detr_qvhighlight.ckpt"),
         "feature_profile": PROFILE, "coarse_cadence_seconds": 2., "queries": len(records),
         "all_ranked_nms_proposals_archived": keep_all,
         "test_accessed": split == "test", "elapsed_seconds": time.monotonic() - started})


class RetrievedDataset(BoundaryDataset):
    def __init__(self, root, candidates):
        super().__init__(root, "dev")
        self.candidates = {r["case_id"]: r for r in read_jsonl(candidates / "candidates.jsonl")}
        if self.candidates.keys() != {r["case_id"] for r in self.rows}:
            raise ValueError("Every dev case must have an independently retrieved candidate")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        coarse = self.candidates[row["case_id"]]["coarse"]
        a, b = coarse
        if not 0 <= a < b <= row["duration_seconds"]:
            raise ValueError("Invalid real coarse interval")
        window = max(0., a - 6), min(row["duration_seconds"], b + 6)
        return self.observe(row, coarse, window, "real_retrieval_top1")


def audit_candidates(root, candidates):
    truth = {r["case_id"]: r for r in read_jsonl(root / "manifests/dev.complete.jsonl")}
    proposals = read_jsonl(candidates / "candidates.jsonl")
    rows, bounds, observable = [], [], []
    for proposal in proposals:
        row = truth[proposal["case_id"]]
        a, b = proposal["coarse"]
        lo, hi = max(0., a - 6), min(row["duration_seconds"], b + 6)
        start, end = row["target_interval_seconds"]
        available = max(0., min(hi, end) - max(lo, start))
        bounds.append(available / (end - start))
        observable.append(lo <= start < end <= hi)
        rows.append({"case_id": row["case_id"], "video_id": row["video_id"], "kind": "real_retrieval_top1",
                     "metrics": interval_metrics([a, b], [start, end])})
    report = {"scope": "development diagnostics, not candidate selection or final test",
              "coarse": summarize(rows), "target_fully_in_context_fraction": float(np.mean(observable)),
              "mean_oracle_iou_upper_bound_within_context": float(np.mean(bounds)),
              "test_accessed": False}
    dump(candidates / "baseline_audit.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--build-candidates", action="store_true")
    parser.add_argument("--candidate-split", choices=["train", "dev", "test"], default="dev")
    parser.add_argument("--keep-all-proposals", action="store_true")
    parser.add_argument("--final-protocol", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(3)
    if args.build_candidates:
        build_candidates(args.root, args.candidates, split=args.candidate_split, keep_all=args.keep_all_proposals,
                         final_protocol=args.final_protocol)
        return
    if args.audit_only:
        audit_candidates(args.root, args.candidates)
        return
    if args.run is None or args.output is None:
        parser.error("--run and --output required for evaluation")
    if args.output.exists():
        raise ValueError("Preserve earlier evaluation snapshots")
    dataset = RetrievedDataset(args.root, args.candidates)
    loader = DataLoader(dataset, batch_size=96, collate_fn=collate, num_workers=0)
    saved = torch.load(args.run / "best.pt", map_location="cpu", weights_only=True)
    config = saved["configuration"]
    device = args.device
    model = BoundaryRefiner(config["variant"], config["hidden"]).to(device)
    model.load_state_dict(saved["model"], strict=True)
    refined, predictions = evaluate(model, loader, device)
    coarse_rows, observable = [], []
    for i in range(len(dataset)):
        sample = dataset[i]
        a, b = sample["target"]
        lower, upper = sample["window"]
        observable.append(bool(lower <= a < b <= upper))
        coarse_rows.append({"case_id": sample["case_id"], "video_id": sample["video_id"], "kind": sample["kind"],
                            "prediction": sample["coarse"].tolist(), "target": sample["target"].tolist(),
                            "coarse": sample["coarse"].tolist(), "metrics": interval_metrics(sample["coarse"], sample["target"])})
    args.output.mkdir(parents=True)
    jsonl(args.output / "refined_predictions.jsonl", predictions)
    jsonl(args.output / "coarse_predictions.jsonl", coarse_rows)
    report = {"scope": "actual coarse top1 -> neural refinement on development queries; excludes Agent review/export",
              "run": str(args.run), "coarse": summarize(coarse_rows), "refined": refined,
              "target_fully_in_observation_window_fraction": float(np.mean(observable)),
              "test_accessed": False, "null_or_rejection_model_available": False}
    dump(args.output / "results.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
