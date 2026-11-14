"""Freeze local neural encoder; cache real proposal-set features for ranker study."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset import to_device
from interval_model import IntervalQualityRefiner
from prepare_data import DEFAULT_ROOT, dump, file_hash, immutable_bytes
from rank_topk import CandidatePool, pool_collate
from train import interval_metrics
from final_protocol import load_frozen_protocol

META_FIELDS = ["coarse_score", "rank_fraction", "coarse_start", "coarse_end", "coarse_width",
               "window_start", "window_end", "window_width", "pred_start", "pred_end", "pred_width",
               "pred_iou", "pred_complete", "outside_left", "outside_right", "start_entropy", "end_entropy",
               "clip_inside", "clip_outside", "clip_contrast"]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--train-candidates", type=Path)
    parser.add_argument("--dev-candidates", type=Path)
    parser.add_argument("--test-candidates", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=["train", "dev", "test"], default=["train", "dev"])
    parser.add_argument("--max-k", type=int, default=3)
    parser.add_argument("--final-protocol", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(3)
    for split in args.splits:
        if getattr(args, split + "_candidates") is None:
            parser.error(f"--{split}-candidates required for the requested split")
    is_test = "test" in args.splits
    if is_test:
        if args.splits != ["test"]:
            parser.error("Final-test feature extraction must be isolated from train/dev")
        protocol = load_frozen_protocol(args.final_protocol, args.root)
        if args.max_k != protocol["top_k"] or file_hash(args.run / "best.pt") not in {x["local_sha256"] for x in protocol["pipelines"]}:
            raise ValueError("Different final local encoder or candidate count")
    if json.loads((args.run / "status.json").read_text())["status"] != "complete":
        raise ValueError("Frozen terminal local encoder required")
    args.output.mkdir(parents=True, exist_ok=True)
    config = {"encoder_checkpoint_sha256": file_hash(args.run / "best.pt"), "max_k": args.max_k,
              "decoder": "quantile_q0.3", "local_encoder_frozen": True, "metadata_fields": META_FIELDS,
              "ground_truth_used_in_input_features": False, "test_accessed": is_test}
    immutable_bytes(args.output / "encoder_configuration.json", json.dumps(config, indent=2).encode())
    saved = torch.load(args.run / "best.pt", map_location="cpu", weights_only=True)
    model = IntervalQualityRefiner(backbone_variant=saved["configuration"].get("backbone_variant", "film"))
    model.load_state_dict(saved["model"], strict=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().requires_grad_(False).to(device)
    for split in args.splits:
        dest = args.output / f"{split}.pt"
        if dest.exists():
            raise ValueError(f"Preserve existing frozen feature cache: {dest}")
        candidates = getattr(args, split + "_candidates")
        data = CandidatePool(args.root, candidates, args.max_k, split=split)
        groups = {}
        print(f"Frozen candidate encoding {split}: {len(data)} windows", flush=True)
        for cpu in DataLoader(data, batch_size=64, collate_fn=pool_collate, num_workers=0):
            batch = to_device(cpu, device)
            out = model(batch)
            size = batch["edges"].shape[-1]
            i = (out["start"].softmax(-1).cumsum(-1) < .3).sum(-1).clamp(max=size - 1)
            j = (out["end"].softmax(-1).cumsum(-1) < .7).sum(-1).clamp(max=size - 1)
            j = torch.maximum(j, i + 1).minimum(batch["mask"].sum(-1))
            i = torch.minimum(i, j - 1)
            ids = torch.arange(len(i), device=device)
            pred = torch.stack([batch["edges"][ids, i], batch["edges"][ids, j]], -1)
            embedding = out["temporal_hidden"].sum(1) / batch["mask"].sum(1)[:, None]
            q = F.normalize(batch["query"], dim=-1)
            similarity = (F.normalize(batch["visual"], dim=-1) * q[:, None]).sum(-1)
            inside = batch["mask"] & (batch["times"] >= pred[:, :1]) & (batch["times"] <= pred[:, 1:])
            outside = batch["mask"] & ~inside
            sim_in = (similarity * inside).sum(-1) / inside.sum(-1).clamp(min=1)
            sim_out = (similarity * outside).sum(-1) / outside.sum(-1).clamp(min=1)
            entropy = []
            for key in ("start", "end"):
                p = out[key].softmax(-1)
                entropy.append(-(p * p.clamp(min=1e-8).log()).sum(-1) / batch["edge_mask"].sum(-1).float().log())
            duration = batch["source_duration"]
            coarse, window = batch["coarse"] / duration[:, None], batch["window"] / duration[:, None]
            relative = pred / duration[:, None]
            extra = torch.stack([torch.tensor(cpu["raw_coarse_score"], device=device),
                    torch.tensor(cpu["candidate_rank"], device=device) / args.max_k,
                    coarse[:, 0], coarse[:, 1], coarse[:, 1] - coarse[:, 0], window[:, 0], window[:, 1], window[:, 1] - window[:, 0],
                    relative[:, 0], relative[:, 1], relative[:, 1] - relative[:, 0],
                    out["iou_quality"][ids, i, j].sigmoid(), out["coverage_quality"][ids, i, j].sigmoid(),
                    out["outside"][:, 0].sigmoid(), out["outside"][:, 1].sigmoid(), entropy[0], entropy[1],
                    sim_in, sim_out, sim_in - sim_out], -1)
            for n, case in enumerate(cpu["case_id"]):
                if case not in groups:
                    groups[case] = {"case_id": case, "video_id": cpu["video_id"][n], "query": q[n].cpu().clone(),
                                    "target": cpu["target"][n].clone(), "items": []}
                # Metrics/labels are constructed AFTER label-free neural features.
                metric = interval_metrics(pred[n].cpu(), cpu["target"][n])
                groups[case]["items"].append({"embedding": embedding[n].cpu().clone(), "meta": extra[n].cpu().clone(),
                      "prediction": pred[n].cpu().clone(), "coarse": cpu["coarse"][n].clone(),
                      "window": cpu["window"][n].clone(), "rank": cpu["candidate_rank"][n],
                      "true_iou": metric["iou"], "true_complete": metric["coverage_at_90"]})
        packed = []
        for item in groups.values():
            nodes = sorted(item.pop("items"), key=lambda x: x["rank"])
            for key in ("embedding", "meta", "prediction", "coarse", "window"):
                item[key] = torch.stack([x[key] for x in nodes])
            for key in ("rank", "true_iou", "true_complete"):
                item[key] = torch.tensor([x[key] for x in nodes])
            packed.append(item)
        pending = dest.with_suffix(".pt.tmp")
        torch.save(packed, pending)
        pending.replace(dest)
        dump(args.output / f"{split}_configuration.json", {**config, "queries": len(packed), "candidate_windows": len(data),
             "candidate_archive_sha256": file_hash(candidates / "candidates.jsonl"),
             "manifest_sha256": file_hash(args.root / "manifests" / f"{split}.complete.jsonl")})
        print(f"Cached {split}: {len(packed)} query sets", flush=True)


if __name__ == "__main__":
    main()
