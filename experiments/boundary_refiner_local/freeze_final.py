"""Freeze a selected, replicated pipeline BEFORE final-test feature extraction."""

import argparse
import json
from pathlib import Path

from extract_features import CLIP_ROOT, CLIP_SHA, PROFILE
from prepare_data import DEFAULT_ROOT, file_hash, immutable_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    exp = args.root / "experiments"
    if (exp / "round_07/status.json").exists() is False:
        raise ValueError("Complete full-pipeline replication first")
    if json.loads((exp / "round_07/status.json").read_text())["status"] != "complete":
        raise ValueError("Full-pipeline replication is not terminal")
    study = json.loads((exp / "round_07/replication_results.json").read_text())
    if study["selected_pipeline_minus_fine_top1"]["iou"]["mean"] <= 0:
        raise ValueError("Selected pipeline has not improved development localization")
    if study["selected_pipeline_minus_fine_top1"]["coverage_at_90"]["mean"] < 0:
        raise ValueError("Selected pipeline still sacrifices development completeness")
    paths = [(17, exp / "round_03/representative_pair_s17", exp / "round_06/independent_s17")]
    paths += [(seed, exp / f"round_07/local_s{seed}", exp / f"round_07/ranker_s{seed}") for seed in (29, 43)]
    pipelines = []
    for seed, local, ranker in paths:
        for folder in (local, ranker):
            if json.loads((folder / "status.json").read_text())["status"] != "complete":
                raise ValueError("Cannot freeze a live/nonterminal checkpoint")
        config = json.loads((ranker / "configuration.json").read_text())
        if config["variant"] != "independent":
            raise ValueError("Protocol is for the development-selected lightweight independent ranker")
        pipelines.append({"seed": seed, "local_checkpoint": str(local / "best.pt"),
                          "local_sha256": file_hash(local / "best.pt"), "ranker_checkpoint": str(ranker / "best.pt"),
                          "ranker_sha256": file_hash(ranker / "best.pt")})
    here = Path(__file__).resolve().parent
    protocol = {"state": "frozen", "version": 1, "dataset_root": str(args.root.resolve()),
        "scope": "Charades-STA strict complete positive subset; interval retrieval/refinement, not semantic absence or Agent/export success",
        "manifest_sha256": {s: file_hash(args.root / "manifests" / f"{s}.complete.jsonl") for s in ("train", "dev", "test")},
        "clip_sha256": CLIP_SHA, "qd_sha256": file_hash(CLIP_ROOT / "weights/clip_qd_detr_qvhighlight.ckpt"),
        "feature_profile": PROFILE, "top_k": 3, "context_seconds": 6., "decoder_quantile": .3,
        "metadata_fields": json.loads((exp / "rerank_features_v1/encoder_configuration.json").read_text())["metadata_fields"],
        "ranker_variant": "independent", "pipelines": pipelines,
        "source_sha256": {path.name: file_hash(path) for path in sorted(here.glob("*.py"))},
        "evaluation": {"references": ["coarse_top1", "fine_top1", "frozen_quality_top3"],
                       "reported_seeds": [17, 29, 43], "no_best_seed_selection": True,
                       "metrics": ["iou", "coverage_at_90", "event_coverage", "redundancy_seconds", "boundary_mae_seconds"],
                       "bootstrap": "source-video clusters, 2000 draws, seed-averaged case metrics", "bootstrap_seed": 187},
        "acceptance_audit": {"iou_vs_coarse_ci_lower_positive": True, "coverage90_vs_coarse_ci_lower_nonnegative": True,
                             "redundancy_vs_coarse_mean_nonpositive": True, "iou_vs_fine_top1_mean_positive": True,
                             "coverage90_vs_fine_top1_mean_nonnegative": True},
        "interpretation": "Failure must be reported. Do not retune this protocol using its final-test scores.",
        "development_replication_sha256": file_hash(exp / "round_07/replication_results.json")}
    immutable_bytes(args.output, json.dumps(protocol, ensure_ascii=False, indent=2).encode())
    print(f"Frozen inference protocol: {args.output}; final-test metrics have not been consumed")


if __name__ == "__main__":
    main()
