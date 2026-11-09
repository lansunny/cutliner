"""Audit a frozen head against a stronger paired-action backbone baseline."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from actions import OPPOSITES
from encoder import FrozenVideoEncoder
from prepare_model import ROOT, sha256_file
from train_head import choose_threshold, load_manifest, metrics
from verifier import ACTION_ORDER, load_head


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    args = parser.parse_args()
    rows = load_manifest(args.manifest)
    encoder = FrozenVideoEncoder()
    head, checkpoint = load_head(args.head, encoder.fingerprint)
    if checkpoint["manifest_sha256"] != sha256_file(args.manifest):
        raise ValueError("Audit manifest differs from training manifest")
    labels, splits, baseline, learned = [], [], [], []
    for row in rows:
        if row["split"] == "train":
            continue
        feature, logits, _, _ = encoder.encode(row["video_path"], row["start"], row["end"])
        pair = torch.tensor([logits[encoder.action_labels[row["action"]]], logits[encoder.action_labels[OPPOSITES[row["action"]]]]])
        baseline.append(float(pair.softmax(-1)[0]))
        with torch.inference_mode():
            learned.append(float(head(torch.from_numpy(feature)[None], torch.tensor([ACTION_ORDER.index(row["action"])] )).sigmoid()))
        splits.append(row["split"])
        labels.append(row["label"])
    y, b, h = torch.tensor(labels), torch.tensor(baseline), torch.tensor(learned)
    val, test = [i for i, s in enumerate(splits) if s == "val"], [i for i, s in enumerate(splits) if s == "test"]
    threshold, val_metrics = choose_threshold(b[val], y[val])
    report = {"head_sha256": sha256_file(args.head), "manifest_sha256": sha256_file(args.manifest),
              "paired_baseline_definition": "softmax over only query/opposite logits from the same frozen VideoMAE",
              "paired_baseline_threshold_on_val": threshold, "paired_baseline_val": val_metrics,
              "paired_baseline_test": metrics(b[test], y[test], threshold),
              "frozen_head_test": metrics(h[test], y[test], checkpoint["threshold"]),
              "audit_timing": "Stronger baseline added after first held-out report; no head or rejection policy retuning",
              "scope": "SSV2 primary-action pairs; test independent of our head tuning, not external-domain validation"}
    folder = ROOT / "audits" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True)
    (folder / "baseline_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Audit: {folder}")


if __name__ == "__main__":
    main()
