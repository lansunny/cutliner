"""Build primary-action pairs from lawfully obtained official SSV2 annotations.

This script never downloads datasets or accepts license terms. Opposite-class
negatives supervise the primary template, not exhaustive absence of every action.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from actions import ACTIONS, OPPOSITES, normalize_label


def build_records(annotation_path, videos, split, limit, seed):
    rows = json.loads(annotation_path.read_text(encoding="utf-8"))
    by_template = {normalize_label(spec["label"]): action for action, spec in ACTIONS.items()}
    grouped = {action: [] for action in ACTIONS}
    for row in rows:
        action = by_template.get(normalize_label(row.get("template", "")))
        if action is None:
            continue
        video_id = str(row["id"])
        source = next((videos / f"{video_id}{suffix}" for suffix in (".webm", ".mp4")
                       if (videos / f"{video_id}{suffix}").is_file()), None)
        if source is not None:
            grouped[action].append((video_id, source.resolve()))
    records = []
    for action, candidates in grouped.items():
        candidates.sort(key=lambda item: item[0])
        random.Random(seed).shuffle(candidates)
        selected = candidates[:limit] if limit else candidates
        if not selected:
            raise ValueError(f"No actual videos for {split}/{action}; annotations alone are insufficient")
        for video_id, source in selected:
            for query, label in ((action, 1), (OPPOSITES[action], 0)):
                records.append({
                    "sample_id": f"ssv2_{video_id}_{query}", "group_id": f"ssv2_{video_id}",
                    "video_path": str(source), "start": 0.0, "end": None,
                    "action": query, "label": label, "split": split,
                    "annotation_source": "official_ssv2_primary_template",
                    "supervision_scope": "primary_action_match_not_exhaustive_presence",
                    "negative_rule": "paired_opposite_primary_template" if not label else None,
                })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--val-json", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-per-class", type=int, default=200)
    parser.add_argument("--val-per-class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new manifest path to preserve it")
    if args.train_per_class < 0 or args.val_per_class < 0:
        parser.error("Per-class limits must be nonnegative (0 means all)")
    train = build_records(args.train_json, args.videos, "train", args.train_per_class, args.seed)
    val = build_records(args.val_json, args.videos, "val", args.val_per_class, args.seed)
    if {row["group_id"] for row in train} & {row["group_id"] for row in val}:
        raise ValueError("Source videos overlap between training and validation")
    records = train + val
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(args.output.resolve()), "pairs": len(records),
                      "source_videos": len({row["group_id"] for row in records}),
                      "pairs_per_split": dict(Counter(row["split"] for row in records)),
                      "scope": "Primary action template matching; not exhaustive action presence labels"}, indent=2))


if __name__ == "__main__":
    main()
