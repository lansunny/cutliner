"""Prepare an immutable, video-disjoint small action study from public SSV2 data."""

import argparse
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path

import requests

from actions import ACTIONS, OPPOSITES, normalize_label
from prepare_data import ARCHIVE, BASE, BudgetReader, PilotBudgetReached, labels
from prepare_model import ROOT, sha256_file


def dump_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def make_plan(annotations, pilot_manifest, quotas, seed):
    # Those labels already influenced the pilot. Never relabel them as test.
    pilot = [json.loads(line) for line in pilot_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    used_val = {row["group_id"].removeprefix("ssv2_") for row in pilot if row["split"] != "train"}
    by_label = {normalize_label(spec["label"]): action for action, spec in ACTIONS.items()}
    targets = {}
    for official_split, filename in (("train", "train"), ("validation", "validation")):
        rows = json.loads((annotations / f"something-something-v2-{filename}.json").read_text(encoding="utf-8"))
        for row in rows:
            action = by_label.get(normalize_label(row.get("template", "")))
            if not action:
                continue
            video_id = str(row["id"])
            if video_id in targets:
                raise ValueError("Official annotation ID leakage")
            split = "train"
            if official_split == "validation":
                bucket = int(hashlib.sha256(f"{seed}:{video_id}".encode()).hexdigest(), 16) % 2
                split = "val" if video_id in used_val or bucket == 0 else "test"
            targets[video_id] = {"split": split, "action": action, "official_split": official_split}
    return {"version": 1, "seed": seed, "quotas": quotas, "targets": targets,
            "pilot_manifest_sha256": sha256_file(pilot_manifest),
            "previously_seen_val_ids": sorted(used_val),
            "selection": "bounded archive prefix until per-action quotas; no score-based selection",
            "test_scope": "held out from our head training/tuning, not external to pretrained SSV2 backbone"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="action_v1")
    parser.add_argument("--train-per-class", type=int, default=32)
    parser.add_argument("--val-per-class", type=int, default=8)
    parser.add_argument("--test-per-class", type=int, default=8)
    parser.add_argument("--budget-mib", type=int, default=3072)
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.name.replace("_", "").isalnum():
        parser.error("Use a simple experiment name")
    if min(args.train_per_class, args.val_per_class, args.test_per_class, args.budget_mib, args.seconds) <= 0:
        parser.error("Quotas and budgets must be positive")
    data = ROOT / "data"
    annotations, videos = data / "ssv2_annotations", data / "ssv2_pilot_videos"
    labels(annotations)
    videos.mkdir(exist_ok=True)
    quotas = {"train": args.train_per_class, "val": args.val_per_class, "test": args.test_per_class}
    plan = make_plan(annotations, data / "pilot_pairs.jsonl", quotas, args.seed)
    plan_path = data / f"{args.name}_split_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Existing frozen split plan differs; choose a new experiment name")
    else:
        dump_new(plan_path, plan)
    manifest = data / f"{args.name}_pairs.jsonl"
    if manifest.exists():
        print(f"Immutable manifest already prepared: {manifest}")
        return
    targets = plan["targets"]
    selected, counts = {}, Counter()
    # Reuse old files, with the same label-blind deterministic selection rule.
    for path in sorted(videos.glob("*.webm")):
        target = targets.get(path.stem)
        if not target or not path.stat().st_size:
            continue
        key = (target["split"], target["action"])
        if counts[key] < quotas[target["split"]]:
            selected[path.stem] = target
            counts[key] += 1

    def complete():
        return all(counts[(split, action)] >= quota for split, quota in quotas.items() for action in ACTIONS)

    consumed = 0
    if not complete():
        byte_budget = args.budget_mib * 1024**2
        print(f"Frozen pools saved. Streaming <= {args.budget_mib} MiB of official data", flush=True)
        with requests.get(BASE + ARCHIVE, headers={"Range": f"bytes=0-{byte_budget - 1}"}, stream=True, timeout=(10, 30)) as response:
            response.raise_for_status()
            reader = BudgetReader(response.raw, byte_budget, args.seconds)
            try:
                with tarfile.open(fileobj=reader, mode="r|gz") as archive:
                    for member in archive:
                        name = Path(member.name).name
                        video_id = Path(name).stem
                        if not member.isfile() or not name.endswith(".webm") or video_id not in targets:
                            continue
                        target = targets[video_id]
                        key = (target["split"], target["action"])
                        if video_id in selected or counts[key] >= quotas[target["split"]]:
                            continue
                        path = videos / name
                        if not path.exists():
                            if not 0 < member.size <= 32 * 1024**2:
                                raise ValueError("Unexpected member size")
                            payload = archive.extractfile(member).read()
                            if len(payload) != member.size:
                                raise ValueError("Incomplete video; refusing to save")
                            with path.open("xb") as stream:
                                stream.write(payload)
                        selected[video_id] = target
                        counts[key] += 1
                        print(f"Selected {key[0]}/{key[1]} {counts[key]}/{quotas[key[0]]}: {video_id}", flush=True)
                        if complete():
                            break
            except (PilotBudgetReached, tarfile.ReadError, EOFError):
                print("Budget ended; complete files retained. Rerun with a larger budget if needed", flush=True)
            consumed = reader.count
    report = {"complete": complete(), "compressed_bytes_read": consumed,
              "counts": {f"{split}/{action}": counts[(split, action)] for split in quotas for action in ACTIONS},
              "split_plan_sha256": sha256_file(plan_path), "test_scope": plan["test_scope"]}
    # Each invocation gets a new report; old evidence is never overwritten.
    from datetime import datetime
    report_path = data / f"{args.name}_download_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    dump_new(report_path, report)
    if not complete():
        print(json.dumps(report, indent=2))
        raise SystemExit("Incomplete quotas: no training manifest written")
    records, digest_splits = [], {}
    for video_id, target in sorted(selected.items()):
        path = (videos / f"{video_id}.webm").resolve()
        digest = sha256_file(path)
        digest_splits.setdefault(digest, set()).add(target["split"])
        for query, label in ((target["action"], 1), (OPPOSITES[target["action"]], 0)):
            records.append({"sample_id": f"ssv2_{video_id}_{query}", "group_id": f"ssv2_{video_id}",
                            "video_path": str(path), "video_sha256": digest, "start": 0.0, "end": None,
                            "action": query, "label": label, "split": target["split"],
                            "annotation_source": "official_ssv2_primary_template",
                            "supervision_scope": "primary_action_match_not_exhaustive_presence",
                            "negative_rule": None if label else "paired_opposite_primary_template"})
    if any(len(splits) > 1 for splits in digest_splits.values()):
        raise ValueError("Byte-identical source files overlap splits")
    with manifest.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n")
    print(json.dumps({**report, "source_videos": len(selected), "pairs": len(records), "manifest": str(manifest)}, indent=2))


if __name__ == "__main__":
    main()
