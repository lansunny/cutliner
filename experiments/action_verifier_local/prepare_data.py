"""Fetch public official SSV2 labels and a bounded pilot video subset.

The source is the official public Qualcomm download page. No account, agreement
acceptance, mirror, or credential is used. Use is subject to its research terms.
This is a pipeline pilot, not a dataset size sufficient to claim generalization.
"""

import argparse
import json
import tarfile
import time
import zipfile
from collections import Counter
from pathlib import Path

import requests

from actions import ACTIONS, normalize_label
from prepare_model import ROOT

BASE = "https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Something-Something-V2/"
LABELS = "20bn-something-something-download-package-labels.zip"
ARCHIVE = "20bn-something-something-v2-00"


class PilotBudgetReached(Exception):
    pass


class BudgetReader:
    def __init__(self, raw, byte_budget, seconds):
        self.raw, self.limit = raw, byte_budget
        self.count = 0
        self.deadline = time.monotonic() + seconds
        self.next_report = 16 * 1024**2

    def read(self, size=-1):
        if self.count >= self.limit or time.monotonic() >= self.deadline:
            raise PilotBudgetReached
        size = min(self.limit - self.count, 64 * 1024 if size < 0 else size)
        chunk = self.raw.read(size)
        self.count += len(chunk)
        if self.count >= self.next_report:
            print(f"Read {self.count / 1024**2:.1f} MiB of the public compressed stream", flush=True)
            self.next_report += 16 * 1024**2
        return chunk


def labels(folder):
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / LABELS
    if not archive.exists():
        print("Downloading official label package (~5MB)", flush=True)
        with requests.get(BASE + LABELS, stream=True, timeout=(10, 30)) as response:
            response.raise_for_status()
            partial = archive.with_suffix(".zip.part")
            with partial.open("wb") as stream:
                for chunk in response.iter_content(256 * 1024):
                    stream.write(chunk)
            partial.replace(archive)
    aliases = {
        "train.json": "something-something-v2-train.json",
        "validation.json": "something-something-v2-validation.json",
        "labels.json": "something-something-v2-labels.json",
    }
    wanted = set(aliases.values())
    with zipfile.ZipFile(archive) as package:
        for info in package.infolist():
            original_name = Path(info.filename).name
            name = aliases.get(original_name, original_name)
            if name not in wanted:
                continue
            payload = package.read(info)
            json.loads(payload)
            target = folder / name
            if target.exists() and target.read_bytes() != payload:
                raise ValueError(f"Existing annotations differ; preserving {target}")
            if not target.exists():
                target.write_bytes(payload)
    missing = [name for name in wanted if not (folder / name).is_file()]
    if missing:
        raise ValueError(f"Unexpected label archive layout; missing {missing}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-mib", type=int, default=128)
    parser.add_argument("--seconds", type=int, default=90)
    parser.add_argument("--train-per-class", type=int, default=4)
    parser.add_argument("--val-per-class", type=int, default=2)
    parser.add_argument("--labels-only", action="store_true")
    args = parser.parse_args()
    if min(args.budget_mib, args.seconds, args.train_per_class, args.val_per_class) <= 0:
        parser.error("Pilot limits must be positive")
    annotations = ROOT / "data" / "ssv2_annotations"
    videos = ROOT / "data" / "ssv2_pilot_videos"
    labels(annotations)
    if args.labels_only:
        print(f"Official annotations ready: {annotations}")
        return
    videos.mkdir(parents=True, exist_ok=True)
    by_label = {normalize_label(spec["label"]): action for action, spec in ACTIONS.items()}
    targets = {}
    for split, filename in (("train", "something-something-v2-train.json"), ("val", "something-something-v2-validation.json")):
        for row in json.loads((annotations / filename).read_text(encoding="utf-8")):
            action = by_label.get(normalize_label(row.get("template", "")))
            if action:
                video_id = str(row["id"])
                if video_id in targets:
                    raise ValueError("Annotation video IDs overlap across splits")
                targets[video_id] = (split, action)
    counts = Counter()
    selected = []
    limits = {"train": args.train_per_class, "val": args.val_per_class}
    for path in videos.glob("*.webm"):
        if path.stem in targets and path.stat().st_size:
            split, action = targets[path.stem]
            counts[(split, action)] += 1
            selected.append(path.stem)
    complete = lambda: all(counts[(split, action)] >= limit for split, limit in limits.items() for action in ACTIONS)
    consumed = 0
    if not complete():
        byte_budget = args.budget_mib * 1024**2
        print(f"Streaming a bounded prefix (at most {args.budget_mib} MiB), not downloading the 19.4GB dataset", flush=True)
        with requests.get(BASE + ARCHIVE, headers={"Range": f"bytes=0-{byte_budget - 1}"}, stream=True, timeout=(10, 30)) as response:
            response.raise_for_status()
            bounded = BudgetReader(response.raw, byte_budget, args.seconds)
            try:
                with tarfile.open(fileobj=bounded, mode="r|gz") as archive:
                    for member in archive:
                        name = Path(member.name).name
                        if not member.isfile() or not name.endswith(".webm"):
                            continue
                        video_id = Path(name).stem
                        if video_id not in targets:
                            continue
                        split, action = targets[video_id]
                        target = videos / name
                        if counts[(split, action)] >= limits[split] or target.exists():
                            continue
                        if not 0 < member.size <= 32 * 1024**2:
                            raise ValueError("Unexpected pilot member size")
                        source = archive.extractfile(member)
                        payload = source.read()
                        if len(payload) != member.size:
                            raise ValueError("Incomplete video member; refusing to save")
                        with target.open("xb") as stream:
                            stream.write(payload)
                        counts[(split, action)] += 1
                        selected.append(video_id)
                        print(f"Selected {split}/{action}: {video_id} ({member.size} bytes)", flush=True)
                        if complete():
                            break
            except (PilotBudgetReached, tarfile.ReadError, EOFError):
                print("Pilot stream budget ended; retained complete selected videos only", flush=True)
            consumed = bounded.count
    report = {"source_page": "https://www.qualcomm.com/developer/software/something-something-v-2-dataset/downloads",
              "source_archive": BASE + ARCHIVE, "compressed_bytes_read": consumed,
              "pilot_quota_complete": complete(), "selected_video_ids": selected,
              "counts": {f"{split}/{action}": counts[(split, action)] for split in limits for action in ACTIONS},
              "note": "Small execution pilot only; no training/effectiveness claims. Research-use terms apply."}
    (ROOT / "data" / "pilot_download.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
