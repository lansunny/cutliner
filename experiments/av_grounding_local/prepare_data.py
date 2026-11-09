"""Prepare video-source-disjoint QVHighlights AV subsets from the author archive."""

import argparse
import hashlib
import io
import json
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

from public_zip import ARCHIVE_SHA256, AUTHOR_SOURCE, ROOT, fetch_member, fetch_members

ANNOTATIONS = ROOT.parent / "video_grounding_local/vendor/lighthouse/data/qvhighlight"
LOCAL = threading.local()


def stable(value, seed):
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def source_group(row):
    return row["vid"].rsplit("_", 2)[0]


def feature_names(row):
    return [f"qvhighlights/clip_features/{row['vid']}.npz",
            f"qvhighlights/pann_features/{row['vid']}.npy",
            f"qvhighlights/clip_text_features/qid{row['qid']}.npz"]


def round_robin(rows, count, seed):
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda row: stable(row["qid"], seed)):
        groups[source_group(row)].append(row)
    ordered = sorted(groups, key=lambda group: stable(group, seed))
    selected, level = [], 0
    while len(selected) < count:
        added = 0
        for group in ordered:
            if level < len(groups[group]):
                selected.append(groups[group][level])
                added += 1
                if len(selected) == count:
                    return selected
        if not added:
            raise ValueError(f"Only {len(selected)} eligible rows, requested {count}")
        level += 1
    return selected


def validate_payload(payload, name):
    data = np.load(io.BytesIO(payload), allow_pickle=False)
    if isinstance(data, np.ndarray):
        arrays = [data]
    else:
        with data:
            arrays = [data[key] for key in data.files]
    if not arrays or any(array.dtype.kind not in "fiu" or not np.isfinite(array).all() for array in arrays):
        raise ValueError(f"Non-numeric/non-finite feature in {name}")


def download(entry, destination):
    if destination.exists():
        payload = destination.read_bytes()
        import zlib
        if len(payload) != entry["size"] or zlib.crc32(payload) & 0xffffffff != entry["crc"]:
            raise ValueError("Existing feature differs from frozen ZIP index")
        return 0
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
    payload, transferred = fetch_member(entry, LOCAL.session)
    validate_payload(payload, entry["name"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(payload)
    return transferred


def cached(entry):
    path = ROOT / "data/features" / entry["name"]
    if not path.exists():
        return False
    import zlib
    payload = path.read_bytes()
    if len(payload) != entry["size"] or zlib.crc32(payload) & 0xffffffff != entry["crc"]:
        raise ValueError(f"Existing feature differs from source index: {path}")
    return True


def download_group(entries):
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
    for attempt in range(3):
        try:
            payloads, transferred = fetch_members(entries, LOCAL.session)
            break
        except requests.RequestException as error:
            # Retry only incidental network/5xx failures. Auth/permission errors stop.
            status = error.response.status_code if error.response is not None else None
            if attempt == 2 or status is not None and status < 500:
                raise
            time.sleep(1 + 2 * attempt)
    for entry in entries:
        payload = payloads[entry["name"]]
        validate_payload(payload, entry["name"])
        destination = ROOT / "data/features" / entry["name"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(payload)
    return transferred, len(entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="av_v1")
    parser.add_argument("--train", type=int, default=1536)
    parser.add_argument("--dev", type=int, default=384)
    parser.add_argument("--test", type=int, default=384)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-transfer-mib", type=int, default=1536)
    args = parser.parse_args()
    if not args.name.replace("_", "").isalnum() or min(args.train, args.dev, args.test, args.workers) <= 0 or args.workers > 8:
        parser.error("Invalid name/counts/workers")
    entries = json.loads((ROOT / "data/qv_public_index.json").read_text(encoding="utf-8"))["entries"]
    by_name = {entry["name"]: entry for entry in entries}
    official = {split: [json.loads(line) for line in (ANNOTATIONS / f"highlight_{split}_release.jsonl").read_text(encoding="utf-8").splitlines()]
                for split in ("train", "val")}
    eligible = {split: [row for row in rows if all(name in by_name for name in feature_names(row))]
                for split, rows in official.items()}
    if {source_group(row) for row in official["train"]} & {source_group(row) for row in official["val"]}:
        raise ValueError("Official source-video split overlap")
    train_pool = [row for row in eligible["train"] if int(stable(source_group(row), args.seed), 16) % 5 != 0]
    dev_pool = [row for row in eligible["train"] if int(stable(source_group(row), args.seed), 16) % 5 == 0]
    selected = {"train": round_robin(train_pool, args.train, args.seed),
                "dev": round_robin(dev_pool, args.dev, args.seed),
                "test": round_robin(eligible["val"], args.test, args.seed)}
    memberships = defaultdict(set)
    for split, rows in selected.items():
        for row in rows:
            memberships[source_group(row)].add(split)
    if any(len(value) > 1 for value in memberships.values()):
        raise ValueError("Original-video leakage, including neighboring excerpts")
    plan = {"version": 1, "seed": args.seed, "archive_sha256_published": ARCHIVE_SHA256,
            "source": AUTHOR_SOURCE, "clip_seconds": 2.0, "selection": "source-balanced hash order, no model scores",
            "selected": selected, "eligible_counts": {split: len(rows) for split, rows in eligible.items()},
            "source_groups": {split: len({source_group(row) for row in rows}) for split, rows in selected.items()},
            "scope": "English query-conditioned temporal retrieval; audio features do not transcribe dialogue",
            "test_policy": "Official val subset held out from all new training and tuning; official test GT is unavailable"}
    folder = ROOT / "data" / args.name
    folder.mkdir(parents=True, exist_ok=True)
    plan_path = folder / "split_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Existing frozen AV plan differs; choose another name")
    else:
        with plan_path.open("x", encoding="utf-8") as stream:
            json.dump(plan, stream, ensure_ascii=False, indent=2)
    needed = sorted({name for rows in selected.values() for row in rows for name in feature_names(row)})
    pending = sorted([by_name[name] for name in needed if not cached(by_name[name])], key=lambda entry: entry["offset"])
    groups = []
    for entry in pending:
        entry_end = entry["offset"] + 30 + len(entry["name"].encode()) + entry["compressed_size"] + 8192
        if groups:
            first, last = groups[-1][0], groups[-1][-1]
            last_end = last["offset"] + 30 + len(last["name"].encode()) + last["compressed_size"] + 8192
            if entry["offset"] - last_end <= 384 * 1024 and entry_end - first["offset"] <= 8 * 1024**2:
                groups[-1].append(entry)
                continue
        groups.append([entry])
    estimated = sum(group[-1]["offset"] + 30 + len(group[-1]["name"].encode()) + group[-1]["compressed_size"] + 8192 - group[0]["offset"] for group in groups)
    if estimated > args.max_transfer_mib * 1024**2:
        raise ValueError(f"Planned transfer exceeds limit: {estimated} bytes")
    print(json.dumps({"queries": {key: len(rows) for key, rows in selected.items()}, "source_groups": plan["source_groups"],
                      "files": len(needed), "remaining_files": len(pending), "bounded_requests": len(groups),
                      "max_transfer_mib": estimated / 1024**2}, indent=2), flush=True)
    started, transferred = time.perf_counter(), 0
    # Only whitelisted numeric feature members; no arbitrary ZIP extraction or pickle.
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_group, group): group for group in groups}
        completed = len(needed) - len(pending)
        try:
            for index, future in enumerate(as_completed(futures), 1):
                new_bytes, new_files = future.result()
                transferred += new_bytes
                completed += new_files
                if index % 25 == 0 or index == len(groups):
                    print(f"Features {completed}/{len(needed)}, transferred={transferred / 1024**2:.1f} MiB, elapsed={time.perf_counter()-started:.1f}s", flush=True)
        except Exception:
            for future in futures:
                future.cancel()
            raise
    for split, rows in selected.items():
        path = folder / f"{split}.jsonl"
        payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != payload:
            raise ValueError("Existing manifest differs")
        if not path.exists():
            with path.open("x", encoding="utf-8") as stream:
                stream.write(payload)
    report = {"source": AUTHOR_SOURCE, "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
              "files": len(needed), "bytes_transferred": transferred, "elapsed_seconds": time.perf_counter()-started,
              "integrity": "Each selected member validated with source ZIP CRC and numeric-array checks; no full archive hash claim"}
    with (folder / f"download_{datetime.now():%Y%m%d_%H%M%S_%f}.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(f"Data ready: {folder}", flush=True)


if __name__ == "__main__":
    main()
