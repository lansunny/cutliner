"""Author-verified AVE event metadata and source-disjoint, class-capped development data."""

import argparse
import hashlib
import io
import json
import re
import threading
import time
import zipfile
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import h5py
import numpy as np
import requests

from public_zip import ROOT, PublicRangeReader, fetch_members

DATA = ROOT/"data/ave_events_v1"
AUTHOR = "https://raw.githubusercontent.com/YapengTian/AVE-ECCV18/master/data/"
REPO = "UnFaZeD07/AVE-Dataset"
ARCHIVE_URL = f"https://hf-mirror.com/datasets/{REPO}/resolve/main/videos.zip"
ARCHIVE_SIZE = 5665225063
LOCAL = threading.local()


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def author_file(name):
    target = DATA/"metadata"/name
    if not target.exists():
        response = requests.get(AUTHOR+name, timeout=(10, 30))
        response.raise_for_status()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(response.content)
    return target.read_bytes()


def array_from_h5(payload):
    with h5py.File(io.BytesIO(payload), "r") as handle:
        if len(handle.keys()) != 1:
            raise ValueError("Ambiguous author HDF5 data")
        return handle[next(iter(handle.keys()))][:]


def index():
    path = DATA/"video_zip_index.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["entries"]
    response = requests.get(f"https://hf-mirror.com/api/datasets/{REPO}/tree/main", timeout=(10, 30))
    response.raise_for_status()
    item = next(item for item in response.json() if item["path"] == "videos.zip")
    if item["size"] != ARCHIVE_SIZE:
        raise ValueError("Public AVE archive size changed")
    with PublicRangeReader(ARCHIVE_URL, ARCHIVE_SIZE, 32*1024**2) as reader, zipfile.ZipFile(reader) as archive:
        entries = [{"name": info.filename, "offset": info.header_offset, "size": info.file_size,
                    "compressed_size": info.compress_size, "crc": info.CRC}
                   for info in archive.infolist() if not info.is_dir()]
    with path.open("x", encoding="utf-8") as stream:
        json.dump({"repository": REPO, "archive_metadata": item, "full_archive_hash_verified": False,
                   "integrity": "Each selected public ZIP member size/CRC; not full archive hash", "entries": entries}, stream)
    return entries


def prepare():
    DATA.mkdir(parents=True, exist_ok=True)
    annotation = author_file("Annotations.txt")
    rows = []
    for i, line in enumerate(annotation.decode("utf-8-sig").splitlines()):
        category, video, quality, start, end = line.strip().split("&")
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video):
            raise ValueError("Unexpected original video source identifier")
        if quality != "good" or not 0 <= int(start) <= int(end) <= 10:
            raise ValueError("Unexpected author event annotation")
        rows.append({"index": i, "video": video, "category": category, "start": int(start), "end": int(end)})
    labels = array_from_h5(author_file("labels.h5"))
    if labels.shape != (len(rows), 10, 29) or not np.isfinite(labels).all() or not np.all(labels.sum(-1) == 1):
        raise ValueError("Author frame labels do not match AVE metadata")
    classes = {}
    for row in rows:
        value = labels[row["index"]].argmax(-1)
        if row["start"] < row["end"]:
            unique = np.unique(value[row["start"]:row["end"]])
            if len(unique) != 1 or unique[0] == 28:
                raise ValueError("Author class interval differs from HDF5 truth")
            target = int(unique[0])
            if row["category"] in classes and classes[row["category"]] != target:
                raise ValueError("Category mapping is not stable")
            classes[row["category"]] = target
    for row in rows:
        target = classes[row["category"]]
        expected = np.full(10, 28)
        expected[row["start"]:row["end"]] = target
        if not np.array_equal(labels[row["index"]].argmax(-1), expected):
            raise ValueError(f"Text and HDF5 event bounds differ for {row['video']}")
        row["class_id"], row["labels"] = target, expected.tolist()
    if set(classes.values()) != set(range(28)):
        raise ValueError("Not all 28 event classes are represented")
    entries = index()
    videos = {}
    for entry in entries:
        filename = entry["name"].rsplit("/", 1)[-1]
        if filename.endswith(".mp4"):
            key = filename[:-4]
            if key in videos:
                raise ValueError("Ambiguous public video member")
            videos[key] = entry
    canonical, groups = {}, {}
    assigned_indices = set()
    for split, order_file in (("train", "train_order.h5"), ("dev", "val_order.h5"), ("test", "test_order.h5")):
        order = array_from_h5(author_file(order_file)).reshape(-1)
        if not np.issubdtype(order.dtype, np.integer) or not np.all((order >= 0) & (order < len(rows))):
            raise ValueError("Invalid official source split")
        canonical[split] = len(order)
        for i in order:
            if int(i) in assigned_indices:
                raise ValueError("Canonical row-index split overlaps")
            assigned_indices.add(int(i))
            row = rows[int(i)]
            if row["video"] not in videos:
                raise ValueError(f"Author-annotated video absent from mirror: {row['video']}")
            groups.setdefault(row["video"], []).append({**row, "canonical_split": split})
    if len(assigned_indices) != len(rows):
        raise ValueError("Author split does not cover all annotation rows")
    grouped = []
    for video, annotations in groups.items():
        # Multiple annotated categories on identical media are retained as a
        # multi-label target, not arbitrarily forced into one exclusive class.
        target = np.zeros((10, 28), np.uint8)
        for item in annotations:
            for t, label in enumerate(item["labels"]):
                if label != 28:
                    target[t, label] = 1
        # A source occurring in multiple canonical splits is held out entirely:
        # test > dev > train. Its training copy is never allowed to leak.
        split = max((item["canonical_split"] for item in annotations), key={"train": 0, "dev": 1, "test": 2}.get)
        grouped.append({"video": video, "class_ids": sorted({item["class_id"] for item in annotations}),
                        "labels": target.tolist(), "split": split, "zip_member": videos[video]["name"],
                        "annotation_indices": [item["index"] for item in annotations]})
    selected = {}
    for split, cap in (("train", 32), ("dev", 8), ("test", 8)):
        by_class, chosen = {}, {}
        for row in grouped:
            if row["split"] == split:
                for label in row["class_ids"]:
                    by_class.setdefault(label, []).append(row)
        for label, group in sorted(by_class.items()):
            for row in sorted(group, key=lambda row: digest(f"ave_events_v1:{row['video']}".encode()))[:cap]:
                chosen[row["video"]] = row
        selected[split] = sorted(chosen.values(), key=lambda row: row["video"])
    duplicate_audit = {video: annotations for video, annotations in groups.items() if len(annotations) > 1}
    plan = {"version": 1, "annotation_source": AUTHOR, "media_mirror": REPO,
            "canonical_split_sizes": canonical, "classes": classes,
            "group_cleaned_split_sizes": dict(Counter(row["split"] for row in grouped)),
            "duplicate_source_annotation_audit": duplicate_audit,
            "zero_duration_annotation_audit": [row for row in rows if row["start"] == row["end"]],
            "selection": "Group-cleaned author splits with test>dev>train source priority, hash caps train32/dev8/test8; derived subset, not full benchmark",
            "task": "28-class multi-label second-level audible-and-visible event localization; auxiliary Agent tool, not arbitrary query grounding",
            "scope": "Does not validate speech reasoning, long-video retrieval, or original film-domain queries",
            "source_hashes": {name: digest(author_file(name)) for name in ("Annotations.txt", "labels.h5", "train_order.h5", "val_order.h5", "test_order.h5")},
            "selected": selected}
    path = DATA/"split_plan.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Frozen AVE selection changed")
    else:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(plan, stream, indent=2)
    print(f"Verified author AVE labels and source splits: {canonical}", flush=True)
    print(f"Frozen class-capped subset: { {s:len(v) for s,v in selected.items()} }", flush=True)
    print(f"ZIP members: {len(entries)}; selected raw video bytes: {sum(videos[r['video']]['size'] for group in selected.values() for r in group)/1e9:.3f} GB", flush=True)
    return plan, videos


def ensure_video(row, entry):
    target = DATA/"videos"/f"{row['video']}.mp4"
    if target.exists():
        payload = target.read_bytes()
        if len(payload) != entry["size"] or zlib.crc32(payload) & 0xffffffff != entry["crc"]:
            raise ValueError("Existing AVE video failed source CRC")
        return
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
    for attempt in range(3):
        try:
            payloads, _ = fetch_members([entry], LOCAL.session, url=ARCHIVE_URL, archive_size=ARCHIVE_SIZE,
                                        max_range_bytes=64*1024**2)
            break
        except requests.RequestException as error:
            status = error.response.status_code if error.response is not None else None
            if attempt == 2 or status is not None and status < 500:
                raise
            time.sleep(1+attempt)
    target.parent.mkdir(exist_ok=True)
    with target.open("xb") as stream:
        stream.write(payloads[entry["name"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-splits", nargs="*", choices=["train", "dev"], default=[])
    parser.add_argument("--limit", type=int, help="Download only this many frozen entries per requested split; safe to resume")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    plan, videos = prepare()
    jobs = [row for split in args.download_splits for row in plan["selected"][split][:args.limit]]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(ensure_video, row, videos[row["video"]]) for row in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            if i%25 == 0 or i == len(jobs):
                print(f"Verified AVE development media: {i}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
