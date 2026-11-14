"""Acquire and audit Charades-STA media without training or changing old data.

Run with the existing video_grounding_local Python environment (requests, cv2).
Large generated files stay in --root, outside the Git checkout. No model API calls.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sqlite3
import threading
import time
import zipfile
import zlib

import requests
from urllib3.exceptions import HTTPError as Urllib3HTTPError


REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = Path("E:/cutliner_data/charades_sta_v1")
MEDIA_URL = "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip"
MEDIA_SIZE = 16_339_546_533
MEDIA_SHA256 = "6e8325ffe5d9247dc89cdb507f552927836b24c8565b691dee22e3828a8a73c7"
META_URL = "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades.zip"
META_SHA256 = "c616913ef79c2ddde06d9c562eae57bb8901d459d7568a0d27bf09cbf33ae866"
VSLNET_REVISION = "3793c625f2e251a5f19a0d59f0c83b12e386f808"
VSLNET_BASE = f"https://raw.githubusercontent.com/26hzhang/VSLNet/{VSLNET_REVISION}/data/dataset/charades"
GIB = 1024**3
SEED = "cutliner-boundary-v1"
_LOCAL = threading.local()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.replace(path)


def jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    with pending.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    pending.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def immutable_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Refusing to overwrite different source snapshot: {path}")
    else:
        with path.open("xb") as stream:
            stream.write(payload)


def session(proxy: str | None = None) -> requests.Session:
    key = proxy or "direct"
    if getattr(_LOCAL, "proxy", None) != key:
        if hasattr(_LOCAL, "session"):
            _LOCAL.session.close()
        _LOCAL.session = requests.Session()
        _LOCAL.proxy = key
        if proxy:
            _LOCAL.session.proxies.update({"http": proxy, "https": proxy})
    return _LOCAL.session


def small_get(url: str, proxy: str | None, cap=8 * 1024**2) -> bytes:
    with session(proxy).get(url, timeout=(10, 30), stream=True) as response:
        response.raise_for_status()
        chunks, size, started = [], 0, time.monotonic()
        for block in response.iter_content(64 * 1024):
            size += len(block)
            if size > cap or time.monotonic() - started > 90:
                raise ValueError(f"Small source download exceeded bound: {url}")
            chunks.append(block)
    return b"".join(chunks)


def initialize(root: Path) -> None:
    root = root.resolve()
    if root == Path(root.anchor) or root == REPO.resolve():
        raise ValueError("Dataset root must be a dedicated subdirectory")
    marker = root / "dataset_identity.json"
    identity = {"name": "cutliner_charades_sta_v1", "format_version": 1}
    if root.exists() and not marker.exists() and any(root.iterdir()):
        raise ValueError("Refusing to use an existing nonempty, unmarked directory")
    root.mkdir(parents=True, exist_ok=True)
    immutable_bytes(marker, (json.dumps(identity, indent=2) + "\n").encode())
    for name in ("sources", "downloads", "videos", "audits", "manifests"):
        (root / name).mkdir(exist_ok=True)


def prepare(root: Path, proxy: str | None) -> None:
    initialize(root)
    source = root / "sources"
    metadata = source / "Charades.zip"
    if not metadata.exists():
        immutable_bytes(metadata, small_get(META_URL, proxy))
    if file_hash(metadata) != META_SHA256:
        raise ValueError("Charades metadata archive SHA256 mismatch")
    with zipfile.ZipFile(metadata) as archive:
        for name in ("license.txt", "README.txt", "Charades_v1_train.csv", "Charades_v1_test.csv",
                     "Charades_v1_classes.txt"):
            immutable_bytes(source / name, archive.read("Charades/" + name))
    sources = []
    comparison = {}
    for split in ("train", "test"):
        raw_path = source / f"charades_sta_{split}.txt"
        raw_url = f"{VSLNET_BASE}/{raw_path.name}"
        if not raw_path.exists():
            immutable_bytes(raw_path, small_get(raw_url, proxy))
        local_path = REPO / "experiments/video_grounding_local/vendor/lighthouse/data/charades" / f"charades_{split}_release.jsonl"
        snapshot = source / local_path.name
        immutable_bytes(snapshot, local_path.read_bytes())
        cleaned = read_jsonl(snapshot)
        lookup = {(x["vid"], x["query"].strip()): x for x in cleaned}
        differences, originals = [], []
        for line_number, line in enumerate(raw_path.read_text(encoding="utf-8").splitlines(), 1):
            prefix, query = line.split("##", 1)
            vid, start, end = prefix.split()
            original = {"line": line_number, "video_id": vid, "query": query.strip(),
                        "start": float(start), "end": float(end)}
            originals.append(original)
            current = lookup.get((vid, query.strip()))
            if current is None:
                differences.append({**original, "difference": "not_in_lighthouse_release"})
            elif current["relevant_windows"][0] != [float(start), float(end)]:
                differences.append({**original, "difference": "boundary_changed",
                                    "release_interval": current["relevant_windows"][0],
                                    "release_duration": current["duration"]})
        jsonl(source / f"original_{split}.jsonl", originals)
        jsonl(root / "audits" / f"annotation_differences_{split}.jsonl", differences)
        comparison[split] = {"raw_queries": len(originals), "release_queries": len(cleaned),
                             "release_videos": len({x["vid"] for x in cleaned}),
                             "differences": dict(Counter(x["difference"] for x in differences))}
        sources.extend([{"file": raw_path.name, "url": raw_url, "sha256": file_hash(raw_path),
                         "role": "pinned_research_author_mirror_of_original_STA_annotations"},
                        {"file": snapshot.name, "url": "https://github.com/line/lighthouse",
                         "sha256": file_hash(snapshot), "role": "existing_cleaned_release_snapshot"}])
    lock = {"dataset": "Charades-STA", "media_url": MEDIA_URL, "media_size": MEDIA_SIZE,
            "media_sha256_published": MEDIA_SHA256,
            "hash_reference": "https://huggingface.co/datasets/HuggingFaceM4/charades/blob/main/dataset_infos.json",
            "metadata_url": META_URL, "metadata_sha256_verified": META_SHA256,
            "annotations": sources, "comparison": comparison,
            "original_author": "https://github.com/jiyanggao/TALL",
            "author_drive_status_at_acquisition": "author train download link returned HTTP 404; using pinned VSLNet mirror, retaining both versions",
            "license": {"path": "sources/license.txt", "scope": "local non-commercial academic research",
                        "redistribution": "do not publicly upload videos, frames or derived data; inspect original terms before distributing weights",
                        "gate_or_signature_submitted": False},
            "label_policy": "Preserve raw and cleaned boundaries; primary strict-complete set excludes changed/censored boundaries. No fabricated negatives."}
    dump(root / "sources" / "source_lock.json", lock)
    print(json.dumps({"phase": "prepared_sources", **comparison}), flush=True)


def exact_range(url: str, start: int, end: int, size: int, etag: str, proxy: str | None) -> bytes:
    headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity", "If-Match": etag}
    with session(proxy).get(url, headers=headers, stream=True, timeout=(10, 40)) as response:
        response.raise_for_status()
        if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
            raise ValueError("Exact Range not honored; refusing unbounded/full fallback")
        if response.headers.get("ETag") != etag:
            raise ValueError("Remote archive changed during acquisition")
        expected = end - start + 1
        payload = response.raw.read(expected + 1)
        if len(payload) != expected:
            raise ValueError("Incomplete/oversized archive range")
    return payload


def download_archive(root: Path, proxy: str | None, workers: int, reserve_gib: float) -> Path:
    archive = root / "downloads" / "Charades_v1_480.zip"
    verified = root / "downloads" / "archive_verified.json"
    if archive.exists():
        if archive.stat().st_size != MEDIA_SIZE or file_hash(archive) != MEDIA_SHA256:
            raise ValueError("Existing media archive failed expected SHA256; keeping it unchanged")
        print("Existing archive SHA256 verified", flush=True)
        return archive
    partial = archive.with_suffix(".zip.part")
    state_path = root / "downloads" / "ranges.sqlite3"
    with session(proxy).head(MEDIA_URL, timeout=(10, 30)) as response:
        response.raise_for_status()
        if int(response.headers.get("Content-Length", 0)) != MEDIA_SIZE:
            raise ValueError("Unexpected remote archive size")
        etag = response.headers["ETag"]
    chunk_size = 4 * 1024**2
    configuration = {"url": MEDIA_URL, "size": MEDIA_SIZE, "etag": etag, "chunk_size": chunk_size}
    marker = root / "downloads" / "download_configuration.json"
    immutable_bytes(marker, json.dumps(configuration, indent=2).encode())
    if not partial.exists():
        if shutil.disk_usage(root).free < MEDIA_SIZE + reserve_gib * GIB:
            raise ValueError("Insufficient free space for archive plus reserve")
        with partial.open("xb") as stream:
            stream.truncate(MEDIA_SIZE)
    if partial.stat().st_size != MEDIA_SIZE:
        raise ValueError("Unexpected partial archive length")
    db = sqlite3.connect(state_path)
    db.execute("CREATE TABLE IF NOT EXISTS chunks (idx INTEGER PRIMARY KEY, sha256 TEXT NOT NULL, bytes INTEGER NOT NULL)")
    db.commit()
    known = {i: (sha, length) for i, sha, length in db.execute("SELECT idx,sha256,bytes FROM chunks")}
    # A database record alone is insufficient for safe resume: rehash completed bytes.
    with partial.open("rb") as stream:
        for idx, (sha, length) in list(known.items()):
            stream.seek(idx * chunk_size)
            if digest(stream.read(length)) != sha:
                db.execute("DELETE FROM chunks WHERE idx=?", (idx,))
                del known[idx]
    db.commit()
    chunks = math.ceil(MEDIA_SIZE / chunk_size)
    pending = [idx for idx in range(chunks) if idx not in known]
    completed_bytes = sum(x[1] for x in known.values())
    starting_bytes, started, last_progress = completed_bytes, time.monotonic(), 0.0
    failures = []

    def fetch(idx: int):
        start, end = idx * chunk_size, min(MEDIA_SIZE - 1, (idx + 1) * chunk_size - 1)
        for attempt in range(4):
            try:
                if shutil.disk_usage(root).free < reserve_gib * GIB:
                    raise RuntimeError("Disk reserve reached")
                payload = exact_range(MEDIA_URL, start, end, MEDIA_SIZE, etag, proxy)
                with partial.open("r+b") as stream:
                    stream.seek(start)
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                return idx, digest(payload), len(payload)
            except (requests.RequestException, Urllib3HTTPError, ValueError, OSError) as error:
                if attempt == 3:
                    raise RuntimeError(f"Range {idx} failed: {error}") from error
                time.sleep(min(2**attempt, 4))

    print(f"Downloading official archive: {len(known)}/{chunks} verified resumable ranges", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Bound queued work; no thousands of concurrent request bodies in memory.
        from concurrent.futures import wait, FIRST_COMPLETED
        iterator = iter(pending)
        active = {pool.submit(fetch, idx) for idx in [next(iterator, None) for _ in range(workers * 2)] if idx is not None}
        while active:
            ready, active = wait(active, timeout=5, return_when=FIRST_COMPLETED)
            for future in ready:
                try:
                    idx, sha, length = future.result()
                    db.execute("INSERT OR REPLACE INTO chunks VALUES (?,?,?)", (idx, sha, length))
                    db.commit()
                    completed_bytes += length
                except Exception as error:
                    failures.append(str(error))
                idx = next(iterator, None)
                if idx is not None:
                    active.add(pool.submit(fetch, idx))
            now = time.monotonic()
            if now - last_progress >= 15 or not active:
                progress = {"phase": "download", "verified_gib": round(completed_bytes / GIB, 3),
                            "total_gib": round(MEDIA_SIZE / GIB, 3),
                            "percent": round(100 * completed_bytes / MEDIA_SIZE, 2),
                            "mib_per_second": round((completed_bytes - starting_bytes) / max(now - started, 1) / 1024**2, 2),
                            "failed_ranges": len(failures)}
                dump(root / "audits" / "download_progress.json", progress)
                print(json.dumps(progress), flush=True)
                last_progress = now
    db.close()
    if failures:
        dump(root / "audits" / "download_failures.json", failures)
        raise RuntimeError(f"{len(failures)} ranges failed. Rerun to resume; successful ranges are retained.")
    print("Checking complete official archive SHA256...", flush=True)
    actual_hash = file_hash(partial)
    if actual_hash != MEDIA_SHA256:
        raise ValueError("Full archive hash mismatch; partial and resume state retained")
    partial.replace(archive)
    dump(verified, {**configuration, "sha256": actual_hash, "full_archive_hash_verified": True,
                    "status": "complete", "previous_range_failure_report_is_historical": (root / "audits/download_failures.json").exists()})
    return archive


def requested_videos(root: Path) -> set[str]:
    return {x["vid"] for split in ("train", "test")
            for x in read_jsonl(root / "sources" / f"charades_{split}_release.jsonl")}


def extract(root: Path, archive: Path, reserve_gib: float, limit: int | None = None) -> None:
    requested = requested_videos(root)
    with zipfile.ZipFile(archive) as media:
        index = {}
        for info in media.infolist():
            path = Path(info.filename)
            if path.suffix.lower() == ".mp4" and path.stem in requested:
                if path.stem in index:
                    raise ValueError(f"Duplicate archive video ID: {path.stem}")
                index[path.stem] = info
        missing = sorted(requested - index.keys())
        dump(root / "audits" / "archive_coverage.json", {
            "requested_videos": len(requested), "found_videos": len(index), "missing": missing,
            "selected_uncompressed_gib": round(sum(x.file_size for x in index.values()) / GIB, 4),
            "validation": "official complete archive SHA256 and per-member ZIP CRC"})
        if missing:
            raise ValueError(f"Missing video IDs in official archive: {missing[:20]}")
        rows = []
        for ordinal, (vid, info) in enumerate(sorted(index.items()), 1):
            if limit is not None and ordinal > limit:
                break
            dest = root / "videos" / f"{vid}.mp4"
            if not dest.exists():
                if shutil.disk_usage(root).free < info.file_size + reserve_gib * GIB:
                    raise RuntimeError("Disk reserve reached during extraction")
                pending = dest.with_suffix(".mp4.part")
                # ZipFile.read checks CRC for the entire decompressed member.
                payload = media.read(info)
                with pending.open("wb") as stream:
                    stream.write(payload)
                pending.replace(dest)
            payload = dest.read_bytes()
            if len(payload) != info.file_size or zlib.crc32(payload) & 0xFFFFFFFF != info.CRC:
                raise ValueError(f"Extracted media CRC mismatch: {vid}; preserving file for inspection")
            rows.append({"video_id": vid, "file_path": str(dest), "bytes": len(payload),
                         "sha256": digest(payload), "zip_crc32": info.CRC,
                         "archive_member": info.filename})
            if ordinal % 250 == 0:
                print(f"Extracted and CRC checked: {ordinal}/{len(index)} videos", flush=True)
        jsonl(root / "audits" / "media_integrity.jsonl", rows)
        print(f"Extracted and integrity checked {len(rows)} videos", flush=True)


def audit(root: Path, workers: int) -> None:
    import cv2

    cv2.setNumThreads(1)
    references = defaultdict(list)
    for split in ("train", "test"):
        for row in read_jsonl(root / "sources" / f"charades_{split}_release.jsonl"):
            references[row["vid"]].append(row)
    integrity = read_jsonl(root / "audits" / "media_integrity.jsonl")
    previous_path = root / "audits" / "video_audit.jsonl"
    previous = {x["video_id"]: x for x in read_jsonl(previous_path)} if previous_path.exists() else {}

    def check(record):
        cached = previous.get(record["video_id"])
        if cached and cached.get("sha256") == record["sha256"] and cached.get("status") == "ok":
            return cached
        cap = cv2.VideoCapture(record["file_path"], cv2.CAP_FFMPEG,
                               [cv2.CAP_PROP_N_THREADS, 1])
        result = {**record, "status": "rejected", "validation_scope": "CRC plus real frames at start/middle/end; not exhaustive frame decoding"}
        try:
            if not cap.isOpened():
                return {**result, "reason": "cannot_open"}
            fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not math.isfinite(fps) or fps <= 0 or count < 2:
                return {**result, "reason": "invalid_fps_or_frame_count"}
            duration = count / fps
            metadata_durations = sorted({r["duration"] for r in references[record["video_id"]]})
            # Metadata length may round; >0.25s disagreement is quarantined, not rescaled.
            if any(abs(duration - value) > 0.25 for value in metadata_durations):
                return {**result, "reason": "duration_mismatch", "actual_duration": duration,
                        "annotation_durations": metadata_durations}
            decoded = []
            for position in sorted({0, count // 2, max(0, count - 2)}):
                cap.set(cv2.CAP_PROP_POS_FRAMES, position)
                ok, frame = cap.read()
                if not ok or frame is None or not frame.size:
                    return {**result, "reason": "sample_decode_failed", "frame": position}
                decoded.append(position)
            return {**result, "status": "ok", "actual_duration": duration, "fps": fps,
                    "frame_count": count, "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "annotation_durations": metadata_durations, "decoded_frame_indices": decoded}
        except cv2.error as error:
            return {**result, "reason": "decoder_error", "detail": str(error)[:400]}
        finally:
            cap.release()

    rows, started = [], time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(check, record) for record in integrity]):
            rows.append(future.result())
            if len(rows) % 200 == 0:
                jsonl(previous_path, sorted(rows, key=lambda x: x["video_id"]))
                print(f"Decoded/audited {len(rows)}/{len(integrity)} videos ({time.monotonic() - started:.1f}s)", flush=True)
    jsonl(previous_path, sorted(rows, key=lambda x: x["video_id"]))
    print(json.dumps({"phase": "video_audit", "counts": dict(Counter(x["status"] for x in rows)),
                      "reasons": dict(Counter(x.get("reason") for x in rows if x["status"] != "ok"))}), flush=True)


def build(root: Path) -> None:
    audits = {x["video_id"]: x for x in read_jsonl(root / "audits" / "video_audit.jsonl")}
    release = {split: read_jsonl(root / "sources" / f"charades_{split}_release.jsonl") for split in ("train", "test")}
    origins = {}
    for split in ("train", "test"):
        origins[split] = {(x["video_id"], x["query"]): x for x in read_jsonl(root / "sources" / f"original_{split}.jsonl")}
    train_ids = {x["vid"] for x in release["train"]}
    test_ids = {x["vid"] for x in release["test"]}
    if train_ids & test_ids:
        raise ValueError("Official source video train/test overlap")
    train_hashes = {audits[v]["sha256"] for v in train_ids if v in audits}
    test_hashes = {audits[v]["sha256"] for v in test_ids if v in audits}
    crossing_hashes = train_hashes & test_hashes
    # Hash-identical sources stay together when splitting official training videos.
    eligible_hashes = sorted(train_hashes - crossing_hashes,
                             key=lambda x: hashlib.sha256(f"{SEED}:{x}".encode()).hexdigest())
    dev_hashes = set(eligible_hashes[:max(1, round(len(eligible_hashes) * 0.1))])
    sets = {name: [] for name in ("train", "dev", "test")}
    complete = {name: [] for name in sets}
    partial = {name: [] for name in sets}
    excluded, censored = [], []
    for original_split, rows in release.items():
        for row in rows:
            vid = row["vid"]
            key = f"charades_sta_{original_split}_{row['qid']:05d}"
            media = audits.get(vid)
            if not media or media["status"] != "ok":
                excluded.append({"case_id": key, "video_id": vid, "reason": "missing_or_rejected_media"})
                continue
            if original_split == "train" and media["sha256"] in crossing_hashes:
                excluded.append({"case_id": key, "video_id": vid, "reason": "same_content_as_official_test"})
                continue
            split = "test" if original_split == "test" else ("dev" if media["sha256"] in dev_hashes else "train")
            start, end = row["relevant_windows"][0]
            original = origins[original_split].get((vid, row["query"].strip()))
            if original is None:
                raise ValueError(f"No raw annotation counterpart for {key}")
            changed = [original["start"], original["end"]] != [start, end]
            if not (0 <= start < end <= media["actual_duration"] + 1 / media["fps"]):
                excluded.append({"case_id": key, "video_id": vid, "reason": "out_of_media_bounds"})
                continue
            sample = {"case_id": key, "dataset": "Charades-STA", "split": split,
                      "official_split": original_split, "video_id": vid,
                      "video_path": media["file_path"], "video_sha256": media["sha256"],
                      "duration_seconds": media["actual_duration"], "query": row["query"],
                      "target_interval_seconds": [start, end],
                      "original_interval_seconds": [original["start"], original["end"]],
                      "release_boundary_changed": changed,
                      "target_touches_source_edge": start <= 0.1 or end >= media["actual_duration"] - 0.1,
                      "boundary_loss_mask": {
                          "start": original["start"] == start and start > 0.1,
                          "end": original["end"] == end and end < media["actual_duration"] - 0.1},
                      "label_scope": "one annotated text-relevant interval; not exhaustive absence or action stages",
                      "annotation_precision": "source timestamps; subsecond precision not independently verified",
                      "complete_boundary_eligible": not changed and start > 0.1 and end < media["actual_duration"] - 0.1}
            sets[split].append(sample)
            if changed:
                censored.append({"case_id": key, "split": split, "video_id": vid,
                                 "original": sample["original_interval_seconds"], "release": [start, end],
                                 "reason": "cleaned_boundary_changed_possible_right_censoring"})
            elif sample["complete_boundary_eligible"]:
                complete[split].append(sample)
            else:
                censored.append({"case_id": key, "split": split, "video_id": vid,
                                 "original": sample["original_interval_seconds"], "release": [start, end],
                                 "reason": "target_touches_source_edge_completion_unverified"})
            if not sample["complete_boundary_eligible"] and any(sample["boundary_loss_mask"].values()):
                partial[split].append(sample)
    for split in sets:
        jsonl(root / "manifests" / f"{split}.release.jsonl", sets[split])
        jsonl(root / "manifests" / f"{split}.complete.jsonl", complete[split])
        jsonl(root / "manifests" / f"{split}.partial.jsonl", partial[split])
    jsonl(root / "audits" / "excluded_queries.jsonl", excluded)
    jsonl(root / "audits" / "reserved_boundary_queries.jsonl", censored)

    # Training-only augmentation: these are windows, not extra independent labels.
    windows = []
    for sample in complete["train"]:
        start, end = sample["target_interval_seconds"]
        duration = sample["duration_seconds"]
        rng = random.Random(hashlib.sha256((SEED + sample["case_id"]).encode()).hexdigest())
        for variant in range(3):
            left, right = rng.uniform(2, 7), rng.uniform(2, 7)
            lower, upper = max(0, start - left), min(duration, end + right)
            coarse_start = max(lower, min(end - 0.1, start + rng.uniform(-left * 0.8, (end - start) * 0.2)))
            coarse_end = min(upper, max(coarse_start + 0.1, end + rng.uniform(-(end - start) * 0.2, right * 0.8)))
            windows.append({**sample, "window_id": f"{sample['case_id']}_w{variant}",
                            "observation_window_seconds": [lower, upper],
                            "coarse_candidate_seconds": [coarse_start, coarse_end],
                            "coarse_candidate_origin": "synthetic_ground_truth_jitter_for_training_only",
                            "target_relative_seconds": [start - lower, end - lower],
                            "target_fully_contained": True, "planned_real_frame_cadence_seconds": 0.5,
                            "features_extracted": False})
    jsonl(root / "manifests" / "train.windows.jsonl", windows)
    groups = {s: {x["video_id"] for x in rows} for s, rows in sets.items()}
    hashes = {s: {x["video_sha256"] for x in rows} for s, rows in sets.items()}
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        if groups[a] & groups[b] or hashes[a] & hashes[b]:
            raise ValueError("Cross-split source/content leakage")
    durations = sorted(x["target_interval_seconds"][1] - x["target_interval_seconds"][0]
                       for rows in complete.values() for x in rows)
    report = {"dataset_root": str(root), "ready_for": "dense real-frame feature extraction, then positive local boundary training",
              "archive_full_sha256_verified": json.loads((root / "downloads/archive_verified.json").read_text(encoding="utf-8"))["full_archive_hash_verified"],
              "not_ready_for": ["NULL/negative training without verified absence labels", "claimed subsecond ground truth", "end-to-end refinement results"],
              "media": {"requested": len(requested_videos(root)), "audited": len(audits),
                        "valid": sum(x["status"] == "ok" for x in audits.values()),
                        "total_gib": round(sum(x["bytes"] for x in audits.values()) / GIB, 4),
                        "decode_validation": "start/middle/end actual frames, not every frame"},
              "release_splits": {s: {"queries": len(rows), "videos": len(groups[s])} for s, rows in sets.items()},
              "strict_complete_splits": {s: {"queries": len(rows), "videos": len({x['video_id'] for x in rows})}
                                         for s, rows in complete.items()},
              "partial_boundary_supervision_splits": {s: len(rows) for s, rows in partial.items()},
              "reserved_boundary_queries": len(censored),
              "reserved_boundary_reasons": dict(Counter(x["reason"] for x in censored)),
              "excluded_queries": len(excluded),
              "synthetic_train_windows": len(windows), "independent_train_queries": len(complete["train"]),
              "video_id_and_content_sha256_split_overlap": 0, "cross_official_duplicate_content_groups": len(crossing_hashes),
              "strict_events_at_most_2_seconds": sum(d <= 2 for d in durations),
              "strict_event_duration_median_seconds": durations[len(durations) // 2] if durations else None,
              "negative_pairs_generated": 0, "training_started": False, "seed": SEED}
    dump(root / "audits" / "dataset_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "download", "extract", "audit", "build", "all"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--proxy", default=None, help="Optional already-configured HTTP proxy; no credential extraction")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reserve-gib", type=float, default=20)
    parser.add_argument("--limit", type=int, help="Pilot extraction count; omit for the complete relevant-video set")
    args = parser.parse_args()
    if not 1 <= args.workers <= 16 or args.reserve_gib < 10:
        parser.error("workers must be 1..16 and disk reserve at least 10 GiB")
    root = args.root.resolve()
    if args.phase in ("prepare", "all"):
        prepare(root, args.proxy)
    else:
        initialize(root)
        if not (root / "sources" / "source_lock.json").exists():
            parser.error("Run prepare first")
    if args.phase in ("download", "all"):
        download_archive(root, args.proxy, args.workers, args.reserve_gib)
    if args.phase in ("extract", "all"):
        path = root / "downloads" / "Charades_v1_480.zip"
        if path.stat().st_size != MEDIA_SIZE or file_hash(path) != MEDIA_SHA256:
            raise ValueError("Full media archive SHA256 must match before extraction")
        extract(root, path, args.reserve_gib, args.limit)
    if args.phase in ("audit", "all"):
        audit(root, min(args.workers, 6))
    if args.phase in ("build", "all"):
        build(root)


if __name__ == "__main__":
    main()
