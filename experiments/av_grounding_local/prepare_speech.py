"""Prepare independent speech/vision grounding data, without treating ASR as truth."""

import hashlib
import io
import json
import re
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import torch

from public_zip import PublicRangeReader, ROOT, fetch_members

DATA = ROOT / "data/speech_v1"
BASE = "https://hf-mirror.com/j-min/HiREST-baseline/resolve/main/"
SOURCES = {
    "visual": {"file": "eva_clip_features.zip", "size": 3705363457,
               "sha256": "a02b84ae016258f6cd0a44a785f2c47388d46275f48ed801c154a71566f297e7"},
    "speech": {"file": "ASR_feats_all-MiniLM-L6-v2.zip", "size": 256848863,
               "sha256": "2b6540d4f7c066b35cbdb917c3a0cbbb146690b2efb15e6fd054d592f312fdc4"},
}
LOCAL = threading.local()


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def annotation(split):
    path = DATA / f"official_{split}.json"
    if not path.exists():
        response = requests.get(f"https://raw.githubusercontent.com/j-min/HiREST/main/data/splits/all_data_{split}.json", timeout=(10, 30))
        response.raise_for_status()
        json.loads(response.content)
        with path.open("xb") as stream:
            stream.write(response.content)
    return json.loads(path.read_text(encoding="utf-8"))


def catalog(kind):
    path = DATA / f"{kind}_zip_index.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["entries"]
    source = SOURCES[kind]
    with PublicRangeReader(BASE + source["file"], source["size"], 64*1024**2) as reader, zipfile.ZipFile(reader) as archive:
        entries = {info.filename: {"name": info.filename, "offset": info.header_offset, "compressed_size": info.compress_size,
                                   "size": info.file_size, "crc": info.CRC} for info in archive.infolist() if not info.is_dir()}
    with path.open("x", encoding="utf-8") as stream:
        json.dump({"source": source, "full_archive_hash_verified": False,
                   "integrity": "Individual source ZIP member CRC, not entire archive hash", "entries": entries}, stream)
    return entries


def feature_name(kind, video):
    video_id = video.removesuffix(".mp4")
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("Unexpected public video identifier")
    return f"eva_clip_features/{video}.pt" if kind == "visual" else f"ASR_feats_all-MiniLM-L6-v2/{video_id}.pt"


def ensure_feature(kind, entry):
    path = DATA / "features" / entry["name"]
    if path.exists():
        import zlib
        payload = path.read_bytes()
        if len(payload) != entry["size"] or zlib.crc32(payload) & 0xffffffff != entry["crc"]:
            raise ValueError("Existing speech/vision feature CRC differs")
        return
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
    source = SOURCES[kind]
    for attempt in range(3):
        try:
            payloads, _ = fetch_members([entry], LOCAL.session, url=BASE + source["file"], archive_size=source["size"], max_range_bytes=32*1024**2)
            break
        except requests.RequestException as error:
            status = error.response.status_code if error.response is not None else None
            if attempt == 2 or status is not None and status < 500:
                raise
            time.sleep(1 + 2*attempt)
    payload = payloads[entry["name"]]
    value = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    expected_dim = 1024 if kind == "visual" else 384
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != expected_dim or not value.isfinite().all():
        raise ValueError("Unexpected safely-loaded source feature")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def prepare_transcripts(videos):
    path = DATA / "ASR.zip"
    expected = "0b452d38e30064dc7273a58b7b73ec33e307ff83d30048a472777f56e3a29fbc"
    if not path.exists():
        response = requests.get(BASE + "ASR.zip", timeout=(10, 30))
        response.raise_for_status()
        if len(response.content) != 7160738 or digest(response.content) != expected:
            raise ValueError("Author transcript archive hash mismatch")
        with path.open("xb") as stream:
            stream.write(response.content)
    if digest(path.read_bytes()) != expected:
        raise ValueError("Existing transcript archive differs")
    with zipfile.ZipFile(path) as archive:
        for video in videos:
            name = f"ASR/{video.removesuffix('.mp4')}.srt"
            payload = archive.read(name)
            payload.decode("utf-8-sig")
            target = DATA / name
            target.parent.mkdir(exist_ok=True)
            if target.exists() and target.read_bytes() != payload:
                raise ValueError("Existing transcript differs")
            if not target.exists():
                with target.open("xb") as stream:
                    stream.write(payload)


def seconds(value):
    h, m, s = value.replace(",", ".").split(":")
    return 3600*int(h) + 60*int(m) + float(s)


def srt_intervals(text):
    return [(seconds(start), seconds(end)) for start, end in re.findall(
        r"(\d+:\d\d:\d\d[,.]\d+)\s*-->\s*(\d+:\d\d:\d\d[,.]\d+)", text)]


def main():
    torch.set_num_threads(1)
    DATA.mkdir(parents=True, exist_ok=True)
    annotations = {split: annotation(split) for split in ("train", "val")}
    indexes = {kind: catalog(kind) for kind in SOURCES}
    videos = {}
    for split, prompts in annotations.items():
        for prompt, pairs in prompts.items():
            for video, item in pairs.items():
                if not item.get("relevant") or not item.get("clip") or not item.get("steps"):
                    continue
                if not all(feature_name(kind, video) in indexes[kind] for kind in SOURCES):
                    raise ValueError(f"Missing required public feature for annotated video {video}")
                if video in videos:
                    raise ValueError("Duplicate step-annotated video; inspect ambiguity before sampling")
                videos[video] = {"official_split": split, "task_context": prompt, **item}
    train_ids = sorted([video for video, item in videos.items() if item["official_split"] == "train"],
                       key=lambda video: digest(f"speech_v1:20261107:{video}".encode()))
    if len(train_ids) < 480:
        raise ValueError("Insufficient original videos for the predeclared 400/80 source split")
    chosen = {"train": train_ids[:400], "dev": train_ids[400:480],
              "test": sorted(video for video, item in videos.items() if item["official_split"] == "val")}
    plan = {"version": 1, "sources": SOURCES, "selected": chosen,
            "annotation_sha256": {split: digest((DATA / f"official_{split}.json").read_bytes()) for split in annotations},
            "task": "Derived step-caption-as-query temporal grounding, not HiREST official step-caption generation",
            "supervision": "Human step time intervals, never ASR timestamps as ground-truth action boundaries",
            "evaluation": "Official val videos held out from all new model training/tuning; official test step labels are not public",
            "scope": "Instructional videos; not proof of film-domain, Chinese, or no-target-query performance"}
    plan_path = DATA / "split_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Speech source split changed")
    else:
        with plan_path.open("x", encoding="utf-8") as stream:
            json.dump(plan, stream, indent=2)
    selected_videos = [video for ids in chosen.values() for video in ids]
    print(f"Frozen speech/vision source split: { {s: len(ids) for s, ids in chosen.items()} }", flush=True)
    prepare_transcripts(selected_videos)
    jobs = [(kind, indexes[kind][feature_name(kind, video)]) for video in selected_videos for kind in SOURCES]
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(ensure_feature, kind, entry) for kind, entry in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            if i % 50 == 0 or i == len(jobs):
                print(f"Speech/vision features {i}/{len(jobs)}", flush=True)
    manifests = {}
    audit = {"zero_duration_annotations_not_used_as_interval_truth": [], "rounded_end_clamps": [],
             "policy": "Preserve source video splits; point labels have undefined interval IoU, are not negatives, and are reported separately"}
    for split, ids in chosen.items():
        rows = []
        for video in ids:
            item = videos[video]
            speech = torch.load(DATA / "features" / feature_name("speech", video), map_location="cpu", weights_only=True)
            intervals = srt_intervals((DATA / f"ASR/{video.removesuffix('.mp4')}.srt").read_text(encoding="utf-8-sig"))
            if len(intervals) != len(speech):
                raise ValueError(f"Sentence-vector/timestamp count mismatch for {video}")
            by_query = {}
            for step in item["steps"]:
                query = step["heading"].strip()
                start, end = [float(value) for value in step["absolute_bounds"]]
                if query and 0 <= start == end <= float(item["v_duration"]) + 1.0:
                    audit["zero_duration_annotations_not_used_as_interval_truth"].append(
                        {"video": video, "split": split, "query": query, "timestamp": start})
                    continue
                if not query or not 0 <= start < end <= float(item["v_duration"]) + 1.0:
                    raise ValueError(f"Unusable step annotation: {video}, {query}, {start}, {end}")
                if end > float(item["v_duration"]):
                    audit["rounded_end_clamps"].append({"video": video, "query": query, "original_end": end,
                                                       "metadata_end": float(item["v_duration"])})
                by_query.setdefault(query, []).append([start, min(end, float(item["v_duration"]))])
            for query, windows in by_query.items():
                rows.append({"qid": digest(f"{video}:{query}".encode())[:20], "video": video, "query": query,
                             "task_context": item["task_context"], "duration": float(item["v_duration"]),
                             "relevant_windows": windows, "source_group": video, "split": split,
                             "annotation_source": "HiREST human step caption and absolute_bounds"})
        manifests[split] = rows
        payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
        path = DATA / f"{split}.jsonl"
        if path.exists() and path.read_text(encoding="utf-8") != payload:
            raise ValueError("Existing speech query manifest differs")
        if not path.exists():
            with path.open("x", encoding="utf-8") as stream:
                stream.write(payload)
    audit_path = DATA / "annotation_audit.json"
    if audit_path.exists():
        if json.loads(audit_path.read_text(encoding="utf-8")) != audit:
            raise ValueError("Speech annotation audit changed")
    else:
        with audit_path.open("x", encoding="utf-8") as stream:
            json.dump(audit, stream, ensure_ascii=False, indent=2)
    print(f"Speech/vision data ready: { {split:len(rows) for split, rows in manifests.items()} }", flush=True)
    print(f"Point annotations retained separately: {len(audit['zero_duration_annotations_not_used_as_interval_truth'])}", flush=True)


if __name__ == "__main__":
    main()
