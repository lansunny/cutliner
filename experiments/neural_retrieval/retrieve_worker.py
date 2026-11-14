"""Local neural candidate generation. No VLM or chat API is used in this worker."""

import contextlib
import gc
import json
import math
import sys
from pathlib import Path

import imageio_ffmpeg
import torch
from torch.nn import functional as F

PROJECT = Path(__file__).resolve().parents[2]
VISUAL = PROJECT/"experiments/video_grounding_local"
AV = PROJECT/"experiments/av_grounding_local"


def metadata(video):
    reader = imageio_ffmpeg.read_frames(str(video))
    try:
        return next(reader)
    finally:
        reader.close()


def overlap(a, b):
    return max(0., min(a[1], b[1])-max(a[0], b[0]))


def qd_proposals(request, device, info, runtime=None):
    sys.path.insert(0, str(VISUAL))
    import run_local as qd
    if runtime is None:
        model, config = qd.load_checkpoint(device)
        encoder, _ = qd.clip.load(str(VISUAL/"weights/ViT-B-32.pt"), device=device, jit=False)
    else:
        model, config, encoder = runtime
    encoder.eval().requires_grad_(False)
    duration, start = request["duration_seconds"], request["start_seconds"]
    tone_map = any(tag in info.get("pix_fmt", "") for tag in ("smpte2084", "arib-std-b67", "bt2020"))
    print(f"QD-DETR query: {request['visual_query_en']}", flush=True)
    with torch.inference_mode():
        features, cache, hit = qd.extract_features(Path(request["video"]), start, duration, encoder, device,
            float(config.clip_length), 4, tone_map, "none")
        # Preserve the query instead of silently truncating to 32 tokens.
        tokens = qd.clip.tokenize(request["visual_query_en"], max_valid_length=100000).to(device)
        length = int(tokens[0].argmax()) + 1
        text = F.normalize(encoder.encode_text(tokens)["last_hidden_state"][:, :length].float(), dim=-1, eps=1e-5)
        proposals, windows = qd.locate(model, features, text, torch.ones(1, length, device=device), config, start, duration, device)
    result = []
    for item in proposals[:request["top_k"]]:
        lo, hi = max(start, item["start"]), min(start+duration, item["end"])
        if lo < hi:
            evidence = {"modality": "visual", "model_name": "QD-DETR/CLIP QVHighlights", "label": request["visual_query_en"],
                        "start_seconds": lo, "end_seconds": hi, "score": item["score"], "window_index": item["window"]}
            result.append({"start_seconds": lo, "end_seconds": hi, "score": item["score"], "model_evidence": [evidence]})
    del features, text
    if runtime is None:
        del model, encoder
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    return result, {"feature_cache": cache, "feature_cache_hit": hit, "feature_windows": windows,
                    "sample_interval_seconds": float(config.clip_length), "tone_map": tone_map,
                    "limitations": "Independent-frame CLIP and learned localization; uncalibrated foreground/window scores; not verified event existence"}


def event_windows(start, end, source_duration):
    if source_duration < 9.5:
        raise ValueError("AV model needs approximately ten seconds of observed source context")
    if source_duration < 10:
        return [0.]
    first = max(0., min(source_duration-10, start-1))
    last = min(source_duration-10, max(first, end-9))
    starts = [first]
    while starts[-1]+8 < last-1e-6:
        starts.append(starts[-1]+8)
    if last > starts[-1]+1e-6:
        starts.append(last)
    return starts


def merge_events(events):
    merged = []
    for item in sorted(events, key=lambda e: (e["label"], e["start_seconds"])):
        if merged and merged[-1]["label"] == item["label"] and item["start_seconds"] <= merged[-1]["end_seconds"]+.05:
            previous = merged[-1]
            previous["end_seconds"] = max(previous["end_seconds"], item["end_seconds"])
            previous["score"] = max(previous["score"], item["score"])
            previous["window_index"] = None  # The merged range is supported by more than one context.
        else:
            merged.append(dict(item))
    return merged


def av_proposals(request, device, info):
    sys.path.insert(0, str(AV))
    from predict_events import EventRuntime, decode_window
    from event_dataset import contiguous
    runtime = EventRuntime(Path(request["av_run"]), device)
    start, end = request["start_seconds"], request["start_seconds"]+request["duration_seconds"]
    windows = event_windows(start, end, float(info["duration"]))
    events = []
    hdr = any(tag in info.get("pix_fmt", "") for tag in ("smpte2084", "arib-std-b67", "bt2020"))
    if hdr:
        print("Warning: AV benchmark covers SDR; HDR-domain quality remains unvalidated", flush=True)
    for index, position in enumerate(windows):
        probability = runtime.probabilities(decode_window(Path(request["video"]), position))
        for name in request["audio_event_classes"]:
            values = probability[:, runtime.classes.index(name)]
            for a, b in contiguous(values >= runtime.threshold):
                lo, hi = max(start, position+a), min(end, position+b)
                if lo < hi:
                    events.append({"modality": "audio_visual", "model_name": f"AV-event/{runtime.mode}/seed{runtime.seed}",
                                   "label": name, "start_seconds": lo, "end_seconds": hi,
                                   "score": float(values[a:b].mean()), "window_index": index})
        if (index+1)%10 == 0 or index+1 == len(windows):
            print(f"AV event windows {index+1}/{len(windows)}", flush=True)
    events = merge_events(events)
    classes = request["audio_event_classes"]
    if request["audio_match_mode"] == "any" or len(classes) == 1:
        groups = [[event] for event in events]
    else:
        by_class = {name: [event for event in events if event["label"] == name] for name in classes}
        groups, seen = [], set()
        if all(by_class.values()):
            for anchor in events:
                chosen = [min(by_class[name], key=lambda e: (max(0., e["start_seconds"]-anchor["end_seconds"],
                                  anchor["start_seconds"]-e["end_seconds"]), -e["score"])) for name in classes]
                key = tuple((e["label"], e["start_seconds"], e["end_seconds"]) for e in chosen)
                if key not in seen:
                    groups.append(chosen)
                    seen.add(key)
    result = []
    for group in groups:
        lo = max(start, min(e["start_seconds"] for e in group)-request["padding_seconds"])
        hi = min(end, max(e["end_seconds"] for e in group)+request["padding_seconds"])
        result.append({"start_seconds": lo, "end_seconds": hi, "score": min(e["score"] for e in group),
                       "model_evidence": group})
    return result, {"window_count": len(windows), "window_seconds": 10, "stride_seconds": 8,
                    "timestamp_resolution_seconds": 1, "threshold": runtime.threshold,
                    "event_class_count": 28, "audio_match_mode": request["audio_match_mode"],
                    "hdr_domain_unvalidated": hdr, "limitations": "Sounding-source events, not dialogue content, identity, or full-query certification"}


def retrieve(request, *, visual_runtime=None, source_info=None):
    torch.set_num_threads(3)
    device = "cuda" if request.get("device", "auto") == "auto" and torch.cuda.is_available() else request.get("device", "cpu")
    if device == "auto":
        device = "cpu"
    if request["duration_seconds"] <= 0 or request["top_k"] < 1:
        raise ValueError("Invalid interval or candidate budget")
    info = metadata(request["video"]) if source_info is None else source_info
    visual, av, details = [], [], {"device": device, "vlm_calls": 0, "chat_api_calls": 0}
    if request["route"] in ("visual", "hybrid"):
        visual, details["visual"] = qd_proposals(request, device, info, visual_runtime)
    if request["route"] in ("audio_visual", "hybrid"):
        av, details["audio_visual"] = av_proposals(request, device, info)
    proposals = visual if request["route"] == "visual" else av
    if request["route"] == "hybrid":
        for item in proposals:
            supports = [v for v in visual if overlap((item["start_seconds"], item["end_seconds"]),
                                                    (v["start_seconds"], v["end_seconds"])) > 0]
            if supports:
                best = max(supports, key=lambda v: v["score"])
                item["model_evidence"] += best["model_evidence"]
                item["score"] = math.sqrt(item["score"]*best["score"])
            else:
                item["retrieval_notes"] = ["No overlapping QD-DETR support; retained for Agent review, not silently excluded"]
                item["score"] *= .5
    selected = []
    for item in sorted(proposals, key=lambda p: (-p["score"], p["start_seconds"])):
        if not all(math.isfinite(item[key]) for key in ("start_seconds", "end_seconds", "score")):
            raise ValueError("Non-finite neural candidate")
        if any(overlap((item["start_seconds"], item["end_seconds"]), (old["start_seconds"], old["end_seconds"])) /
               max(1e-8, max(item["end_seconds"], old["end_seconds"])-min(item["start_seconds"], old["start_seconds"])) >= .8 for old in selected):
            continue
        selected.append(item)
        if len(selected) == request["top_k"]:
            break
    return {"status": "ok", "route": request["route"], "predictions": selected, "details": details,
            "certifies_full_query": False, "no_match_is_proof_of_absence": False}


if __name__ == "__main__":
    request = json.loads(sys.stdin.readline())
    with contextlib.redirect_stdout(sys.stderr):
        result = retrieve(request)
    print(json.dumps(result, ensure_ascii=False))
