"""Local neural candidate generation. No VLM or chat API is used in this worker."""

import contextlib
import gc
import json
import math
import sys
from copy import copy
from pathlib import Path

import imageio_ffmpeg
import torch
from torch.nn import functional as F

PROJECT = Path(__file__).resolve().parents[2]
VISUAL = PROJECT/"experiments/video_grounding_local"
AV = PROJECT/"experiments/av_grounding_local"

sys.path.insert(0, str(PROJECT))
from mini_agent.video.candidate_pool import finalize_refined_pool, select_distinct


def metadata(video):
    reader = imageio_ffmpeg.read_frames(str(video))
    try:
        return next(reader)
    finally:
        reader.close()


def overlap(a, b):
    return max(0., min(a[1], b[1])-max(a[0], b[0]))


def load_visual_runtime(device):
    sys.path.insert(0, str(VISUAL))
    import run_local as qd
    model, config = qd.load_checkpoint(device)
    encoder, _ = qd.clip.load(str(VISUAL/"weights/ViT-B-32.pt"), device=device, jit=False)
    return model, config, encoder.eval().requires_grad_(False)


def rank_visual_features(qd, runtime, features, request, device, *, rank_all=False):
    """Shared callable for production and controlled offline comparisons."""
    from visual_rank import mix_scales, rerank

    model, config, encoder = runtime
    duration, start = request["duration_seconds"], request["start_seconds"]
    mode = request.get("visual_ranking", "legacy")
    if mode not in {"legacy", "frame", "consensus"}:
        raise ValueError("Unknown visual ranking mode")
    original = request["visual_query_en"]
    queries = [original]
    if mode == "consensus":
        for hint in request.get("visual_search_hints_en", [])[:2]:
            if isinstance(hint, str) and hint.strip() and hint.strip().casefold() not in [x.casefold() for x in queries]:
                queries.append(hint.strip())
    tracks, anchors, short_track = [], [], None
    short_windows = 0
    window_batch = request.get("window_batch", 16 if device == "cuda" else 1)
    use_short = (request.get("multi_context", False) and mode != "legacy" and device == "cuda"
                 and duration >= 150 and request["top_k"] >= 3)

    def locate(text, length, cfg, stride=None):
        mask = torch.ones(1, length, device=device)
        try:
            return qd.locate(model, features, text, mask, cfg, start, duration, device,
                             window_batch=window_batch, window_stride=stride)
        except torch.cuda.OutOfMemoryError:
            if window_batch == 1:
                raise
        torch.cuda.empty_cache()
        return qd.locate(model, features, text, mask, cfg, start, duration, device, window_stride=stride)

    with torch.inference_mode():
        for i, query in enumerate(queries):
            tokens = qd.clip.tokenize(query, max_valid_length=100000).to(device)
            length = int(tokens[0].argmax()) + 1
            encoded = encoder.encode_text(tokens)
            text = F.normalize(encoded["last_hidden_state"][:, :length].float(), dim=-1, eps=1e-5)
            proposals, windows = locate(text, length, config, request.get("window_stride"))
            tracks.append(sorted(proposals, key=lambda item: (-item["score"], item["start"])))
            anchors.append(F.normalize(encoded["pooler_output"].float(), dim=-1).cpu())
            if i == 0 and use_short:
                short_config = copy(config)
                short_config.max_v_l = 24
                # 48 seconds of context with a 12-second overlap. Long proposals
                # still cover the less frequent events spanning more time.
                nearby, short_windows = locate(text, length, short_config, 18)
                short_track = sorted(nearby, key=lambda item: (-item["score"], item["start"]))
        if mode == "legacy":
            return tracks[0], windows, {"mode": mode, "queries": queries}
        visual = F.normalize(features.float(), dim=-1)
        similarity = (visual @ torch.cat(anchors).T).numpy()
    rows, pool_size = rerank(tracks, similarity, float(config.clip_length), start, request["top_k"],
                             mode=mode, rank_all=rank_all)
    if short_track:
        short_rows, short_pool = rerank([short_track], similarity[:, :1], float(config.clip_length), start,
                                       request["top_k"], mode="frame", rank_all=rank_all)
        pool_size += short_pool
        rows = mix_scales(rows, [{**row, "short_context": True} for row in short_rows],
                          None if rank_all else request["top_k"])
    return rows, windows, {"mode": mode, "queries": queries, "pooled_candidates": pool_size,
                            "short_context_windows": short_windows, "long_context_windows_per_query": windows}


def qd_proposals(request, device, info, runtime=None):
    sys.path.insert(0, str(VISUAL))
    import run_local as qd
    if runtime is None:
        model, config, encoder = load_visual_runtime(device)
    else:
        model, config, encoder = runtime
    encoder.eval().requires_grad_(False)
    duration, start = request["duration_seconds"], request["start_seconds"]
    tone_map = any(tag in info.get("pix_fmt", "") for tag in ("smpte2084", "arib-std-b67", "bt2020"))
    print(f"QD-DETR query: {request['visual_query_en']}", flush=True)
    decode_details = {}
    with torch.inference_mode():
        features, cache, hit = qd.extract_features(Path(request["video"]), start, duration, encoder, device,
            float(config.clip_length), 16 if device == "cuda" and duration >= 60 else 4, tone_map, "auto",
            pixel_format=info.get("pix_fmt"), diagnostics=decode_details, frame_view=request.get("frame_view", "fit"))
        # Visual candidates share a budget only after the tracks and window scales
        # have been combined. Hybrid still uses visual rows just as support.
        rank_all = request["route"] == "visual"
        proposals, windows, ranking = rank_visual_features(
            qd, (model, config, encoder), features, request, device, rank_all=rank_all)
    result = []
    for priority, item in enumerate(proposals if rank_all else proposals[:request["top_k"]], 1):
        lo, hi = max(start, item["start"]), min(start+duration, item["end"])
        if lo < hi:
            evidence = {"modality": "visual", "model_name": "QD-DETR/CLIP QVHighlights",
                        "label": ranking["queries"][item.get("track", 0)],
                        "start_seconds": lo, "end_seconds": hi, "score": item["score"], "window_index": item["window"]}
            row = {"start_seconds": lo, "end_seconds": hi, "score": item["score"], "model_evidence": [evidence]}
            if ranking["mode"] != "legacy":
                row["review_priority"] = priority
                row["retrieval_notes"] = ["本地候选重排；线索只用于召回，不代表原始条件已经核实"]
                if item.get("short_context"):
                    row["short_context"] = True
                    row["retrieval_notes"].append("较短时间窗补充的候选，用于召回局部动作")
            result.append(row)
    del features
    if runtime is None:
        del model, encoder
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    return result, {"feature_cache": cache, "feature_cache_hit": hit, "feature_windows": windows,
                    "sample_interval_seconds": float(config.clip_length), "tone_map": tone_map, "decode": decode_details,
                    "ranking": ranking,
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


def av_proposals(request, device, info, runtime=None):
    sys.path.insert(0, str(AV))
    from predict_events import EventRuntime, decode_window
    from event_dataset import contiguous
    frame_view = request.get("frame_view", "fit")
    if runtime is None:
        runtime = EventRuntime(Path(request["av_run"]), device, frame_view=frame_view)
    elif frame_view != runtime.frame_view:
        raise ValueError("Requested AV frame geometry differs from the loaded head")
    start, end = request["start_seconds"], request["start_seconds"]+request["duration_seconds"]
    windows = event_windows(start, end, float(info["duration"]))
    events = []
    hdr = any(tag in info.get("pix_fmt", "") for tag in ("smpte2084", "arib-std-b67", "bt2020"))
    if hdr:
        print("Warning: AV benchmark covers SDR; HDR-domain quality remains unvalidated", flush=True)
    for index, position in enumerate(windows):
        probability = runtime.probabilities(decode_window(Path(request["video"]), position, frame_view=frame_view))
        for name in request["audio_event_classes"]:
            values = probability[:, runtime.classes.index(name)]
            for a, b in contiguous(values >= runtime.threshold):
                lo, hi = max(start, position+a), min(end, position+b)
                if lo < hi:
                    events.append({"modality": "audio_visual", "model_name": f"AV-event/{runtime.mode}/seed{runtime.seed}/{runtime.model_profile}",
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
                    "frame_view": frame_view, "model_profile": runtime.model_profile,
                    "timestamp_resolution_seconds": 1, "threshold": runtime.threshold,
                    "event_class_count": 28, "audio_match_mode": request["audio_match_mode"],
                    "hdr_domain_unvalidated": hdr, "limitations": "Sounding-source events, not dialogue content, identity, or full-query certification"}


def retrieve(request, *, visual_runtime=None, av_runtime=None, source_info=None, retain_reserve=False):
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
        av, details["audio_visual"] = av_proposals(request, device, info, av_runtime)
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
    order = (lambda p: (p.get("review_priority", request["top_k"] + 1), -p["score"], p["start_seconds"]))
    if request["route"] != "visual":
        order = lambda p: (-p["score"], p["start_seconds"])
    def ranked_candidates():
        for item in sorted(proposals, key=order):
            if not all(math.isfinite(item[key]) for key in ("start_seconds", "end_seconds", "score")):
                raise ValueError("Non-finite neural candidate")
            yield item

    ranked = ranked_candidates()
    selected = select_distinct(ranked, request["top_k"])
    result = {"status": "ok", "route": request["route"], "predictions": selected, "details": details,
              "certifies_full_query": False, "no_match_is_proof_of_absence": False}
    if request["route"] == "visual" and (retain_reserve or request.get("retain_candidate_reserve")):
        result["candidate_reserve"] = list(ranked)
    return result


def retrieve_with_shared_refiner(request):
    boundary = request.get("shared_boundary")
    if request["route"] != "visual" or not boundary:
        return retrieve(request)
    torch.set_num_threads(3)
    device = request.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    runtime = load_visual_runtime(device)
    result = retrieve(request, visual_runtime=runtime, retain_reserve=True)
    encoder = runtime[2]
    del runtime
    if device == "cuda":
        torch.cuda.empty_cache()
    coarse = result["predictions"]
    if not coarse:
        result.pop("candidate_reserve", None)
        return result
    for index, proposal in enumerate(coarse, 1):
        proposal["candidate_id"] = f"neural_{index:04d}"
    print("CUTLINER_STAGE:refinement", flush=True)
    sys.path.insert(0, str(PROJECT / "experiments/boundary_refiner_local"))
    from predict_boundary import predict
    from mini_agent.video.boundary_refiner import merge_refinements
    payload = {**boundary, "video": request["video"], "device": device,
               "frame_view": request.get("frame_view", "fit"),
               "duration_seconds": request["duration_seconds"], "query": request["visual_query_en"],
               "candidates": [{k: item[k] for k in ("start_seconds", "end_seconds", "score")} for item in coarse[:3]]}
    refinement = predict(payload, encoder=encoder)
    refinement["shared_visual_encoder"] = True
    result["coarse_predictions"] = coarse
    result["predictions"] = merge_refinements(coarse, refinement, request["duration_seconds"])
    finalize_refined_pool(result, request["top_k"])
    result["details"]["boundary_refinement"] = refinement
    return result


if __name__ == "__main__":
    request = json.loads(sys.stdin.readline())
    with contextlib.redirect_stdout(sys.stderr):
        result = retrieve_with_shared_refiner(request)
    print(json.dumps(result, ensure_ascii=False))
