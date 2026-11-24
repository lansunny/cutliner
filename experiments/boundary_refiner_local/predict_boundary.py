"""Label-free deployment worker for the frozen visual boundary pipeline.

Reads a video, query and at most three actual coarse proposals from stdin.
It never reads training/test manifests, calls APIs or changes model parameters.
"""

import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

import imageio_ffmpeg
import numpy as np
import torch
from torch.nn import functional as F

from candidate_model import CandidateRanker
from interval_model import IntervalQualityRefiner


HERE = Path(__file__).resolve().parent
VISUAL = HERE.parent / "video_grounding_local"
sys.path.insert(0, str(HERE.parents[1]))
from mini_agent.video.frame_geometry import FIT_PROFILE, square_geometry

PROFILE = "clip_b32_actual_pts_half_second_bins_bicubic_center224_v1"
PTS = re.compile(r"\bn:\s*\d+.*?\bpts_time:\s*([-+\d.eE]+)")
FIRST_PTS = re.compile(r"\bn:\s*0\s+pts:\s*(-?\d+)\s+pts_time:\s*([-+\d.eE]+)")
INPUT_START = re.compile(r"Duration:\s*[^,]+,\s*start:\s*([-+\d.eE]+)")
TIME_BASE = re.compile(r"\bconfig in time_base:\s*(\d+)/(\d+)")
SEEK_PREROLL = 1.25
MEDIA_THREADS = str(min(8, os.cpu_count() or 2))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _ffmpeg(video, input_options, output_options, hwaccel, timeout):
    hardware = ["-hwaccel", "cuda"] if hwaccel.startswith("cuda") else []
    if hwaccel.startswith("cuda_sparse_"):
        hardware += ["-hwaccel_output_format", "cuda"]
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-v", "info",
               "-xerror", "-threads", MEDIA_THREADS, *hardware, *input_options,
               "-i", str(video), "-map", "0:v:0", "-an", *output_options]
    completed = subprocess.run(command, capture_output=True, timeout=timeout,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    stderr = completed.stderr.decode("utf-8", errors="replace")
    if completed.returncode:
        raise RuntimeError("局部真实帧解码失败：" + stderr[-1200:])
    return completed.stdout, stderr


def _source_origin(video):
    _, stderr = _ffmpeg(video, ["-copyts"], ["-vf", "showinfo", "-frames:v", "1", "-f", "null", "-"],
                        "cpu", timeout=20)
    match = FIRST_PTS.search(stderr)
    start = INPUT_START.search(stderr)
    base = TIME_BASE.search(stderr)
    if match is None or start is None or base is None:
        raise ValueError("无法确定源视频首帧时间戳")
    pixel_format = re.search(r"\bfmt:(\w+)", stderr[match.end():])
    pts = int(match[1])
    seconds = pts * int(base[1]) / int(base[2])
    return pts, seconds, pixel_format[1] if pixel_format else None, float(start[1])


def _merged_groups(windows):
    groups = []
    for lo, hi in sorted(windows):
        if not groups or lo > groups[-1][1] + 2.:
            groups.append((lo, hi, [(lo, hi)]))
        else:
            a, b, items = groups[-1]
            items.append((lo, hi))
            groups[-1] = (a, max(b, hi), items)
    return groups


def _decode(video, windows, hwaccel, *, origin=None, frame_view="center"):
    # Keep the frozen preprocessing: select the first real frame in each global
    # half-second bin before applying the union of observation windows.
    condition = "+".join(f"gte(t,{max(0., lo - 1e-4):.9f})*lt(t,{hi + 1e-4:.9f})" for lo, hi in windows)
    hi = max(hi for _, hi in windows)
    lo = min(lo for lo, _ in windows)
    seeked = False
    if origin is None:
        pts = "STARTPTS"
        input_options = ["-t", str(hi + 1.)]
    else:
        # Accurate input seeking decodes from the previous keyframe. Keep the
        # original integer PTS and leave more than two half-second bins of preroll.
        ticks, source_time, _, container_start = origin
        pts = f"({ticks})"
        start = max(0., lo - SEEK_PREROLL)
        input_options = ["-copyts"]
        if start:
            # Input -ss is container-relative. With -copyts a timestamp-mode seek
            # can discard the preroll of nonzero-start MP4s a second time.
            start = max(0., start + source_time - container_start)
            if start:
                seeked = True
                input_options += ["-ss", f"{start:.9f}"]
                condition = f"eq(n,0)+({condition})"
        input_options += ["-t", f"{hi + source_time - container_start - start + 1.:.9f}"]
    transfer = (",hwdownload,format=" + hwaccel.removeprefix("cuda_sparse_")
                if hwaccel.startswith("cuda_sparse_") else "")
    geometry, padding = square_geometry(frame_view, bicubic=True)
    filters = (f"setpts=PTS-{pts},"
               "select='isnan(prev_selected_t)+gt(floor(t*2),floor(prev_selected_t*2))',"
               f"select='{condition}'{transfer},"
               f"{geometry}{padding},showinfo")
    pixels, stderr = _ffmpeg(video, input_options, ["-vf", filters, "-threads", "2",
                             "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"],
                             hwaccel, timeout=1500 if origin is None else 240)
    size = 224 * 224 * 3
    if len(pixels) % size:
        raise ValueError("局部帧数据不完整")
    frames = np.frombuffer(pixels, np.uint8).reshape(-1, 224, 224, 3)
    times = np.asarray([float(x) for x in PTS.findall(stderr)], dtype=np.float32)
    if len(frames) != len(times) or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError("真实帧与源时间戳不一致")
    if seeked:
        # The first sampled frame is a sentinel, not a model input. Check that
        # seeking observed at least one full bin prior to the requested window.
        if not len(times) or times[0] > lo - .55:
            raise ValueError("视频跳读没有保留足够的网格预读")
        frames, times = frames[1:], times[1:]
    return frames, times


def decode_windows(video, windows, *, hwaccel="cpu", diagnostics=None, frame_view="center"):
    prefix_seconds = max(hi for _, hi in windows) + 1.
    groups = _merged_groups(windows)
    seek_seconds = sum(hi - max(0., lo - SEEK_PREROLL) + 1. for lo, hi, _ in groups)
    seek = prefix_seconds >= 30. and seek_seconds < .8 * prefix_seconds
    report = diagnostics if diagnostics is not None else {}
    report["frame_view"] = frame_view
    origin = None
    if seek or hwaccel == "cuda":
        try:
            origin = _source_origin(video)
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired):
            # Fall back to decoding from the source origin if seeking cannot be
            # proven to retain the trained timestamp grid.
            seek = False
    fmt = {"yuv420p": "nv12", "yuvj420p": "nv12", "yuv420p10le": "p010le"}.get(origin[2] if origin else None)
    backends = ([f"cuda_sparse_{fmt}"] if fmt and hwaccel == "cuda" else [])
    backends += ["cuda", "cpu"] if hwaccel == "cuda" else ["cpu"]
    for decoder in backends:
        if seek:
            try:
                parts = [_decode(video, items, decoder, origin=origin, frame_view=frame_view) for _, _, items in groups]
                frames = np.concatenate([part[0] for part in parts])
                times = np.concatenate([part[1] for part in parts])
                if not np.all(np.diff(times) > 0):
                    raise ValueError("跳读得到重叠的源时间戳")
                for lo, hi in windows:
                    if hi - lo < 1.5:
                        continue
                    selected = times[(times >= lo - 1e-4) & (times < hi + 1e-4)]
                    if (len(selected) < 2 or selected[0] - lo > .8 or hi - selected[-1] > .8
                            or np.max(np.diff(selected)) > .8):
                        raise ValueError("跳读未完整覆盖候选窗口")
                report.update(mode="source_pts_seek", backend=decoder, groups=len(groups),
                              requested_input_seconds=seek_seconds, prefix_scan_seconds=prefix_seconds)
                return frames, times
            except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
                print(f"局部跳读未通过校验（{decoder}, {type(error).__name__}），尝试保真回退", flush=True)
        else:
            try:
                result = _decode(video, windows, decoder, frame_view=frame_view)
                report.update(mode="prefix", backend=decoder, groups=1,
                              requested_input_seconds=prefix_seconds, prefix_scan_seconds=prefix_seconds)
                return result
            except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired):
                if decoder == "cpu":
                    raise
    result = _decode(video, windows, "cpu", frame_view=frame_view)
    report.update(mode="prefix_fallback", backend="cpu", groups=1,
                  requested_input_seconds=prefix_seconds, prefix_scan_seconds=prefix_seconds)
    return result


def pack_observations(visual, times, query, candidates, duration, context=6.):
    """Same feature cells as training, with no target/label placeholders."""
    samples, skipped = [], []
    for index, proposal in enumerate(candidates, 1):
        a, b = proposal["start_seconds"], proposal["end_seconds"]
        lower, upper = max(0., a - context), min(duration, b + context)
        mask = (times >= lower) & (times < upper)
        selected = times[mask]
        if len(selected) < 2:
            skipped.append({"coarse_index": index, "reason": "insufficient_real_observations"})
            continue
        edges = np.concatenate(([lower], (selected[:-1] + selected[1:]) / 2, [upper])).astype(np.float32)
        if not np.all(np.diff(edges) > 0) or np.max(np.diff(selected)) > .8:
            raise ValueError("局部观察存在缺失或非法时间单元，不能复制画面补齐")
        samples.append({"visual": visual[mask], "times": selected, "edges": edges,
                        "coarse": [a, b], "window": [lower, upper], "coarse_index": index,
                        "coarse_score": proposal["score"]})
    if not samples:
        return None, samples, skipped
    count, length = len(samples), max(len(s["times"]) for s in samples)
    batch = {"visual": torch.zeros(count, length, 512), "times": torch.zeros(count, length),
             "edges": torch.zeros(count, length + 1), "mask": torch.zeros(count, length, dtype=torch.bool),
             "edge_mask": torch.zeros(count, length + 1, dtype=torch.bool),
             "query": torch.from_numpy(np.repeat(query[None], count, axis=0)).float(),
             "coarse": torch.tensor([s["coarse"] for s in samples], dtype=torch.float32),
             "window": torch.tensor([s["window"] for s in samples], dtype=torch.float32)}
    for i, sample in enumerate(samples):
        n = len(sample["times"])
        for key in ("visual", "times"):
            batch[key][i, :n] = torch.from_numpy(sample[key])
        batch["edges"][i, :n + 1] = torch.from_numpy(sample["edges"])
        batch["mask"][i, :n] = True
        batch["edge_mask"][i, :n + 1] = True
    return batch, samples, skipped


@torch.inference_mode()
def score_observations(local, ranker, batch, samples, duration, device):
    batch = {k: v.to(device) for k, v in batch.items()}
    out = local(batch)
    size = batch["edges"].shape[-1]
    i = (out["start"].softmax(-1).cumsum(-1) < .3).sum(-1).clamp(max=size - 1)
    j = (out["end"].softmax(-1).cumsum(-1) < .7).sum(-1).clamp(max=size - 1)
    j = torch.maximum(j, i + 1).minimum(batch["mask"].sum(-1))
    i = torch.minimum(i, j - 1)
    ids = torch.arange(len(i), device=device)
    prediction = torch.stack([batch["edges"][ids, i], batch["edges"][ids, j]], -1)
    embedding = out["temporal_hidden"].sum(1) / batch["mask"].sum(1)[:, None]
    query = F.normalize(batch["query"], dim=-1)
    similarity = (F.normalize(batch["visual"], dim=-1) * query[:, None]).sum(-1)
    inside = batch["mask"] & (batch["times"] >= prediction[:, :1]) & (batch["times"] <= prediction[:, 1:])
    outside = batch["mask"] & ~inside
    sim_in = (similarity * inside).sum(-1) / inside.sum(-1).clamp(min=1)
    sim_out = (similarity * outside).sum(-1) / outside.sum(-1).clamp(min=1)
    entropy = []
    for key in ("start", "end"):
        p = out[key].softmax(-1)
        entropy.append(-(p * p.clamp(min=1e-8).log()).sum(-1) / batch["edge_mask"].sum(-1).float().log())
    # Preserve the feature order and float32 scaling of the frozen ranker.
    scale = torch.full((len(samples),), duration, dtype=torch.float32, device=device)
    coarse, window, relative = batch["coarse"] / scale[:, None], batch["window"] / scale[:, None], prediction / scale[:, None]
    quality = out["iou_quality"][ids, i, j].sigmoid()
    meta = torch.stack([torch.tensor([s["coarse_score"] for s in samples], device=device),
        torch.tensor([s["coarse_index"] for s in samples], device=device) / 3,
        coarse[:, 0], coarse[:, 1], coarse[:, 1] - coarse[:, 0], window[:, 0], window[:, 1], window[:, 1] - window[:, 0],
        relative[:, 0], relative[:, 1], relative[:, 1] - relative[:, 0], quality,
        out["coverage_quality"][ids, i, j].sigmoid(), out["outside"][:, 0].sigmoid(), out["outside"][:, 1].sigmoid(),
        entropy[0], entropy[1], sim_in, sim_out, sim_in - sim_out], -1)
    ranked = ranker({"embedding": embedding[None], "meta": meta[None], "query": query[:1],
                     "mask": torch.ones(1, len(samples), dtype=torch.bool, device=device)})
    logits = ranked["score"][0]
    if not all(torch.isfinite(t).all() for t in (prediction, meta, logits)):
        raise ValueError("边界网络产生非有限数值")
    return prediction.cpu().tolist(), quality.cpu().tolist(), logits.cpu().tolist(), meta.cpu()


def load_models(manifest, artifact_root, device):
    if (manifest["version"], manifest["feature_profile"], manifest["top_k"], manifest["context_seconds"],
        manifest["decoder_quantile"]) != (1, PROFILE, 3, 6., .3):
        raise ValueError("部署参数不符合已冻结的边界网络")
    for name, expected in manifest["source_sha256"].items():
        if sha256(HERE / name) != expected:
            raise ValueError(f"已验证的网络结构被修改：{name}")
    paths = [artifact_root / manifest["local_checkpoint"], artifact_root / manifest["ranker_checkpoint"]]
    for path, key in zip(paths, ("local_sha256", "ranker_sha256")):
        if sha256(path) != manifest[key]:
            raise ValueError(f"部署权重不匹配：{path.name}")
    if sha256(VISUAL / "weights/ViT-B-32.pt") != manifest["clip_sha256"]:
        raise ValueError("CLIP权重不匹配")
    local, ranker = IntervalQualityRefiner(), CandidateRanker("independent")
    for model, path in zip((local, ranker), paths):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"], strict=True)
        model.eval().requires_grad_(False).to(device)
    return local, ranker


@torch.inference_mode()
def encode_visual(encoder, frames, device):
    mean = torch.tensor([.48145466, .4578275, .40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([.26862954, .26130258, .27577711], device=device).view(1, 3, 1, 1)
    features = []
    for offset in range(0, len(frames), 32):
        pixels = torch.from_numpy(frames[offset:offset + 32].copy()).permute(0, 3, 1, 2).to(device).float() / 255
        encoded = encoder.encode_image((pixels - mean) / std).float().cpu().numpy()
        features.append(encoded.astype(np.float16).astype(np.float32))
    if not features:
        raise ValueError("候选范围内没有真实画面")
    return np.concatenate(features)


@torch.inference_mode()
def encode_query(encoder, query, device):
    sys.path.insert(0, str(VISUAL / "vendor/UniVTG"))
    from run_on_video import clip
    tokens = clip.tokenize(query, max_valid_length=100000).to(device)
    return encoder.encode_text(tokens)["pooler_output"][0].float().cpu().numpy().astype(np.float16).astype(np.float32)


def _observation_cache(video, windows, manifest, device, frame_view="center"):
    stat = video.stat()
    identity = [str(video.resolve()), stat.st_size, stat.st_mtime_ns, sorted(windows),
                manifest["clip_sha256"], PROFILE, device, "pts-seek-v2"]
    if frame_view == "fit":
        identity.append(FIT_PROFILE)
    key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24]
    return HERE / "cache" / f"{key}.npz"


def _read_observation_cache(path):
    if path is None or os.getenv("CUTLINER_BYPASS_VISUAL_CACHE") == "1" or not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as cached:
            visual, times = cached["visual"].astype(np.float32), cached["times"]
            if (times.ndim == 1 and len(times) > 0 and visual.shape == (len(times), 512)
                    and np.isfinite(visual).all() and np.isfinite(times).all() and np.all(np.diff(times) > 0)):
                return visual, times
    except (OSError, KeyError, ValueError, EOFError):
        return None
    return None


def _write_observation_cache(path, visual, times):
    path.parent.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        target = Path(stream.name)
    try:
        np.savez_compressed(target, visual=visual.astype(np.float16), times=times)
        target.replace(path)
    finally:
        target.unlink(missing_ok=True)


@torch.inference_mode()
def refine_features(manifest, local, ranker, visual, times, text, candidates, duration, query, device):
    """Shared label-free inference for production and source-cached evaluation."""
    batch, samples, skipped = pack_observations(visual, times, text, candidates, duration)
    refinements = []
    if batch is not None:
        prediction, quality, logits, _ = score_observations(local, ranker, batch, samples, duration, device)
        for sample, proposed, score, logit in zip(samples, prediction, quality, logits):
            lo, hi = sample["window"]
            a, b = max(lo, proposed[0]), min(hi, proposed[1])
            if not a < b:
                raise ValueError("非法的局部边界预测")
            evidence = {"model_name": manifest["model_name"], "feature_profile": PROFILE,
                "frozen_protocol_sha256": manifest["frozen_protocol_sha256"], "deployment_seed": manifest["deployment_seed"],
                "coarse_start_seconds": sample["coarse"][0], "coarse_end_seconds": sample["coarse"][1],
                "observation_start_seconds": lo, "observation_end_seconds": hi,
                "proposed_start_seconds": a, "proposed_end_seconds": b, "coarse_rank": sample["coarse_index"],
                "selection_logit": logit, "observed_frames": len(sample["times"])}
            refinements.append({"coarse_index": sample["coarse_index"], "start_seconds": a, "end_seconds": b,
                                "predicted_iou": score, "selection_logit": logit, "boundary_refinement": evidence})
            print(f"候选{sample['coarse_index']}：{sample['coarse'][0]:.2f}–{sample['coarse'][1]:.2f}s → {a:.2f}–{b:.2f}s", flush=True)
    return {"status": "ok", "model_name": manifest["model_name"], "query": query,
            "deployment_seed": manifest["deployment_seed"], "refinements": refinements, "skipped": skipped,
            "real_observed_frames": len(times), "cadence_seconds": .5,
            "local_checkpoint_sha256": manifest["local_sha256"], "ranker_checkpoint_sha256": manifest["ranker_sha256"],
            "certifies_full_query": False, "absence_detection_available": False,
            "observation_policy": "global half-second bins; actual source PTS; no interpolation or per-window timestamp reset",
            "domain_limits": "Trained on Charades visual intervals; film/HDR/audio boundary quality is not established"}


@torch.inference_mode()
def predict(request, *, encoder=None):
    torch.set_num_threads(3)
    manifest = json.loads(Path(request["manifest"]).read_text(encoding="utf-8"))
    candidates, duration = request["candidates"], request["duration_seconds"]
    if not 1 <= len(candidates) <= 3 or not math.isfinite(duration) or duration <= 0 or not request["query"]:
        raise ValueError("非法的边界网络输入")
    for item in candidates:
        if not all(math.isfinite(item[k]) for k in ("start_seconds", "end_seconds", "score")) or not (
            0 <= item["start_seconds"] < item["end_seconds"] <= duration and 0 <= item["score"] <= 1):
            raise ValueError("非法粗候选，不能靠截断掩盖错误")
    device = request.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu", "cuda"):
        raise ValueError("非法设备")
    root = Path(request.get("artifact_root", manifest["artifact_root"]))
    local, ranker = load_models(manifest, root, device)
    windows = [(max(0., c["start_seconds"] - 6), min(duration, c["end_seconds"] + 6)) for c in candidates]
    video = Path(request["video"])
    frame_view = request.get("frame_view", "fit")
    square_geometry(frame_view)
    cache = _observation_cache(video, windows, manifest, device, frame_view) if duration >= 60 else None
    decode_details = {"frame_view": frame_view}
    started = time.perf_counter()
    if encoder is None:
        sys.path.insert(0, str(VISUAL / "vendor/UniVTG"))
        from run_on_video import clip
        encoder, _ = clip.load(str(VISUAL / "weights/ViT-B-32.pt"), device=device, jit=False)
        encoder.eval().requires_grad_(False)
    cached = _read_observation_cache(cache)
    if cached is not None:
        visual, times = cached
        decode_details.update(mode="feature_cache", backend="stored_clip_features", requested_input_seconds=0.,
                              seconds=time.perf_counter() - started, from_cache=True)
        print(f"复用 {len(times)} 个局部视觉特征，无需重新解码", flush=True)
    else:
        print(f"重读 {len(windows)} 个候选附近的真实画面，使用源视频全局0.5秒网格", flush=True)
        started = time.perf_counter()
        frames, times = decode_windows(video, windows, hwaccel="cuda" if device == "cuda" and duration >= 60 else "cpu",
                                      diagnostics=decode_details, frame_view=frame_view)
        decode_details.update(seconds=time.perf_counter() - started, from_cache=False)
        print(f"局部解码：{decode_details['mode']} / {decode_details['backend']}，"
              f"输入窗口 {decode_details['requested_input_seconds']:.1f}s", flush=True)
        visual = encode_visual(encoder, frames, device)
        if cache is not None and cache == _observation_cache(video, windows, manifest, device, frame_view):
            _write_observation_cache(cache, visual, times)
    text = encode_query(encoder, request["query"], device)
    result = refine_features(manifest, local, ranker, visual, times, text, candidates, duration, request["query"], device)
    if frame_view == "fit":
        for item in result["refinements"]:
            item["boundary_refinement"]["feature_profile"] = f"clip_b32_actual_pts_half_second_bins_{FIT_PROFILE}"
        result["domain_limits"] += "; full-frame input differs from center-crop training"
    result["decode"] = decode_details
    result["manifest_sha256"] = sha256(request["manifest"])
    return result


if __name__ == "__main__":
    request = json.loads(sys.stdin.readline())
    with contextlib.redirect_stdout(sys.stderr):
        result = predict(request)
    print(json.dumps(result, ensure_ascii=False))
