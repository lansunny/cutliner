"""Local QD-DETR smoke run: CLIP features -> learned temporal intervals.

Run this file directly in PyCharm using .venv/Scripts/python.exe.
The English CLIP baseline does not translate arbitrary Chinese queries.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
sys.path.insert(0, str(ROOT / "vendor" / "lighthouse"))
sys.path.insert(0, str(ROOT / "vendor" / "UniVTG"))

from lighthouse.common.qd_detr import build_model
from lighthouse.common.utils.span_utils import span_cxw_to_xx
from run_on_video import clip
from mini_agent.video.frame_geometry import FIT_PROFILE, square_geometry


def load_checkpoint(device):
    path = ROOT / "weights" / "clip_qd_detr_qvhighlight.ckpt"
    with torch.serialization.safe_globals([argparse.Namespace, EasyDict]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["opt"]
    config = EasyDict(vars(config) if isinstance(config, argparse.Namespace) else config)
    config.device = device
    model, _ = build_model(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().to(device)
    return model, config


def _decode_profiles(hwaccel, device, duration, pixel_format):
    if hwaccel == "none" or (hwaccel == "auto" and (device != "cuda" or (duration or 0) < 60)):
        return ("none",)
    fmt = (pixel_format or "").split("(", 1)[0].strip()
    transfer = {"yuv420p": "nv12", "yuvj420p": "nv12", "yuv420p10le": "p010le"}.get(fmt)
    if transfer:
        return (f"cuda_sparse_{transfer}", "cuda", "none")
    return ("cuda", "none")


def _sample_frames(video, start, duration, cadence, tone_map, backend, *, frame_view="center"):
    threads = "2" if (duration or 0) < 60 else str(min(8, os.cpu_count() or 2))
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-v", "error",
               "-xerror", "-threads", threads]
    filters = f"fps=1/{cadence}"
    if backend.startswith("cuda"):
        command += ["-hwaccel", "cuda"]
    if backend.startswith("cuda_sparse_"):
        # Thin frames on the decoding device. Copy only the frames CLIP sees.
        command += ["-hwaccel_output_format", "cuda"]
        filters += ",hwdownload,format=" + backend.removeprefix("cuda_sparse_")
    if start:
        command += ["-ss", str(start)]
    command += ["-i", str(video)]
    if duration is not None:
        command += ["-t", str(duration)]
    geometry, padding = square_geometry(frame_view)
    filters += "," + geometry
    if tone_map:
        filters += ",zscale=t=linear:npl=100,format=gbrpf32le,tonemap=hable:desat=0,zscale=p=bt709:t=bt709:m=bt709:r=tv"
    filters += padding
    command += ["-map", "0:v:0", "-an", "-vf", filters, "-fps_mode", "passthrough",
                "-threads", "2", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    size = 224 * 224 * 3
    with tempfile.TemporaryFile() as errors, subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=errors,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    ) as process:
        try:
            while raw := process.stdout.read(size):
                if len(raw) != size:
                    raise RuntimeError("FFmpeg 输出了不完整的检索帧")
                yield raw
            if process.wait(timeout=10):
                errors.seek(0, os.SEEK_END)
                errors.seek(max(0, errors.tell() - 1200))
                raise RuntimeError("FFmpeg 检索解码失败：" + errors.read().decode("utf8", errors="replace"))
        finally:
            if process.poll() is None:
                process.kill()


def extract_features(video, start, duration, clip_model, device, cadence, batch_size, tone_map, hwaccel,
                     *, pixel_format=None, diagnostics=None, frame_view="center"):
    square_geometry(frame_view)  # Validate before consulting a cache.
    stat = video.stat()
    signature = {
        "path": str(video.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        "start": start, "duration": duration, "cadence": cadence,
        "encoder": "CLIP-ViT-B-32-40d36571", "crop": "center224" if frame_view == "center" else FIT_PROFILE,
        "tone_map": tone_map,
        "preprocess_version": 1,
    }
    backends = _decode_profiles(hwaccel, device, duration, pixel_format)
    details = diagnostics if diagnostics is not None else {}
    details["frame_view"] = frame_view

    def cache_for(backend):
        key = hashlib.sha256(json.dumps({**signature, "decoder": backend}, sort_keys=True).encode()).hexdigest()[:20]
        return ROOT / "cache" / f"{key}.npz"

    bypass_cache = os.getenv("CUTLINER_BYPASS_VISUAL_CACHE") == "1"
    if not bypass_cache:
        for backend in backends:
            cache = cache_for(backend)
            if cache.is_file():
                with np.load(cache, allow_pickle=False) as data:
                    features = torch.from_numpy(data["features"])
                details.update(backend=backend, visual_frames=len(features), from_cache=True)
                print(f"Reusing {len(features)} cached visual features", flush=True)
                return features, str(cache), True

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    for backend in backends:
        print(f"Visual decoder: {backend}", flush=True)
        started = time.perf_counter()
        features, batch = [], []

        def encode():
            tensor = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).to(device).float() / 255
            features.append(clip_model.encode_image((tensor - mean) / std).float().cpu())
            batch.clear()

        reader = _sample_frames(video, start, duration, cadence, tone_map, backend, frame_view=frame_view)
        try:
            count = 0
            for count, raw in enumerate(reader, 1):
                batch.append(np.frombuffer(raw, np.uint8).reshape(224, 224, 3))
                if len(batch) == batch_size:
                    encode()
                if count % 25 == 0:
                    print(f"Encoded {count} frames; scanned about {count * cadence:.0f}s", flush=True)
            if not count or ((duration or 0) >= 60 and count < int(duration / cadence) - 3):
                raise RuntimeError(f"检索解码帧数不足：{count}")
            if batch:
                encode()
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            if backend == backends[-1] or isinstance(exc, torch.cuda.OutOfMemoryError):
                raise
            print(f"{backend} 解码不可用，尝试下一后端：{str(exc)[-260:]}", flush=True)
            continue
        finally:
            reader.close()
        result = torch.cat(features)
        cache = cache_for(backend)
        cache.parent.mkdir(exist_ok=True)
        # Avoid publishing partial arrays to concurrent queries.
        with tempfile.NamedTemporaryFile(dir=cache.parent, suffix=".npz", delete=False) as temp:
            target = Path(temp.name)
        try:
            np.savez_compressed(target, features=result.numpy())
            target.replace(cache)
        finally:
            target.unlink(missing_ok=True)
        details.update(backend=backend, visual_frames=len(result), from_cache=False,
                       decode_encode_seconds=time.perf_counter() - started)
        return result, str(cache), False


def temporal_iou(first, second):
    intersection = max(0.0, min(first["end"], second["end"]) - max(first["start"], second["start"]))
    union = max(first["end"], second["end"]) - min(first["start"], second["start"])
    return intersection / union if union > 0 else 0.0


def locate(model, visual, text, text_mask, config, start, duration, device, *, window_batch=1, window_stride=None):
    # The released model was trained on up to 150s; retain that scale and slide.
    if window_batch < 1 or (window_stride is not None and window_stride < 1):
        raise ValueError("The temporal window batch and stride must be positive")
    cadence = float(config.clip_length)
    maximum = min(75, int(config.max_v_l))
    count = len(visual)
    if count <= maximum:
        starts = [0]
    else:
        stride = min(maximum, window_stride) if window_stride is not None else max(1, maximum - 15)
        starts = list(range(0, count - maximum + 1, stride))
        if starts[-1] != count - maximum:
            starts.append(count - maximum)
    length = min(count, maximum)
    begin = torch.arange(length, dtype=torch.float32) / length
    tef = torch.stack([begin, begin + 1.0 / length], dim=-1)
    proposals = []
    for first in range(0, len(starts), window_batch):
        offsets = starts[first:first+window_batch]
        size = len(offsets)
        chunks = torch.stack([visual[i:i+length] for i in offsets])
        chunks = F.normalize(chunks, dim=-1, eps=1e-5)
        inputs = {
            "src_vid": torch.cat([chunks, tef.expand(size, -1, -1)], dim=-1).to(device),
            "src_vid_mask": torch.ones(size, length, device=device),
            "src_txt": text.expand(size, -1, -1), "src_txt_mask": text_mask.expand(size, -1),
        }
        output = model(**inputs)
        scores = output["pred_logits"].softmax(-1)[..., 0].cpu().tolist()
        spans = (span_cxw_to_xx(output["pred_spans"]).cpu().clamp(0, 1) * length * cadence).tolist()
        for batch_index, offset in enumerate(offsets):
            for span, score in zip(spans[batch_index], scores[batch_index]):
                lower = start + offset * cadence + span[0]
                upper = min(start + duration, start + offset * cadence + span[1])
                if upper > lower:
                    proposals.append({"start": lower, "end": upper, "score": score, "window": first+batch_index})
    selected = []
    for proposal in sorted(proposals, key=lambda item: item["score"], reverse=True):
        if all(temporal_iou(proposal, kept) < 0.5 for kept in selected):
            selected.append(proposal)
    return selected, len(starts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=Path("E:/迅雷云盘/Euphoria.S02E08_10m00s-20m00s.mkv"))
    parser.add_argument("--query", default="A woman is riding a wooden horse.")
    parser.add_argument("--query-zh", default="", help="Optional human-written Chinese query label")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--frame-view", choices=["fit", "center"], default="fit",
                        help="Preserve the full display frame; center is a legacy comparison only.")
    parser.add_argument("--hwaccel", choices=["auto", "cuda", "none"], default="auto")
    parser.add_argument("--tone-map", action=argparse.BooleanOptionalAction, default=None,
                        help="HDR-to-SDR conversion; auto-detected from video metadata by default")
    parser.add_argument("--export-top", type=int, default=0)
    args = parser.parse_args()
    if not args.video.is_file():
        parser.error(f"Video does not exist: {args.video}")
    if args.start < 0 or args.batch_size < 1 or (args.duration is not None and args.duration <= 0):
        parser.error("Invalid start/duration/batch size")
    torch.set_num_threads(4)
    started = time.perf_counter()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = "cpu" if device == "auto" else device
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    metadata_reader = imageio_ffmpeg.read_frames(str(args.video))
    try:
        metadata = next(metadata_reader)
    finally:
        metadata_reader.close()
    if args.tone_map is None:
        args.tone_map = any(tag in metadata.get("pix_fmt", "") for tag in ("smpte2084", "arib-std-b67"))
    hwaccel = ("cuda" if device == "cuda" else "none") if args.hwaccel == "auto" else args.hwaccel
    available = float(metadata["duration"]) - args.start
    duration = min(available, args.duration) if args.duration is not None else available
    if duration <= 0:
        parser.error("Start must precede end of video")
    print(f"Device: {device}; video interval: {args.start:.2f}--{args.start + duration:.2f}s", flush=True)
    print(f"Query (English CLIP baseline): {args.query}", flush=True)
    print(f"HDR-to-SDR: {args.tone_map}", flush=True)
    print(f"Hardware decoder: {hwaccel}", flush=True)

    stage = time.perf_counter()
    model, config = load_checkpoint(device)
    print(f"QD-DETR loaded strictly: {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
    clip_model, _ = clip.load(str(ROOT / "weights" / "ViT-B-32.pt"), device=device, jit=False)
    clip_model.eval()
    load_seconds = time.perf_counter() - stage
    stage = time.perf_counter()
    with torch.inference_mode():
        visual, cache, cache_hit = extract_features(
            args.video, args.start, duration, clip_model, device, float(config.clip_length), args.batch_size, args.tone_map, hwaccel,
            pixel_format=metadata.get("pix_fmt"), frame_view=args.frame_view,
        )
        feature_seconds = time.perf_counter() - stage
        stage = time.perf_counter()
        tokens = clip.tokenize(args.query).to(device)
        text = F.normalize(clip_model.encode_text(tokens)["last_hidden_state"].float(), dim=-1, eps=1e-5)
        mask = (tokens != 0).float()
        # Match training preprocessing: L2-normalize CLIP states before projection.
        proposals, window_count = locate(model, visual, text, mask, config, args.start, duration, device)
        if device == "cuda":
            torch.cuda.synchronize()
        localization_seconds = time.perf_counter() - stage

    run_dir = ROOT / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True)
    report = {
        "model": "QD-DETR / Lighthouse CLIP-only, trained on QVHighlights",
        "source": {
            "repository": "https://github.com/line/lighthouse",
            "commit": "3e841d3d662c8440afed514908566fed2d5e82a7",
            "checkpoint": "https://zenodo.org/records/13363606",
            "checkpoint_md5": "d0b73515cdf96772ed1d6916102f0126",
            "clip_adapter_repository": "https://github.com/showlab/UniVTG",
            "clip_adapter_commit": "32659ac7aeba21742a63274f30eba785fc57e247",
        },
        "video": str(args.video.resolve()), "query": args.query, "query_zh": args.query_zh,
        "interval": [args.start, args.start + duration], "device": device,
        "visual_frames": len(visual), "sample_interval_seconds": float(config.clip_length),
        "localization_parameters": sum(p.numel() for p in model.parameters()),
        "encoder_parameters": sum(p.numel() for p in clip_model.parameters()),
        "windows": window_count, "cache": cache, "cache_hit": cache_hit, "tone_map": args.tone_map,
        "hardware_decoder": hwaccel,
        "timing_seconds": {"model_load": load_seconds, "visual_features": feature_seconds,
                           "localization": localization_seconds, "total_before_export": time.perf_counter() - started},
        "peak_allocated_gpu_mib": torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else None,
        "predictions": proposals,
        "limitations": ["Scores are not calibrated event-existence probabilities.",
                        "English checkpoint; no automatic Chinese translation.",
                        "CLIP-only uses independent image features plus learned temporal localization, not a motion encoder.",
                        "Long-video sliding-window scores are not globally calibrated; no recall or accuracy claim.",
                        "Automatic semantic verification is not part of this smoke run."],
        "exports": [],
    }
    for index, item in enumerate(proposals[:5], 1):
        print(f"Top {index}: {item['start']:.2f}--{item['end']:.2f}s; score={item['score']:.4f}", flush=True)
    report_path = run_dir / "result.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for index, item in enumerate(proposals[:max(0, args.export_top)], 1):
        target = run_dir / f"candidate_{index:02d}.mp4"
        filters = "scale=960:-2"
        if args.tone_map:
            filters += ",zscale=t=linear:npl=100,format=gbrpf32le,tonemap=hable:desat=0,zscale=p=bt709:t=bt709:m=bt709:r=tv,format=yuv420p"
        command = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-threads", "2",
                   *(["-hwaccel", "cuda"] if hwaccel == "cuda" else []),
                   "-ss", str(item["start"]), "-i", str(args.video), "-t", str(item["end"] - item["start"]),
                   "-map", "0:v:0", "-map", "0:a?", "-vf", filters, "-c:v", "libx264", "-preset", "veryfast",
                   "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-threads", "2", str(target)]
        subprocess.run(command, check=True)
        report["exports"].append(str(target))
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Result: {report_path}", flush=True)
    print(f"Timing: {report['timing_seconds']}; peak allocated GPU: {report['peak_allocated_gpu_mib']} MiB", flush=True)


if __name__ == "__main__":
    main()
