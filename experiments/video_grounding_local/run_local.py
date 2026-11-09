"""Local QD-DETR smoke run: CLIP features -> learned temporal intervals.

Run this file directly in PyCharm using .venv/Scripts/python.exe.
The English CLIP baseline does not translate arbitrary Chinese queries.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor" / "lighthouse"))
sys.path.insert(0, str(ROOT / "vendor" / "UniVTG"))

from lighthouse.common.qd_detr import build_model
from lighthouse.common.utils.span_utils import span_cxw_to_xx
from run_on_video import clip


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


def extract_features(video, start, duration, clip_model, device, cadence, batch_size, tone_map, hwaccel):
    stat = video.stat()
    signature = {
        "path": str(video.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        "start": start, "duration": duration, "cadence": cadence,
        "encoder": "CLIP-ViT-B-32-40d36571", "crop": "center224", "tone_map": tone_map,
        "preprocess_version": 1, "decoder": hwaccel,
    }
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:20]
    cache = ROOT / "cache" / f"{key}.npz"
    cache.parent.mkdir(exist_ok=True)
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            features = torch.from_numpy(data["features"])
        print(f"Reusing {len(features)} cached visual features", flush=True)
        return features, str(cache), True

    input_params = ["-threads", "2"]
    if hwaccel == "cuda":
        input_params += ["-hwaccel", "cuda"]
    if start:
        input_params += ["-ss", str(start)]
    output_params = []
    if duration is not None:
        output_params += ["-t", str(duration)]
    filters = f"fps=1/{cadence},scale=224:224:force_original_aspect_ratio=increase,crop=224:224"
    if tone_map:
        filters += ",zscale=t=linear:npl=100,format=gbrpf32le,tonemap=hable:desat=0,zscale=p=bt709:t=bt709:m=bt709:r=tv"
    output_params += ["-vf", filters, "-vsync", "0", "-threads", "2"]
    reader = imageio_ffmpeg.read_frames(str(video), input_params=input_params, output_params=output_params)
    features, batch = [], []
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    def encode():
        tensor = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).to(device).float() / 255
        encoded = clip_model.encode_image((tensor - mean) / std).float().cpu()
        features.append(encoded)
        batch.clear()

    try:
        metadata = next(reader)
        if tuple(metadata["size"]) != (224, 224):
            raise RuntimeError(f"Unexpected decoded size: {metadata['size']}")
        for index, raw in enumerate(reader):
            batch.append(np.frombuffer(raw, np.uint8).reshape(224, 224, 3))
            if len(batch) == batch_size:
                encode()
            if (index + 1) % 25 == 0:
                print(f"Encoded {index + 1} frames; scanned about {(index + 1) * cadence:.0f}s", flush=True)
        if batch:
            encode()
    finally:
        reader.close()
    if not features:
        raise RuntimeError("No frames decoded")
    result = torch.cat(features)
    np.savez_compressed(cache, features=result.numpy())
    return result, str(cache), False


def temporal_iou(first, second):
    intersection = max(0.0, min(first["end"], second["end"]) - max(first["start"], second["start"]))
    union = max(first["end"], second["end"]) - min(first["start"], second["start"])
    return intersection / union if union > 0 else 0.0


def locate(model, visual, text, text_mask, config, start, duration, device):
    # The released model was trained on up to 150s; retain that scale and slide.
    cadence = float(config.clip_length)
    maximum = min(75, int(config.max_v_l))
    count = len(visual)
    if count <= maximum:
        starts = [0]
    else:
        stride = max(1, maximum - 15)
        starts = list(range(0, count - maximum + 1, stride))
        if starts[-1] != count - maximum:
            starts.append(count - maximum)
    proposals = []
    for window_index, offset in enumerate(starts):
        chunk = F.normalize(visual[offset:offset + maximum], dim=-1, eps=1e-5)
        length = len(chunk)
        begin = torch.arange(length, dtype=torch.float32) / length
        tef = torch.stack([begin, begin + 1.0 / length], dim=-1)
        inputs = {
            "src_vid": torch.cat([chunk, tef], dim=-1).unsqueeze(0).to(device),
            "src_vid_mask": torch.ones(1, length, device=device),
            "src_txt": text, "src_txt_mask": text_mask,
        }
        output = model(**inputs)
        scores = output["pred_logits"][0].softmax(-1)[:, 0].cpu()
        spans = span_cxw_to_xx(output["pred_spans"][0]).cpu().clamp(0, 1) * length * cadence
        for span, score in zip(spans.tolist(), scores.tolist()):
            lower = start + offset * cadence + span[0]
            upper = min(start + duration, start + offset * cadence + span[1])
            if upper > lower:
                proposals.append({"start": lower, "end": upper, "score": score, "window": window_index})
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
