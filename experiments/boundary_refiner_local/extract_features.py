"""Cache frozen CLIP on observed 0.5s-bin source frames and original query tokens.

No interpolation, padding frames, downloads, API calls or encoder training.
The final test split is excluded by default.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import imageio_ffmpeg
import numpy as np
import torch

from prepare_data import DEFAULT_ROOT, REPO, dump, file_hash, read_jsonl
from final_protocol import load_frozen_protocol

CLIP_ROOT = REPO / "experiments/video_grounding_local"
CLIP_SHA = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
PROFILE = "clip_b32_actual_pts_half_second_bins_bicubic_center224_v1"
PTS = re.compile(r"\bn:\s*\d+.*?\bpts_time:\s*([-+\d.eE]+)")


def save_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".npz.tmp")
    with pending.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    pending.replace(path)


def decode(row):
    path = Path(row["video_path"])
    if file_hash(path) != row["video_sha256"]:
        raise ValueError(f"Media changed after dataset audit: {path}")
    # First genuine frame in every half-second time bin; showinfo preserves the
    # selected frame's PTS. Reset PTS origin to the first video frame explicitly.
    filters = ("setpts=PTS-STARTPTS,"
               "select='isnan(prev_selected_t)+gt(floor(t*2),floor(prev_selected_t*2))',"
               "scale=224:224:flags=bicubic:force_original_aspect_ratio=increase,"
               "crop=224:224,showinfo")
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-v", "info",
               "-xerror", "-threads", "1", "-i", str(path), "-map", "0:v:0", "-an",
               "-vf", filters, "-threads", "1", "-fps_mode", "passthrough",
               "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    completed = subprocess.run(command, capture_output=True, timeout=90)
    stderr = completed.stderr.decode("utf-8", errors="replace")
    if completed.returncode:
        raise RuntimeError(f"Full video decode failed: {path.name}: {stderr[-2000:]}")
    frame_bytes = 224 * 224 * 3
    if len(completed.stdout) % frame_bytes:
        raise ValueError("Truncated RGB output")
    frames = np.frombuffer(completed.stdout, np.uint8).reshape(-1, 224, 224, 3).copy()
    times = np.asarray([float(x) for x in PTS.findall(stderr)], dtype=np.float32)
    if len(frames) != len(times) or len(times) < 2 or not np.all(np.diff(times) > 0):
        raise ValueError(f"Frame/PTS alignment failed: {path.name}: {len(frames)}/{len(times)}")
    if times[0] > 0.1 or times[-1] > row["duration_seconds"] + 0.1:
        raise ValueError(f"Unexpected source time origin/extent: {path.name}")
    # A missing half-second bin is not silently repaired by duplicating frames.
    if np.max(np.diff(times)) > 0.8:
        raise ValueError(f"Missing source observation bins: {path.name}")
    return frames, times


def validate_video(path, row):
    with np.load(path, allow_pickle=False) as data:
        if str(data["profile"]) != PROFILE or str(data["media_sha256"]) != row["video_sha256"]:
            raise ValueError(f"Stale cache: {path}")
        visual, times = data["visual"], data["times"]
        if visual.shape != (len(times), 512) or not np.isfinite(visual).all() or not np.all(np.diff(times) > 0):
            raise ValueError(f"Invalid cache: {path}")
        return len(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--splits", nargs="+", choices=["train", "dev", "test"], default=["train", "dev"])
    parser.add_argument("--limit-videos", type=int)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--decode-workers", type=int, default=3)
    parser.add_argument("--final-protocol", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(3)
    root = args.root.resolve()
    is_test = "test" in args.splits
    if is_test:
        if args.splits != ["test"] or args.limit_videos:
            parser.error("Final test must be a separate, complete extraction invocation")
        protocol = load_frozen_protocol(args.final_protocol, root)
        if protocol["feature_profile"] != PROFILE:
            raise ValueError("Final encoder feature profile differs from frozen protocol")
    prefix = "test_" if is_test else ""
    feature_root = root / "features" / PROFILE
    feature_root.mkdir(parents=True, exist_ok=True)
    rows = [row for s in args.splits for row in read_jsonl(root / "manifests" / f"{s}.complete.jsonl")]
    videos = {row["video_id"]: row for row in rows}
    ordered = sorted(videos)
    if args.limit_videos:
        ordered = ordered[:args.limit_videos]
    selected = set(ordered)
    rows = [r for r in rows if r["video_id"] in selected]
    all_frames, pending = 0, []
    for vid in ordered:
        path = feature_root / "videos" / f"{vid}.npz"
        if path.exists():
            all_frames += validate_video(path, videos[vid])
        else:
            pending.append((videos[vid], path))
    lock = {"profile": PROFILE, "clip_sha256": CLIP_SHA, "cadence_seconds": 0.5,
            "timestamp_policy": "actual selected frame PTS after setpts=PTS-STARTPTS; no interpolation",
            "decode_policy": "full source decode with FFmpeg -xerror; one real frame per nonempty 0.5s bin",
            "text_policy": "original English query; pooled embedding and unprojected last hidden tokens",
            "splits": args.splits, "encoder_frozen": True}
    if is_test:
        lock["final_protocol_sha256"] = file_hash(args.final_protocol)
    dump(feature_root / f"{prefix}feature_configuration.json", lock)
    weights = CLIP_ROOT / "weights/ViT-B-32.pt"
    if file_hash(weights) != CLIP_SHA:
        raise ValueError("Local CLIP weights do not match verified source")
    sys.path.insert(0, str(CLIP_ROOT / "vendor/UniVTG"))
    from run_on_video import clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder, _ = clip.load(str(weights), device=device, jit=False)
    encoder.eval().requires_grad_(False)
    mean = torch.tensor([.48145466, .4578275, .40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([.26862954, .26130258, .27577711], device=device).view(1, 3, 1, 1)
    print(f"CLIP frozen on {device}; pending videos {len(pending)}/{len(ordered)}", flush=True)
    started, encoded_frames = time.monotonic(), 0
    with torch.inference_mode(), ThreadPoolExecutor(max_workers=args.decode_workers) as pool:
        prefetch = max(2, args.decode_workers * 2)
        futures = {i: pool.submit(decode, item[0]) for i, item in enumerate(pending[:prefetch])}
        for i, (row, path) in enumerate(pending):
            frames, times = futures.pop(i).result()
            if i + prefetch < len(pending):
                futures[i + prefetch] = pool.submit(decode, pending[i + prefetch][0])
            features = []
            for offset in range(0, len(frames), args.batch_size):
                pixels = torch.from_numpy(frames[offset:offset + args.batch_size]).permute(0, 3, 1, 2)
                pixels = pixels.to(device).float() / 255
                visual = encoder.encode_image((pixels - mean) / std).float()
                if not torch.isfinite(visual).all():
                    raise ValueError("Non-finite encoder output")
                features.append(visual.cpu().numpy().astype(np.float16))
            save_npz(path, visual=np.concatenate(features), times=times,
                     profile=np.asarray(PROFILE), media_sha256=np.asarray(row["video_sha256"]))
            encoded_frames += len(times)
            all_frames += len(times)
            if (i + 1) % 20 == 0 or i + 1 == len(pending):
                progress = {"phase": "visual", "new_videos": i + 1, "pending_at_start": len(pending),
                            "selected_videos": len(ordered), "observed_frames": all_frames,
                            "new_frames_per_second": round(encoded_frames / max(1, time.monotonic() - started), 2),
                            "elapsed_seconds": round(time.monotonic() - started, 1),
                            "pid": __import__("os").getpid(), "status": "running"}
                dump(feature_root / f"{prefix}progress.json", progress)
                print(json.dumps(progress), flush=True)
    text_pending = []
    for row in rows:
        path = feature_root / "texts" / f"{row['case_id']}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                if str(data["query"]) != row["query"] or str(data["profile"]) != PROFILE:
                    raise ValueError(f"Query cache changed: {path}")
        else:
            text_pending.append((row, path))
    with torch.inference_mode():
        for offset in range(0, len(text_pending), args.batch_size):
            batch = text_pending[offset:offset + args.batch_size]
            tokens = clip.tokenize([r["query"] for r, _ in batch], max_valid_length=100000).to(device)
            output = encoder.encode_text(tokens)
            for j, (row, path) in enumerate(batch):
                length = int(tokens[j].argmax().item()) + 1
                save_npz(path, pooled=output["pooler_output"][j].float().cpu().numpy().astype(np.float16),
                         tokens=output["last_hidden_state"][j, :length].float().cpu().numpy().astype(np.float16),
                         query=np.asarray(row["query"]), profile=np.asarray(PROFILE))
            if offset % (args.batch_size * 20) == 0:
                print(f"Cached text queries {min(offset + args.batch_size, len(text_pending))}/{len(text_pending)}", flush=True)
    report = {**lock, "status": "complete", "videos": len(ordered), "queries": len(rows),
              "real_observed_frames": all_frames, "elapsed_seconds": round(time.monotonic() - started, 1),
              "heldout_test_extracted": "test" in args.splits, "model_training_started_by_this_script": False}
    dump(feature_root / f"{prefix}extraction_report.json", report)
    dump(feature_root / f"{prefix}progress.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
