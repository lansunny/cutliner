"""Label-free deployment worker for the frozen visual boundary pipeline.

Reads a video, query and at most three actual coarse proposals from stdin.
It never reads training/test manifests, calls APIs or changes model parameters.
"""

import contextlib
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

import imageio_ffmpeg
import numpy as np
import torch
from torch.nn import functional as F

from candidate_model import CandidateRanker
from interval_model import IntervalQualityRefiner


HERE = Path(__file__).resolve().parent
VISUAL = HERE.parent / "video_grounding_local"
PROFILE = "clip_b32_actual_pts_half_second_bins_bicubic_center224_v1"
PTS = re.compile(r"\bn:\s*\d+.*?\bpts_time:\s*([-+\d.eE]+)")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_windows(video, windows):
    # Selection starts at the source origin, BEFORE restricting to windows.
    # Seeking/resetting PTS at each crop would shift the trained global grid.
    # FFmpeg compares doubles, while the archived observation mask uses float32
    # PTS. Decode a tiny halo, then apply the exact float32 mask in Python.
    condition = "+".join(f"gte(t,{max(0., lo - 1e-4):.9f})*lt(t,{hi + 1e-4:.9f})" for lo, hi in windows)
    filters = ("setpts=PTS-STARTPTS,"
               "select='isnan(prev_selected_t)+gt(floor(t*2),floor(prev_selected_t*2))',"
               f"select='{condition}',"
               "scale=224:224:flags=bicubic:force_original_aspect_ratio=increase,"
               "crop=224:224,showinfo")
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-v", "info",
               "-xerror", "-threads", "2", "-t", str(max(hi for _, hi in windows) + 1.),
               "-i", str(video), "-map", "0:v:0", "-an", "-vf", filters, "-threads", "2",
               "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    completed = subprocess.run(command, capture_output=True, timeout=1500,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    stderr = completed.stderr.decode("utf-8", errors="replace")
    if completed.returncode:
        raise RuntimeError("局部真实帧解码失败：" + stderr[-2000:])
    size = 224 * 224 * 3
    if len(completed.stdout) % size:
        raise ValueError("局部帧数据不完整")
    frames = np.frombuffer(completed.stdout, np.uint8).reshape(-1, 224, 224, 3)
    times = np.asarray([float(x) for x in PTS.findall(stderr)], dtype=np.float32)
    if len(frames) != len(times) or not np.all(np.diff(times) > 0):
        raise ValueError("真实帧与源时间戳不一致")
    return frames, times


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
def predict(request):
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
    print(f"重读 {len(windows)} 个候选附近的真实画面，使用源视频全局0.5秒网格", flush=True)
    frames, times = decode_windows(Path(request["video"]), windows)
    sys.path.insert(0, str(VISUAL / "vendor/UniVTG"))
    from run_on_video import clip
    encoder, _ = clip.load(str(VISUAL / "weights/ViT-B-32.pt"), device=device, jit=False)
    encoder.eval().requires_grad_(False)
    text = encode_query(encoder, request["query"], device)
    visual = encode_visual(encoder, frames, device)
    result = refine_features(manifest, local, ranker, visual, times, text, candidates, duration, request["query"], device)
    result["manifest_sha256"] = sha256(request["manifest"])
    return result


if __name__ == "__main__":
    request = json.loads(sys.stdin.readline())
    with contextlib.redirect_stdout(sys.stderr):
        result = predict(request)
    print(json.dumps(result, ensure_ascii=False))
