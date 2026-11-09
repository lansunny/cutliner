"""Frozen, ordered 16-frame VideoMAE encoder with reusable local feature cache."""

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import torch
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

from actions import resolve_label_ids
from prepare_model import ROOT, WEIGHTS, WEIGHT_SHA256, sha256_file, validate_files


class FrozenVideoEncoder:
    def __init__(self, device="auto", hwaccel="auto"):
        validate_files()
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if self.device == "auto":
            self.device = "cpu"
        self.hwaccel = ("cuda" if self.device == "cuda" else "none") if hwaccel == "auto" else hwaccel
        torch.set_num_threads(4)
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self.processor = VideoMAEImageProcessor.from_pretrained(WEIGHTS, local_files_only=True)
        self.model, loading = VideoMAEForVideoClassification.from_pretrained(
            WEIGHTS, local_files_only=True, output_loading_info=True,
        )
        if loading["missing_keys"] or loading["unexpected_keys"] or loading.get("mismatched_keys"):
            raise ValueError(f"Checkpoint did not load exactly: {loading}")
        self.model.eval().requires_grad_(False).to(self.device)
        self.action_labels = resolve_label_ids(self.model.config)
        self.num_frames = int(self.model.config.num_frames)
        if self.num_frames != 16:
            raise ValueError("This experiment expects 16-frame weights")
        identity = {
            "weights": WEIGHT_SHA256,
            "config": sha256_file(WEIGHTS / "config.json"),
            "processor": sha256_file(WEIGHTS / "preprocessor_config.json"),
            "implementation": "ordered16_center224_v1",
        }
        self.fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def read_frames(self, video, start=0.0, end=None):
        video = Path(video).resolve()
        reader = imageio_ffmpeg.read_frames(str(video))
        try:
            metadata = next(reader)
        finally:
            reader.close()
        end = float(metadata["duration"]) if end is None else float(end)
        if not 0 <= start < end <= float(metadata["duration"]) + 0.05:
            raise ValueError("Clip interval must be within the source file")
        # This is a local short-action verifier, not a whole-video classifier.
        if end - start > 12:
            raise ValueError("Select a short interval of at most 12 seconds")
        filters = f"fps={self.num_frames / (end - start):.10f}:start_time=0,scale=224:224:force_original_aspect_ratio=increase,crop=224:224"
        tone_map = any(tag in metadata.get("pix_fmt", "") for tag in ("smpte2084", "arib-std-b67"))
        if tone_map:
            filters += ",zscale=t=linear:npl=100,format=gbrpf32le,tonemap=hable:desat=0,zscale=p=bt709:t=bt709:m=bt709:r=tv"
        inputs = ["-threads", "2"]
        if self.hwaccel == "cuda":
            inputs += ["-hwaccel", "cuda"]
        if start:
            inputs += ["-ss", str(start)]
        outputs = ["-t", str(end - start), "-vf", filters, "-frames:v", str(self.num_frames), "-threads", "2"]
        reader = imageio_ffmpeg.read_frames(str(video), input_params=inputs, output_params=outputs)
        try:
            shape = next(reader)["size"]
            if tuple(shape) != (224, 224):
                raise ValueError(f"Unexpected frame dimensions: {shape}")
            frames = [np.frombuffer(raw, np.uint8).reshape(224, 224, 3).copy() for raw in reader]
        finally:
            reader.close()
        if not frames:
            raise ValueError("No source frames decoded")
        padded = max(0, self.num_frames - len(frames))
        frames = (frames + [frames[-1]] * padded)[:self.num_frames]
        return frames, {"start": start, "end": end, "tone_map": tone_map, "tail_padding_frames": padded}

    @torch.inference_mode()
    def encode(self, video, start=0.0, end=None, order="forward", use_cache=True):
        video = Path(video).resolve()
        stat = video.stat()
        signature = {"path": str(video), "size": stat.st_size, "mtime": stat.st_mtime_ns,
                     "start": float(start), "end": end, "order": order,
                     "encoder": self.fingerprint, "decoder": self.hwaccel}
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:24]
        cache = ROOT / "cache" / f"{key}.npz"
        cache.parent.mkdir(exist_ok=True)
        if use_cache and cache.is_file():
            with np.load(cache, allow_pickle=False) as data:
                return data["feature"], data["logits"], json.loads(str(data["metadata"])), True
        frames, metadata = self.read_frames(video, start, end)
        if order == "reverse":
            frames.reverse()
        elif order != "forward":
            raise ValueError("Order must be forward or reverse")
        inputs = self.processor(frames, return_tensors="pt", do_resize=False, do_center_crop=False)
        pixels = inputs["pixel_values"].to(self.device)
        context = torch.autocast("cuda", dtype=torch.float16) if self.device == "cuda" else nullcontext()
        with context:
            sequence = self.model.videomae(pixel_values=pixels).last_hidden_state
            # Exactly the pooling used by the released Transformers classifier.
            feature = self.model.fc_norm(sequence.mean(1)) if self.model.fc_norm is not None else sequence[:, 0]
            logits = self.model.classifier(feature)
        feature = feature[0].float().cpu().numpy()
        logits = logits[0].float().cpu().numpy()
        metadata.update({"frames": self.num_frames, "order": order, "encoder": self.fingerprint})
        if use_cache:
            np.savez_compressed(cache, feature=feature, logits=logits, metadata=json.dumps(metadata))
        return feature, logits, metadata, False
