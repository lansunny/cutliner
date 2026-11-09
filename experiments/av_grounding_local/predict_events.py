"""Real-video local inference for the independently validated audiovisual event head."""

import argparse
import contextlib
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import torch
from torch.nn import functional as F

from evaluate_events import validate_frozen
from event_dataset import contiguous
from extract_ave import FrozenEventFeatures, decode_media
from model_events import EventGrounder, ScoreCalibration
from prepare_ave import DATA
from public_zip import ROOT

sys.path.insert(0, str(ROOT.parents[1]))
from mini_agent.video.audio_visual_events import canonical_events


def source_duration(video):
    reader = imageio_ffmpeg.read_frames(str(video))
    try:
        value = float(next(reader)["duration"])
    finally:
        reader.close()
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Source media has no finite duration")
    return value


def decode_window(video, start):
    if start == 0:
        return decode_media(video)
    executable = imageio_ffmpeg.get_ffmpeg_exe()
    base = [executable, "-nostdin", "-v", "error", "-threads", "2", "-ss", str(start), "-i", str(video), "-t", "10"]
    frames = subprocess.run(base+["-an", "-vf", "fps=1,scale=224:224:flags=bicubic:force_original_aspect_ratio=increase,crop=224:224",
                                  "-threads", "2", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=90).stdout
    audio = subprocess.run(base+["-vn", "-map", "0:a:0", "-ac", "1", "-ar", "32000", "-f", "f32le", "pipe:1"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=90).stdout
    waveform = np.frombuffer(audio, dtype="<f4").copy()
    if len(frames) != 10*224*224*3 or not 9.5*32000 <= len(waveform) <= 10.1*32000:
        raise ValueError("Requested ten-second context lacks sufficient observed media")
    return np.frombuffer(frames, np.uint8).reshape(10, 224, 224, 3).copy(), waveform


class EventRuntime:
    def __init__(self, folder, device):
        self.folder, self.device = Path(folder), device
        frozen = validate_frozen(self.folder)
        report = json.loads((self.folder/"heldout_results.json").read_text(encoding="utf-8"))
        for control in ("visual", "visual_wide", "static_late_fusion"):
            comparisons = report["paired_comparisons"][control]
            if any(comparisons[key]["paired_source_bootstrap95ci_pp"][0] <= 0
                   for key in ("frame_macro_f1", "event_query_R1_iou0.5")):
                raise ValueError("Independent paired audiovisual improvement has not been established")
        # Deployment uses the FIRST prespecified seed, not the best heldout run.
        self.seed = frozen["seeds"][0]
        self.mode = frozen["selected_mode"]
        filename = f"{self.mode}_seed{self.seed}.pt"
        checkpoint = torch.load(self.folder/filename, map_location="cpu", weights_only=True)
        for name in ("model_events.py", "audio_encoder.py", "extract_ave.py"):
            if checkpoint["config"]["source_sha256"][name] != hashlib.sha256((ROOT/name).read_bytes()).hexdigest():
                raise ValueError("Runtime feature/model source changed since head training")
        state = checkpoint["state_dict"]
        calibrations = []
        for prefix in ("visual_calibration.", "audio_calibration."):
            parameters = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
            calibration = ScoreCalibration(parameters["mean"], parameters["std"])
            calibration.load_state_dict(parameters)
            calibrations.append(calibration)
        self.model = EventGrounder(self.mode, *calibrations, checkpoint["alpha"])
        self.model.load_state_dict(state)
        self.model.eval().requires_grad_(False).to(device)
        self.threshold = frozen["thresholds"][filename.removesuffix(".pt")]
        with np.load(DATA/"native_classifiers.npz", allow_pickle=False) as data:
            self.classes = data["classes"].tolist()
            self.visual_text = torch.from_numpy(data["visual_text"].copy()).to(device)
            self.audio_weight = torch.from_numpy(data["audio_weight"].copy()).to(device)
            self.audio_bias = torch.from_numpy(data["audio_bias"].copy()).to(device)
        self.encoder = FrozenEventFeatures(device)

    @torch.inference_mode()
    def probabilities(self, decoded):
        raw_visual, raw_audio = [torch.from_numpy(value).to(self.device) for value in self.encoder.encode(decoded)]
        visual, audio = F.normalize(raw_visual, dim=-1), F.normalize(raw_audio, dim=-1)
        batch = {"visual": visual[None], "audio": audio[None],
                 "visual_prior": (visual@self.visual_text.T)[None],
                 "audio_prior": (raw_audio@self.audio_weight.T+self.audio_bias)[None]}
        return self.model(batch).sigmoid()[0].cpu().numpy()

    def inspect(self, video, start, end, classes):
        duration = source_duration(video)
        if duration < 9.5:
            raise ValueError("Source shorter than the trained ten-second context")
        if not 0 <= start < end <= duration+.1 or end-start > 10.000001:
            raise ValueError("Inspect a valid local interval of at most ten seconds")
        context_start = max(0., min(max(0., duration-10), (start+end)/2-5))
        probability = self.probabilities(decode_window(video, context_start))
        events = []
        for name in classes:
            values = probability[:, self.classes.index(name)]
            for a, b in contiguous(values >= self.threshold):
                event_start, event_end = context_start+a, min(context_start+b, duration)
                lo, hi = max(start, event_start), min(end, event_end)
                if lo < hi:
                    events.append({"event_class": name, "source_start_seconds": event_start, "source_end_seconds": event_end,
                                   "target_overlap_start_seconds": lo, "target_overlap_end_seconds": hi,
                                   "mean_score": float(values[a:b].mean()), "score_is_calibrated_probability": False})
        return {"status": "ok", "source_file": str(Path(video).resolve()), "requested_start_seconds": start,
                "requested_end_seconds": end, "context_start_seconds": context_start,
                "context_end_seconds": min(duration, context_start+10), "events": events,
                "event_classes": classes, "threshold": self.threshold, "model_mode": self.mode, "model_seed": self.seed,
                "timestamp_resolution_seconds": 1., "audio_context_seconds_per_bin": 2.,
                "certifies_full_query": False, "absence_not_proven": True,
                "scope": "有声且可见的28类事件活动；不识别台词内容、具体身份、否定条件或任意动作过程"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--start", type=float, default=0.)
    parser.add_argument("--end", type=float, default=10.)
    parser.add_argument("--classes-json", required=True)
    args = parser.parse_args()
    torch.set_num_threads(3)
    classes = canonical_events(json.loads(args.classes_json))
    with contextlib.redirect_stdout(sys.stderr):
        runtime = EventRuntime(args.run.resolve(), "cuda" if torch.cuda.is_available() else "cpu")
        result = runtime.inspect(args.video.resolve(), args.start, args.end, classes)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
