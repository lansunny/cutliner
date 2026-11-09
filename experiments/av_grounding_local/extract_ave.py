"""Cache frozen CLIP/PANN features from real AVE media, without API calls."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import imageio_ffmpeg
import numpy as np
import torch

from audio_encoder import FrozenAudioEncoder
from prepare_ave import DATA
from public_zip import ROOT

CLIP_ROOT = ROOT.parent/"video_grounding_local"
VISUAL_PROFILE = "clip_vitb32_sha40d365_ffmpeg_bicubic_center224_1fps_v1"


def decode_media(path):
    executable = imageio_ffmpeg.get_ffmpeg_exe()
    base = [executable, "-nostdin", "-v", "error", "-threads", "2", "-i", str(path), "-t", "10"]
    visual = subprocess.run(base+["-an", "-vf", "fps=1,scale=224:224:flags=bicubic:force_original_aspect_ratio=increase,crop=224:224",
                                  "-threads", "2", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=45).stdout
    frame_bytes = 224*224*3
    if len(visual) != 10*frame_bytes:
        raise ValueError(f"Expected ten observed source frames, not synthetic padding: {path.name}, {len(visual)//frame_bytes}")
    audio = subprocess.run(base+["-vn", "-map", "0:a:0", "-ac", "1", "-ar", "32000", "-f", "f32le", "pipe:1"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=45).stdout
    waveform = np.frombuffer(audio, dtype="<f4").copy()
    if not 9.5*32000 <= len(waveform) <= 10.1*32000 or not np.isfinite(waveform).all():
        raise ValueError(f"Audio does not cover the annotated ten-second source: {path.name}")
    return np.frombuffer(visual, np.uint8).reshape(10, 224, 224, 3).copy(), waveform


class FrozenEventFeatures:
    def __init__(self, device="cuda"):
        self.device = device
        sys.path.insert(0, str(CLIP_ROOT/"vendor/UniVTG"))
        from run_on_video import clip
        path = CLIP_ROOT/"weights/ViT-B-32.pt"
        checksum = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024**2), b""):
                checksum.update(chunk)
        if checksum.hexdigest() != "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af":
            raise ValueError("Local CLIP weights differ from verified public source")
        self.vision, _ = clip.load(str(path), device=device, jit=False)
        self.vision.eval().requires_grad_(False)
        self.audio = FrozenAudioEncoder(device)
        self.mean = torch.tensor([.48145466, .4578275, .40821073], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([.26862954, .26130258, .27577711], device=device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def encode(self, decoded):
        frames, waveform = decoded
        pixels = torch.from_numpy(frames).permute(0, 3, 1, 2).to(self.device).float()/255
        visual = self.vision.encode_image((pixels-self.mean)/self.std).float().cpu().numpy()
        audio = self.audio.encode(waveform).numpy()
        if visual.shape != (10, 512) or audio.shape != (10, 512) or not np.isfinite(visual).all():
            raise ValueError("Unexpected encoder outputs")
        return visual, audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", choices=["train", "dev"], default=["train", "dev"])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    torch.set_num_threads(3)
    plan = json.loads((DATA/"split_plan.json").read_text(encoding="utf-8"))
    rows = [row for split in args.splits for row in plan["selected"][split][:args.limit]]
    folder = DATA/"features"
    folder.mkdir(exist_ok=True)
    pending = []
    for row in rows:
        path, media = folder/f"{row['video']}.npz", DATA/"videos"/f"{row['video']}.mp4"
        if not media.exists():
            raise FileNotFoundError(f"Frozen source media not downloaded: {media}")
        signature = hashlib.sha256(media.read_bytes()).hexdigest()
        if path.exists():
            with np.load(path, allow_pickle=False) as cache:
                if str(cache["visual_profile"]) != VISUAL_PROFILE or str(cache["audio_profile"]) != FrozenAudioEncoder.profile or str(cache["media_sha256"]) != signature:
                    raise ValueError("Cached feature source/profile differs")
        else:
            pending.append((row, media, signature, path))
    if not pending:
        print(f"Verified all {len(rows)} existing audiovisual feature caches", flush=True)
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = FrozenEventFeatures(device)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {i: executor.submit(decode_media, item[1]) for i, item in enumerate(pending[:4])}
        for i, (row, media, signature, path) in enumerate(pending):
            decoded = futures.pop(i).result()
            if i+4 < len(pending):
                futures[i+4] = executor.submit(decode_media, pending[i+4][1])
            visual, audio = encoder.encode(decoded)
            np.savez_compressed(path, visual=visual, audio=audio, media_sha256=signature,
                                visual_profile=VISUAL_PROFILE, audio_profile=FrozenAudioEncoder.profile)
            if (i+1)%20 == 0 or i+1 == len(pending):
                print(f"Frozen event features {i+1}/{len(pending)}; {(time.monotonic()-started)/(i+1):.2f} seconds/source", flush=True)


if __name__ == "__main__":
    main()
