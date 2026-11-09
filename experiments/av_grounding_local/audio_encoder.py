"""Frozen PANN Cnn6 frontend for real audio, with checksum-verified public weights.

Architecture/checkpoint: qiuqiangkong/audioset_tagging_cnn Cnn6, Zenodo 3987831.
The task-specific audiovisual head is trained separately; this encoder is not.
"""

import hashlib
import time

import requests
import torch
from torch import nn
from torch.nn import functional as F

from public_zip import ROOT

WEIGHT = ROOT/"weights/panns_cnn6/Cnn6_mAP=0.343.pth"
MD5 = "e25e26b84585b14c7754c91e48efc9be"
URL = "https://zenodo.org/records/3987831/files/Cnn6_mAP%3D0.343.pth?download=1"


def prepare_weights():
    if WEIGHT.exists():
        if hashlib.md5(WEIGHT.read_bytes()).hexdigest() != MD5:
            raise ValueError("Existing Cnn6 checkpoint checksum differs")
        return
    WEIGHT.parent.mkdir(parents=True, exist_ok=True)
    partial = WEIGHT.with_suffix(".part")
    total = partial.stat().st_size if partial.exists() else 0
    if total > 23698321:
        raise ValueError("Partial checkpoint is larger than published source")
    last = time.monotonic()
    for attempt in range(4):
        total = partial.stat().st_size if partial.exists() else 0
        if total == 23698321:
            break
        headers = {"Accept-Encoding": "identity"}
        if total:
            headers["Range"] = f"bytes={total}-23698320"
        try:
            with requests.get(URL, headers=headers, stream=True, timeout=(10, 40)) as response:
                response.raise_for_status()
                if total and (response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {total}-23698320/23698321"):
                    raise ValueError("Refusing to append a non-matching download range")
                with partial.open("ab") as stream:
                    for chunk in response.iter_content(128*1024):
                        stream.write(chunk)
                        total += len(chunk)
                        if time.monotonic()-last > 10:
                            print(f"Cnn6 public checkpoint {total/1e6:.1f}/23.7 MB", flush=True)
                            last = time.monotonic()
            break
        except requests.RequestException as error:
            status = error.response.status_code if error.response is not None else None
            if attempt == 3 or status is not None and status < 500:
                raise
            print("Resuming incomplete public checkpoint transfer", flush=True)
            time.sleep(attempt+1)
    if total != 23698321 or hashlib.md5(partial.read_bytes()).hexdigest() != MD5:
        raise ValueError("Downloaded Cnn6 checkpoint failed published size/MD5")
    partial.replace(WEIGHT)
    print("Verified frozen Cnn6 checkpoint", flush=True)


class ConvStage(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, 5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        return F.avg_pool2d(F.relu(self.bn1(self.conv1(x))), 2)


class FrozenCnn6(nn.Module):
    def __init__(self):
        super().__init__()
        from torchlibrosa.stft import LogmelFilterBank, Spectrogram
        self.spectrogram_extractor = Spectrogram(n_fft=1024, hop_length=320, win_length=1024,
                                                window="hann", center=True, pad_mode="reflect", freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(sr=32000, n_fft=1024, n_mels=64, fmin=50, fmax=14000,
                                                ref=1., amin=1e-10, top_db=None, freeze_parameters=True)
        self.bn0 = nn.BatchNorm2d(64)
        for index, (a, b) in enumerate(((1, 64), (64, 128), (128, 256), (256, 512)), 1):
            setattr(self, f"conv_block{index}", ConvStage(a, b))
        self.fc1, self.fc_audioset = nn.Linear(512, 512), nn.Linear(512, 527)

    def forward(self, waveform):
        x = self.logmel_extractor(self.spectrogram_extractor(waveform))
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        for index in range(1, 5):
            x = getattr(self, f"conv_block{index}")(x)
        x = x.mean(-1)
        return F.relu(self.fc1(x.amax(-1)+x.mean(-1)))


class FrozenAudioEncoder:
    profile = "panns_cnn6_md5_e25e26_32k_2s_context_1s_centers_v1"

    def __init__(self, device="cuda"):
        prepare_weights()
        self.device = device
        self.model = FrozenCnn6()
        checkpoint = torch.load(WEIGHT, map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.eval().requires_grad_(False).to(device)

    @torch.inference_mode()
    def encode(self, waveform, seconds=10):
        waveform = torch.as_tensor(waveform, dtype=torch.float32)
        if waveform.ndim != 1 or not waveform.isfinite().all():
            raise ValueError("Expected finite mono 32kHz waveform")
        windows = []
        for i in range(seconds):
            start, end = round((i-.5)*32000), round((i+1.5)*32000)
            segment = waveform[max(0, start):min(len(waveform), end)]
            segment = F.pad(segment, (max(0, -start), max(0, end-len(waveform))))
            if len(segment) != 64000:
                raise ValueError("Audio context has unexpected length")
            windows.append(segment)
        result = self.model(torch.stack(windows).to(self.device)).float().cpu()
        if not result.isfinite().all():
            raise ValueError("Non-finite PANN features")
        return result


if __name__ == "__main__":
    prepare_weights()
