"""Check retained output audio against its source interval; not semantic sound recognition."""

import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np


def decode_audio(path: Path, start: float, duration: float):
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-threads", "2",
               "-ss", str(start), "-i", str(path), "-t", str(duration), "-vn", "-map", "0:a:0",
               "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        return None
    return np.frombuffer(result.stdout, dtype="<f4").astype(np.float64)


def verify_export_audio(source: Path, source_start: float, duration: float, exported: Path):
    result = {"verified": False, "scope": "Technical source/output audio correspondence only, not new event or dialogue recognition"}
    x, y = decode_audio(source, source_start, duration), decode_audio(exported, 0., duration)
    if x is None or y is None or min(len(x), len(y)) < max(1, round((duration-.12)*16000)):
        return {**result, "reason": "missing_or_incomplete_audio"}
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return {**result, "reason": "invalid_waveform"}
    x, y = x-x.mean(), y-y.mean()
    energy_x, energy_y = float(x@x), float(y@y)
    if min(energy_x/max(1, len(x)), energy_y/max(1, len(y))) < 1e-14:
        return {**result, "reason": "insufficient_audio_energy_for_content_check"}
    size = 1 << (len(x)+len(y)-2).bit_length()
    correlation = np.fft.irfft(np.fft.rfft(x, size)*np.conj(np.fft.rfft(y, size)), size)
    lag = min(round(.08*16000), len(x)-1, len(y)-1)
    values = np.r_[correlation[:lag+1], correlation[-lag:]]
    similarity = float(values.max()/np.sqrt(energy_x*energy_y))
    rms_ratio = float(np.sqrt((energy_y/len(y))/(energy_x/len(x))))
    verified = similarity >= .90 and .5 <= rms_ratio <= 2.
    return {**result, "verified": verified, "reason": "matched" if verified else "audio_content_does_not_match_source",
            "normalized_correlation": similarity, "rms_ratio": rms_ratio, "alignment_tolerance_seconds": .08}
