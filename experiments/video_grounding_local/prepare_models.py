"""Download public inference weights only; no paid API or training data required."""

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
MODELS = [
    {
        "name": "clip_qd_detr_qvhighlight.ckpt",
        "url": "https://zenodo.org/records/13363606/files/clip_qd_detr_qvhighlight.ckpt?download=1",
        "algorithm": "md5",
        "digest": "d0b73515cdf96772ed1d6916102f0126",
    },
    {
        "name": "ViT-B-32.pt",
        "url": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
        "algorithm": "sha256",
        "digest": "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
    },
]


def digest(path, algorithm):
    checksum = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def download(spec):
    folder = ROOT / "weights"
    folder.mkdir(exist_ok=True)
    target = folder / spec["name"]
    if target.exists():
        if digest(target, spec["algorithm"]) != spec["digest"]:
            raise RuntimeError(f"Existing file has an unexpected checksum: {target}")
        print(f"Verified existing {target.name}", flush=True)
        return
    partial = target.with_suffix(target.suffix + ".part")
    downloaded = 0
    last_report = time.monotonic()
    print(f"Downloading {target.name}", flush=True)
    with requests.get(spec["url"], stream=True, timeout=(12, 60)) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with partial.open("wb") as stream:
            for chunk in response.iter_content(chunk_size=512 * 1024):
                if not chunk:
                    continue
                stream.write(chunk)
                downloaded += len(chunk)
                if time.monotonic() - last_report >= 10:
                    print(f"{target.name}: {downloaded / 1e6:.1f}/{total / 1e6:.1f} MB", flush=True)
                    last_report = time.monotonic()
    if digest(partial, spec["algorithm"]) != spec["digest"]:
        raise RuntimeError(f"Downloaded file checksum failed: {partial}")
    partial.replace(target)
    print(f"Verified {target.name}: {downloaded / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(download, MODELS))
