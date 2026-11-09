"""Fetch only the three public model files, without credentials or remote code."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights" / "videomae-small-ssv2"
REPOSITORY = "https://huggingface.co/MCG-NJU/videomae-small-finetuned-ssv2"
WEIGHT_SHA256 = "e3a1aa3caef3828c9c3bc3240f7d2023d037962c0952db6d702779d05bbe5e09"
FILES = ("config.json", "preprocessor_config.json", "pytorch_model.bin")


def sha256_file(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def validate_files():
    missing = [name for name in FILES if not (WEIGHTS / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {missing}. Put official model files in {WEIGHTS}")
    if sha256_file(WEIGHTS / "pytorch_model.bin") != WEIGHT_SHA256:
        raise ValueError("Weight checksum differs from the official published SHA256; refusing to load")
    config = json.loads((WEIGHTS / "config.json").read_text(encoding="utf-8"))
    if config.get("hidden_size") != 384 or len(config.get("id2label", {})) != 174:
        raise ValueError("Expected the 384-dimensional / 174-class VideoMAE-Small SSV2 checkpoint")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["official", "public-mirror"], default="official",
                        help="Mirror transfers public files without account tokens; weight SHA256 must match the official repository")
    args = parser.parse_args()
    endpoint = REPOSITORY if args.source == "official" else REPOSITORY.replace("https://huggingface.co", "https://hf-mirror.com")
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = WEIGHTS / name
        if target.exists():
            print(f"Using existing {name}", flush=True)
            continue
        print(f"Downloading {name}; transfer source={args.source}", flush=True)
        partial = target.with_suffix(target.suffix + ".part")
        count, reported = 0, time.monotonic()
        try:
            with requests.get(f"{endpoint}/resolve/main/{name}", stream=True, timeout=(8, 45)) as response:
                response.raise_for_status()
                with partial.open("wb") as stream:
                    for chunk in response.iter_content(512 * 1024):
                        if chunk:
                            stream.write(chunk)
                            count += len(chunk)
                        if time.monotonic() - reported > 10:
                            print(f"{name}: {count / 1e6:.1f} MB", flush=True)
                            reported = time.monotonic()
            if name == "pytorch_model.bin" and sha256_file(partial) != WEIGHT_SHA256:
                raise ValueError("Downloaded weights failed checksum verification")
            if name.endswith(".json"):
                json.loads(partial.read_text(encoding="utf-8"))
            partial.replace(target)
        except requests.RequestException as error:
            raise SystemExit(f"Official download unavailable ({type(error).__name__}). "
                             f"Download {FILES} manually from {REPOSITORY}/tree/main into {WEIGHTS}. "
                             "No credentials were requested or changed.") from None
    validate_files()
    provenance = {"official_repository": REPOSITORY, "transfer_endpoint": endpoint,
                  "weight_sha256": WEIGHT_SHA256,
                  "note": "Public model transfer only; no account credentials supplied"}
    (WEIGHTS / "download_source.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print("Official weights and configuration verified; ready for inference")


if __name__ == "__main__":
    main()
