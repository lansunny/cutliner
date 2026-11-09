"""Fetch the small frozen sentence encoder matching HiREST's ASR feature source."""

import hashlib
from pathlib import Path

import requests

from public_zip import ROOT

WEIGHTS = ROOT / "weights/all-minilm-l6-v2"
BASE = "https://hf-mirror.com/sentence-transformers/all-MiniLM-L6-v2/resolve/main/"
# Published public Git/LFS identities observed before the experiment.
FILES = {
    "config.json": (612, "git", "72b987fd805cfa2b58c4c8c952b274a11bfd5a00"),
    "model.safetensors": (90868376, "sha256", "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"),
    "special_tokens_map.json": (112, "git", "e7b0375001f109a6b8873d756ad4f7bbb15fbaa5"),
    "tokenizer.json": (466247, "git", "cb202bfe2e3c98645018a6d12f182a434c9d3e02"),
    "tokenizer_config.json": (350, "git", "c79f2b6a0cea6f4b564fed1938984bace9d30ff0"),
    "vocab.txt": (231508, "git", "fb140275c155a9c7c5a3b3e0e77a9e839594a938"),
}


def validate(path, expected):
    size, algorithm, checksum = expected
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest() if algorithm == "sha256" else hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
    if len(payload) != size or actual != checksum:
        raise ValueError(f"Public text encoder checksum mismatch: {path.name}")


def main():
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        target = WEIGHTS / name
        if not target.exists():
            print(f"Downloading frozen text encoder: {name}", flush=True)
            partial = target.with_suffix(target.suffix + ".download")
            with requests.get(BASE + name, stream=True, timeout=(10, 40)) as response:
                response.raise_for_status()
                with partial.open("wb") as stream:
                    for chunk in response.iter_content(512*1024):
                        stream.write(chunk)
            validate(partial, expected)
            partial.replace(target)
        validate(target, expected)
    print(f"Frozen MiniLM weights and tokenizer verified: {WEIGHTS}", flush=True)


if __name__ == "__main__":
    main()
