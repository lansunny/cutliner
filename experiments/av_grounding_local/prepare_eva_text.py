"""Fetch the author's matching EVA checkpoint, then retain only its text tensors."""

import hashlib
import json
import time

import requests
import torch

from public_zip import ROOT

FOLDER = ROOT / "weights/eva_g"
FULL = FOLDER / "eva_clip_psz14.pt"
TEXT = FOLDER / "text_only.pt"
SIZE = 2273012515
SHA256 = "fe1d3a90bb88c3816a8c708b8a16c11c93cc323ec9e7bc7a0e8b979449fa9785"
URL = "https://hf-mirror.com/BAAI/EVA/resolve/main/eva_clip_psz14.pt"


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4*1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    FOLDER.mkdir(parents=True,exist_ok=True)
    partial = FULL.with_suffix(".pt.download")
    if not FULL.exists():
        for attempt in range(3):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset == SIZE:
                break
            print(f"Downloading public matching EVA weights: {offset/1024**2:.1f}/{SIZE/1024**2:.1f} MiB",flush=True)
            try:
                with requests.get(URL,headers={"Range":f"bytes={offset}-{SIZE-1}","Accept-Encoding":"identity"},
                                  stream=True,timeout=(10,45)) as response:
                    response.raise_for_status()
                    if response.status_code == 206:
                        if response.headers.get("Content-Range") != f"bytes {offset}-{SIZE-1}/{SIZE}":
                            raise ValueError("EVA download Range mismatch")
                    elif response.status_code != 200 or offset:
                        raise ValueError("Server refused exact safe resume")
                    with partial.open("ab" if offset else "wb") as stream:
                        reported = offset
                        for chunk in response.iter_content(1024**2):
                            offset += len(chunk)
                            if offset > SIZE:
                                raise ValueError("Checkpoint exceeds expected size")
                            stream.write(chunk)
                            if offset-reported >= 128*1024**2:
                                print(f"EVA weight transfer {offset/1024**2:.1f} MiB",flush=True)
                                reported = offset
                break
            except requests.RequestException as error:
                status = error.response.status_code if error.response is not None else None
                if attempt == 2 or status is not None and status < 500:
                    raise
                time.sleep(2)
        if not partial.is_file() or partial.stat().st_size != SIZE or sha256(partial) != SHA256:
            raise ValueError("EVA checkpoint hash/size mismatch; no model loaded")
        partial.replace(FULL)
    if FULL.stat().st_size != SIZE or sha256(FULL) != SHA256:
        raise ValueError("Existing EVA checkpoint does not match source identity")
    if TEXT.exists():
        existing = torch.load(TEXT,map_location="cpu",weights_only=True)
        if existing.get("source_sha256") != SHA256:
            raise ValueError("Existing text extraction differs")
        print(f"Verified existing EVA text extraction: {TEXT}",flush=True)
        return
    checkpoint = torch.load(FULL,map_location="cpu",weights_only=True,mmap=True)
    for key in ("model","module","state_dict"):
        if isinstance(checkpoint,dict) and key in checkpoint and isinstance(checkpoint[key],dict):
            checkpoint = checkpoint[key]
            break
    if next(iter(checkpoint)).startswith("module."):
        checkpoint = {key[7:]:value for key,value in checkpoint.items()}
    text = {key.removeprefix("text."):value.float().clone() for key,value in checkpoint.items() if key.startswith("text.")}
    if not text or text["token_embedding.weight"].shape != (49408,768) or text["text_projection"].shape != (768,1024):
        raise ValueError(f"Unexpected matching EVA text keys/shapes: {list(text)[:8]}")
    torch.save({"source_sha256":SHA256,"state_dict":text,"architecture":"EVA g14 paired text, GELU, 12x768, 1024 projection"},TEXT)
    print(json.dumps({"text_weights":str(TEXT),"frozen_parameters":sum(value.numel() for value in text.values()),
                      "source_hash_verified":True,"full_vision_model_not_instantiated":True},indent=2),flush=True)


if __name__ == "__main__":
    main()
