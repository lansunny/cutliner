"""Shared guard for the independently held-out boundary test stage."""

import json
from pathlib import Path

from prepare_data import file_hash


def load_frozen_protocol(path, root):
    if path is None:
        raise ValueError("Final-test access requires an already frozen inference protocol")
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    if protocol["state"] != "frozen" or Path(protocol["dataset_root"]).resolve() != Path(root).resolve():
        raise ValueError("Unfrozen protocol or different dataset root")
    for split, expected in protocol["manifest_sha256"].items():
        actual = file_hash(Path(root) / "manifests" / f"{split}.complete.jsonl")
        if actual != expected:
            raise ValueError(f"Frozen {split} manifest changed")
    for pipeline in protocol["pipelines"]:
        for key in ("local", "ranker"):
            if file_hash(Path(pipeline[key + "_checkpoint"])) != pipeline[key + "_sha256"]:
                raise ValueError(f"Frozen {key} checkpoint changed")
    here = Path(__file__).resolve().parent
    weights = here.parent / "video_grounding_local" / "weights"
    if file_hash(weights / "ViT-B-32.pt") != protocol["clip_sha256"]:
        raise ValueError("Frozen CLIP encoder changed")
    if file_hash(weights / "clip_qd_detr_qvhighlight.ckpt") != protocol["qd_sha256"]:
        raise ValueError("Frozen coarse retriever changed")
    for name, expected in protocol["source_sha256"].items():
        if file_hash(here / name) != expected:
            raise ValueError(f"Inference source changed after protocol freeze: {name}")
    return protocol
