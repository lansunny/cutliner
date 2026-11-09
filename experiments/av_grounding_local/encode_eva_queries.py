"""Prepare matching visual-language query features for train/dev only."""

import hashlib
import json

import numpy as np
import torch

from eva_text_encoder import FrozenEVAText
from prepare_speech import DATA
from speech_baselines import rows_for


def main():
    torch.set_num_threads(4)
    encoder = FrozenEVAText()
    for split in ("train","dev"):
        rows = rows_for(split)
        path = DATA/f"{split}_eva_query_features.npz"
        manifest_hash = hashlib.sha256((DATA/f"{split}.jsonl").read_bytes()).hexdigest()
        if path.exists():
            with np.load(path,allow_pickle=False) as old:
                if old["ids"].tolist()!=[row["qid"] for row in rows] or str(old["encoder"])!=encoder.fingerprint:
                    raise ValueError("Existing matched EVA cache differs")
            print(f"Verified existing EVA query features: {split}",flush=True)
            continue
        features = encoder.encode([row["query"] for row in rows])
        np.savez_compressed(path,features=features,ids=np.array([row["qid"] for row in rows]),
                            encoder=encoder.fingerprint,manifest_sha256=manifest_hash)
        print(json.dumps({"split":split,"queries":len(rows),"shape":list(features.shape),"encoder_frozen":True,
                          "heldout_not_accessed":True}),flush=True)


if __name__=="__main__":
    main()
