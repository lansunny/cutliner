"""Frozen MiniLM query encoder compatible with the author's speech embeddings."""

import hashlib

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from prepare_text_encoder import FILES, WEIGHTS, validate


class FrozenSentenceEncoder:
    def __init__(self, device="auto"):
        for name, expected in FILES.items():
            validate(WEIGHTS / name, expected)
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if self.device == "auto":
            self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(WEIGHTS, local_files_only=True)
        self.model = AutoModel.from_pretrained(WEIGHTS, local_files_only=True)
        if self.model.config.hidden_size != 384:
            raise ValueError("Expected the author's 384-dimensional MiniLM sentence encoder")
        self.model.eval().requires_grad_(False).to(self.device)
        self.fingerprint = hashlib.sha256((FILES["model.safetensors"][2] + FILES["tokenizer.json"][2] + "mean_pool_max256_l2_v1").encode()).hexdigest()

    @torch.inference_mode()
    def encode(self, texts, batch_size=64):
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        result = []
        for offset in range(0, len(texts), batch_size):
            tokens = self.tokenizer(texts[offset:offset+batch_size], padding=True, truncation=True,
                                    max_length=256, return_tensors="pt").to(self.device)
            output = self.model(**tokens).last_hidden_state
            mask = tokens["attention_mask"][..., None].to(output.dtype)
            pooled = (output * mask).sum(1) / mask.sum(1).clamp(min=1)
            result.append(F.normalize(pooled, dim=-1).cpu().float().numpy())
        return np.concatenate(result)
