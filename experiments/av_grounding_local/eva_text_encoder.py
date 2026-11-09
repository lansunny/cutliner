"""Matching frozen EVA text tower, without constructing its billion-parameter vision tower.

Architecture matches HiREST's EVA_clip/eva_model.py TextTransformer (GELU, not
QuickGELU). Only already-verified local weights and an identical BPE are used.
"""

import hashlib
import importlib.util
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from prepare_eva_text import SHA256, TEXT
from public_zip import ROOT

BPE_SHA256 = "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"
TOKENIZER_FOLDER = ROOT.parent / "video_grounding_local/vendor/UniVTG/run_on_video/clip"


class TextBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1,self.ln_2 = nn.LayerNorm(768),nn.LayerNorm(768)
        self.attn = nn.MultiheadAttention(768,12,batch_first=True)
        self.mlp = nn.Sequential(OrderedDict([("c_fc",nn.Linear(768,3072)),("ln",nn.Identity()),
                                             ("gelu",nn.GELU()),("c_proj",nn.Linear(3072,768))]))

    def forward(self,x,mask):
        normalized = self.ln_1(x)
        x = x + self.attn(normalized,normalized,normalized,attn_mask=mask,need_weights=False)[0]
        return x + self.mlp(self.ln_2(x))


class MatchingTextTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList([TextBlock() for _ in range(12)])
        self.token_embedding = nn.Embedding(49408,768)
        self.positional_embedding = nn.Parameter(torch.empty(77,768))
        self.ln_final = nn.LayerNorm(768)
        self.text_projection = nn.Parameter(torch.empty(768,1024))
        self.logit_scale = nn.Parameter(torch.zeros(()))
        self.register_buffer("attn_mask",torch.full((77,77),float("-inf")).triu_(1),persistent=False)

    def forward(self,tokens):
        x = self.token_embedding(tokens)+self.positional_embedding
        for block in self.transformer.resblocks:
            x = block(x,self.attn_mask)
        x = self.ln_final(x)
        return x[torch.arange(len(tokens),device=tokens.device),tokens.argmax(-1)] @ self.text_projection


class FrozenEVAText:
    def __init__(self,device="auto"):
        self.device = "cuda" if device=="auto" and torch.cuda.is_available() else device
        if self.device=="auto":
            self.device="cpu"
        bpe = TOKENIZER_FOLDER/"bpe_simple_vocab_16e6.txt.gz"
        if hashlib.sha256(bpe.read_bytes()).hexdigest()!=BPE_SHA256:
            raise ValueError("Tokenizer BPE differs from the author's verified asset")
        spec = importlib.util.spec_from_file_location("verified_existing_clip_tokenizer",TOKENIZER_FOLDER/"simple_tokenizer.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.tokenizer = module.SimpleTokenizer(str(bpe))
        checkpoint = torch.load(TEXT,map_location="cpu",weights_only=True)
        if checkpoint["source_sha256"] != SHA256:
            raise ValueError("EVA text extraction source differs")
        self.model = MatchingTextTower()
        self.model.load_state_dict(checkpoint["state_dict"],strict=True)
        self.model.eval().requires_grad_(False).to(self.device)
        self.fingerprint = hashlib.sha256((SHA256+BPE_SHA256+"eva_text_gelu12_768_eot1024_fp32_v1").encode()).hexdigest()

    @torch.inference_mode()
    def encode(self,texts,batch_size=16):
        result = []
        for offset in range(0,len(texts),batch_size):
            tokens = torch.zeros(min(batch_size,len(texts)-offset),77,dtype=torch.long)
            for i,text in enumerate(texts[offset:offset+batch_size]):
                ids = [49406]+self.tokenizer.encode(text)+[49407]
                if len(ids)>77:
                    ids=ids[:77]
                    ids[-1]=49407
                tokens[i,:len(ids)] = torch.tensor(ids)
            values = self.model(tokens.to(self.device)).float()
            if not values.isfinite().all():
                raise ValueError("Non-finite frozen EVA text features")
            result.append(F.normalize(values,dim=-1).cpu().numpy())
        return np.concatenate(result) if result else np.empty((0,1024),np.float32)
