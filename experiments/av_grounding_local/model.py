"""Trainable query-conditioned temporal localizer, with explicit modality ablations."""

import torch
from torch import nn
from torch.nn import functional as F


class AVGrounder(nn.Module):
    def __init__(self, mode="visual", hidden=128, dropout=0.1):
        super().__init__()
        self.variant = mode
        if mode == "visual_wide":
            mode, hidden = "visual", 176  # More active parameters than either 128-wide AV model.
        if mode not in ("visual", "audio", "concat", "gated"):
            raise ValueError("Unknown modality ablation")
        self.mode, self.hidden = mode, hidden
        self.visual_projection = nn.Sequential(nn.Linear(512, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.audio_projection = nn.Sequential(nn.Linear(2048, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.query_projection = nn.Sequential(nn.Linear(512, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.concat_projection = nn.Linear(2 * hidden, hidden)
        self.gate = nn.Sequential(nn.Linear(5 * hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self.condition = nn.Sequential(nn.Linear(4 * hidden + 3, hidden), nn.LayerNorm(hidden), nn.GELU())
        layer = nn.TransformerEncoderLayer(hidden, 4, 2 * hidden, dropout, batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.relevance = nn.Linear(hidden, 1)
        self.boundaries = nn.Linear(hidden, 2)
        self.raw_clip_scale = nn.Parameter(torch.tensor(10.0))
        # Do not count unused ablation branches as trainable network capacity.
        for module, used in ((self.visual_projection, mode != "audio"),
                             (self.audio_projection, mode != "visual"),
                             (self.concat_projection, mode == "concat"), (self.gate, mode == "gated")):
            module.requires_grad_(used)

    def forward(self, batch, audio_override=None):
        visual = self.visual_projection(F.normalize(batch["visual"], dim=-1))
        raw_audio = batch["audio"] if audio_override is None else audio_override
        audio = self.audio_projection(F.normalize(raw_audio, dim=-1)) * batch["audio_mask"][..., None]
        query = self.query_projection(F.normalize(batch["query"], dim=-1))[:, None].expand_as(visual)
        if self.mode == "visual":
            fused, gate = visual, torch.zeros_like(batch["centers"])[..., None]
        elif self.mode == "audio":
            fused, gate = audio, torch.ones_like(batch["centers"])[..., None]
        elif self.mode == "concat":
            fused = self.concat_projection(torch.cat([visual, audio], dim=-1))
            gate = torch.full_like(batch["centers"], 0.5)[..., None]
        else:
            gate = self.gate(torch.cat([visual, audio, query, visual * query, audio * query], dim=-1)).sigmoid()
            gate = gate * batch["audio_mask"][..., None]
            fused = (1 - gate) * visual + gate * audio
        centers = batch["centers"] / batch["duration"][:, None]
        # Preserve the strong pretrained text/image similarity in every visual
        # variant, rather than forcing a new small model to relearn alignment.
        raw_similarity = (F.normalize(batch["visual"], dim=-1) *
                          F.normalize(batch["query"], dim=-1)[:, None]).sum(-1)
        if self.mode == "audio":
            raw_similarity = torch.zeros_like(raw_similarity)
        position = torch.stack([centers, (centers + 1 / batch["duration"][:, None]).clamp(max=1)], dim=-1)
        conditioned = self.condition(torch.cat([fused, query, fused * query, (fused - query).abs(), position,
                                                raw_similarity[..., None]], dim=-1))
        temporal = self.temporal(conditioned, src_key_padding_mask=~batch["mask"])
        return {"logits": self.relevance(temporal).squeeze(-1) + self.raw_clip_scale * raw_similarity,
                "distances": self.boundaries(temporal).sigmoid(),
                "audio_gate": gate.squeeze(-1)}


def objective(output, batch):
    mask, truth = batch["mask"], batch["inside"] > 0.5
    positive, negative = truth & mask, ~truth & mask
    positive_loss = (F.softplus(-output["logits"]) * positive).sum(1) / positive.sum(1).clamp(min=1)
    negative_loss = (F.softplus(output["logits"]) * negative).sum(1) / negative.sum(1).clamp(min=1)
    relevance = ((positive_loss + negative_loss) / 2).mean()
    boundary = F.smooth_l1_loss(output["distances"], batch["distances"], beta=0.05, reduction="none").mean(-1)
    boundary = ((boundary * positive).sum(1) / positive.sum(1).clamp(min=1)).mean()
    return relevance + 5 * boundary, {"relevance": float(relevance.detach()), "boundary": float(boundary.detach())}
