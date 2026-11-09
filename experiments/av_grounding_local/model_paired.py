"""Query-conditioned audiovisual temporal network in each encoder's own paired space."""

import torch
from torch import nn
from torch.nn import functional as F

from model_curves import TemporalBlock


class PairedGrounder(nn.Module):
    def __init__(self, mode, prior, hidden=64):
        super().__init__()
        self.mode, self.prior = mode, dict(prior)
        self.visual = nn.Sequential(nn.Linear(1024, 32), nn.LayerNorm(32), nn.GELU())
        self.speech = nn.Sequential(nn.Linear(384, 32), nn.LayerNorm(32), nn.GELU())
        self.projection = nn.Conv1d(73, hidden, 1)
        self.blocks = nn.ModuleList(TemporalBlock(hidden, dilation) for dilation in (1, 2, 4, 8, 16))
        self.head = nn.Conv1d(hidden, 3, 1)
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([0., prior["left_seconds"]/30, prior["right_seconds"]/30]))

    def forward(self, batch, disable=None):
        curves = batch["curves"].clone()
        b = len(curves)
        visual_active = torch.ones(b, 1, 1, device=curves.device)
        speech_active = torch.ones_like(visual_active)
        if self.mode == "speech" or disable == "visual":
            visual_active.zero_()
        if self.mode == "visual" or disable == "speech":
            speech_active.zero_()
        if self.training and self.mode == "fusion":
            draw = torch.rand(b, 1, 1, device=curves.device)
            visual_active *= draw >= .1
            speech_active *= (draw < .1) | (draw >= .2)
        curves[:, :, [0, 2, 4, 6]] *= visual_active
        curves[:, :, [1, 3, 5, 7]] *= speech_active
        # Elementwise products retain which semantic dimensions match. No
        # projection is asked to align EVA vision with the unrelated MiniLM space.
        v = self.visual(F.dropout(batch["visual"]*batch["visual_query"][:, None, :]*1024, .2, self.training))*visual_active
        s = self.speech(F.dropout(batch["speech"]*batch["speech_query"][:, None, :]*384, .2, self.training))*speech_active
        mask = batch["mask"].unsqueeze(1)
        hidden = F.gelu(self.projection(torch.cat([v, s, curves], dim=-1).transpose(1, 2)))*mask
        for block in self.blocks:
            hidden = block(hidden, mask)
        out = self.head(hidden).transpose(1, 2)
        prior = self.prior["alpha"]*curves[:, :, 2]+(1-self.prior["alpha"])*curves[:, :, 3]
        return {"logits": (prior+out[:, :, 0]).masked_fill(~batch["mask"], -1e4), "distances": out[:, :, 1:]}
