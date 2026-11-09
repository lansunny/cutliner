"""Frozen semantic classifier priors plus small supervised audiovisual temporal heads."""

import torch
from torch import nn
from torch.nn import functional as F


class ScoreCalibration(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", mean.clone())
        self.register_buffer("std", std.clone().clamp(min=.01))
        self.log_scale = nn.Parameter(torch.zeros(28))
        self.bias = nn.Parameter(torch.full((28,), -2.))

    def forward(self, values):
        return (values-self.mean)/self.std*self.log_scale.clamp(-2, 2).exp()+self.bias


class EventTemporalBlock(nn.Module):
    def __init__(self, hidden, dilation):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.temporal = nn.Conv1d(hidden, hidden, 3, padding=dilation, dilation=dilation)
        self.point = nn.Conv1d(hidden, hidden, 1)

    def forward(self, x):
        value = self.norm(x.transpose(1, 2)).transpose(1, 2)
        return x+F.dropout(self.point(F.gelu(self.temporal(value))), .2, self.training)


class EventGrounder(nn.Module):
    def __init__(self, mode, visual_calibration, audio_calibration, alpha=.5):
        super().__init__()
        if mode not in ("visual", "visual_wide", "audio", "concat", "gated"):
            raise ValueError("Unknown event mode")
        self.mode, self.alpha = mode, alpha
        self.visual_calibration, self.audio_calibration = visual_calibration, audio_calibration
        self.visual_calibration.requires_grad_(False)
        self.audio_calibration.requires_grad_(False)
        hidden = 96 if mode == "visual_wide" else 64
        self.visual = nn.Sequential(nn.Linear(512, hidden//2), nn.LayerNorm(hidden//2), nn.GELU())
        self.audio = nn.Sequential(nn.Linear(512, hidden//2), nn.LayerNorm(hidden//2), nn.GELU())
        self.gate = nn.Linear(hidden, hidden//2) if mode == "gated" else None
        self.projection = nn.Conv1d(hidden+56, hidden, 1)
        self.blocks = nn.ModuleList(EventTemporalBlock(hidden, d) for d in (1, 2))
        self.head = nn.Conv1d(hidden, 28, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        if mode in ("visual", "visual_wide"):
            self.audio.requires_grad_(False)
        if mode == "audio":
            self.visual.requires_grad_(False)

    def forward(self, batch, disable=None):
        v = self.visual(F.dropout(batch["visual"], .2, self.training))
        a = self.audio(F.dropout(batch["audio"], .2, self.training))
        pv = self.visual_calibration(batch["visual_prior"])
        pa = self.audio_calibration(batch["audio_prior"])
        use_v = self.mode != "audio" and disable != "visual"
        use_a = self.mode not in ("visual", "visual_wide") and disable != "audio"
        if not use_v:
            v, pv = v*0, pv*0
        if not use_a:
            a, pa = a*0, pa*0
        alpha = self.alpha if use_v and use_a else float(use_v)
        prior = alpha*pv+(1-alpha)*pa
        if self.gate is not None:
            gate = self.gate(torch.cat([v, a], dim=-1)).sigmoid()
            v, a = v*gate, a*(1-gate)
        hidden = F.gelu(self.projection(torch.cat([v, a, pv, pa], dim=-1).transpose(1, 2)))
        for block in self.blocks:
            hidden = block(hidden)
        return prior+self.head(hidden).transpose(1, 2)
