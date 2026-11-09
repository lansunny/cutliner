"""Train temporal localization on paired pretrained semantic evidence, not raw vectors."""

import torch
from torch import nn
from torch.nn import functional as F


class TemporalBlock(nn.Module):
    def __init__(self, hidden, dilation):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.conv = nn.Conv1d(hidden, hidden, 5, padding=2*dilation, dilation=dilation)
        self.point = nn.Conv1d(hidden, hidden, 1)

    def forward(self, x, mask):
        value = self.norm(x.transpose(1, 2)).transpose(1, 2)
        value = self.point(F.gelu(self.conv(value)))
        return (x + F.dropout(value, p=0.1, training=self.training))*mask


class CurveGrounder(nn.Module):
    def __init__(self, mode, prior, hidden=24):
        super().__init__()
        if mode not in ("visual", "speech", "fusion"):
            raise ValueError("Unknown evidence mode")
        self.mode, self.prior = mode, dict(prior)
        self.projection = nn.Conv1d(9, hidden, 1)
        self.blocks = nn.ModuleList(TemporalBlock(hidden, d) for d in (1, 2, 4, 8, 16))
        self.head = nn.Conv1d(hidden, 3, 1)
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([0., prior["left_seconds"]/30, prior["right_seconds"]/30]))

    def forward(self, batch, disable=None):
        x = batch["curves"].clone()
        if self.mode == "speech" or disable == "visual":
            x[:, :, [0, 2, 4, 6]] = 0
        if self.mode == "visual" or disable == "speech":
            x[:, :, [1, 3, 5, 7]] = 0
        alpha = self.prior["alpha"]
        prior = alpha*x[:, :, 2] + (1-alpha)*x[:, :, 3]
        mask = batch["mask"].unsqueeze(1)
        hidden = F.gelu(self.projection(x.transpose(1, 2)))*mask
        for block in self.blocks:
            hidden = block(hidden, mask)
        prediction = self.head(hidden).transpose(1, 2)
        return {"logits": (prior + prediction[:, :, 0]).masked_fill(~batch["mask"], -1e4),
                "distances": prediction[:, :, 1:]}


def curve_objective(output, batch):
    positive = batch["labels"] & batch["mask"]
    negative = ~batch["labels"] & batch["mask"]
    bce = F.binary_cross_entropy_with_logits(output["logits"], positive.float(), reduction="none")
    positive_loss = (bce*positive).sum(-1)/positive.sum(-1).clamp(min=1)
    negative_loss = (bce*negative).sum(-1)/negative.sum(-1).clamp(min=1)
    # Full-timeline contrast forces the peak into a human-labeled interval.
    logp = output["logits"].log_softmax(-1)
    location = -(logp*positive).sum(-1)/positive.sum(-1).clamp(min=1)
    regression = F.smooth_l1_loss(output["distances"], batch["targets"], reduction="none").mean(-1)
    regression = (regression*positive).sum(-1)/positive.sum(-1).clamp(min=1)
    return (0.5*(positive_loss+negative_loss) + 0.25*location + 2*regression).mean()
