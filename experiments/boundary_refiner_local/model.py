"""Query-conditioned local boundary proposals, not a semantic verification model."""

import torch
from torch import nn
from torch.nn import functional as F


class BoundaryRefiner(nn.Module):
    def __init__(self, variant="film", hidden=128, layers=2, dropout=.1):
        super().__init__()
        if variant not in ("film", "cross", "no_query", "cross_delta"):
            raise ValueError(variant)
        self.variant = variant
        self.visual = nn.Sequential(nn.Linear(512, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.position = nn.Sequential(nn.Linear(7, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.film = nn.Linear(512, hidden * 2)
        if variant in ("cross", "cross_delta"):
            self.token_projection = nn.Sequential(nn.Linear(512, hidden), nn.LayerNorm(hidden))
            self.cross_attention = nn.MultiheadAttention(hidden, 4, dropout=dropout, batch_first=True)
            self.cross_norm = nn.LayerNorm(hidden)
        if variant == "cross_delta":
            # Neighbor differences give order-sensitive local change information;
            # dilated temporal paths cover both quick transitions and sustained acts.
            self.delta = nn.Sequential(nn.Linear(1024, hidden), nn.LayerNorm(hidden), nn.GELU())
            self.local = nn.Conv1d(hidden, hidden, 3, padding=1, groups=hidden)
            self.wide = nn.Conv1d(hidden, hidden, 3, padding=3, dilation=3, groups=hidden)
            self.mix = nn.Linear(hidden * 3, hidden)
        temporal = nn.TransformerEncoderLayer(hidden, 4, hidden * 4, dropout=dropout,
                                               activation="gelu", batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(temporal, layers, enable_nested_tensor=False)
        self.edge = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Linear(hidden, 2))
        self.relevance = nn.Linear(hidden, 1)
        self.similarity_projection = nn.Linear(1, hidden)

    def forward(self, batch):
        mask = batch["mask"]
        raw = F.normalize(batch["visual"], dim=-1) * mask[..., None]
        query = F.normalize(batch["query"], dim=-1)
        if self.variant == "no_query":
            query = torch.zeros_like(query)
        length = (batch["window"][:, 1] - batch["window"][:, 0]).clamp(min=.1)
        rel_t = (batch["times"] - batch["window"][:, :1]) / length[:, None]
        start, end = batch["coarse"][:, :1], batch["coarse"][:, 1:]
        position = torch.stack([rel_t, (batch["times"] - start) / length[:, None],
                                (batch["times"] - end) / length[:, None],
                                ((batch["times"] >= start) & (batch["times"] <= end)).float(),
                                ((end - start) / length[:, None]).expand_as(rel_t),
                                torch.sin(2 * torch.pi * rel_t), torch.cos(2 * torch.pi * rel_t)], dim=-1)
        gamma, beta = self.film(query).chunk(2, -1)
        visual = self.visual(raw)
        visual = visual * (1 + gamma[:, None]) + beta[:, None]
        similarity = (raw * query[:, None]).sum(-1)
        hidden = visual + self.position(position) + self.similarity_projection(similarity[..., None])
        if self.variant in ("cross", "cross_delta"):
            tokens = self.token_projection(F.normalize(batch["tokens"], dim=-1))
            cross, _ = self.cross_attention(hidden, tokens, tokens, key_padding_mask=~batch["token_mask"], need_weights=False)
            hidden = self.cross_norm(hidden + cross)
        if self.variant == "cross_delta":
            previous = F.pad(raw[:, :-1], (0, 0, 1, 0))
            following = F.pad(raw[:, 1:], (0, 0, 0, 1))
            changes = self.delta(torch.cat([raw - previous, following - raw], dim=-1))
            changes = changes * mask[..., None]
            local = self.local(changes.transpose(1, 2)).transpose(1, 2)
            wide = self.wide(changes.transpose(1, 2)).transpose(1, 2)
            hidden = hidden + self.mix(torch.cat([changes, local, wide], -1))
        hidden = self.temporal(hidden * mask[..., None], src_key_padding_mask=~mask)
        hidden = hidden * mask[..., None]
        left, right = F.pad(hidden, (0, 0, 1, 0)), F.pad(hidden, (0, 0, 0, 1))
        edge_hidden = self.edge[:-1](torch.cat([left, right, right - left], dim=-1))
        boundary = self.edge[-1](edge_hidden)
        start_logits = boundary[..., 0].masked_fill(~batch["edge_mask"], -1e4)
        end_logits = boundary[..., 1].masked_fill(~batch["edge_mask"], -1e4)
        return {"start": start_logits, "end": end_logits,
                "membership": self.relevance(hidden).squeeze(-1),
                "edge_hidden": edge_hidden, "temporal_hidden": hidden}


def objective(output, batch):
    edges, valid = batch["edges"], batch["edge_mask"]
    losses = []
    for i, name in enumerate(("start", "end")):
        # Soft supervision reflects a finite observation grid, not fabricated
        # exact subsecond annotations. Invalid padded edges have zero mass.
        target = -0.5 * ((edges - batch["target"][:, i:i + 1]) / .35).square()
        target = target.masked_fill(~valid, -1e4).softmax(-1)
        losses.append(-(target * output[name].log_softmax(-1)).sum(-1).mean())
    membership = F.binary_cross_entropy_with_logits(output["membership"], batch["membership"], reduction="none")
    membership = ((membership * batch["mask"]).sum(1) / batch["mask"].sum(1)).mean()
    loss = losses[0] + losses[1] + .5 * membership
    return loss, {"start": float(losses[0].detach()), "end": float(losses[1].detach()),
                  "membership": float(membership.detach())}


def joint_decode(output, batch, membership_weight=.2):
    start, end = output["start"].log_softmax(-1), output["end"].log_softmax(-1)
    size = start.shape[1]
    indices = torch.arange(size, device=start.device)
    i, j = indices[:, None], indices[None, :]
    legal = j > i
    prefix = F.pad(F.logsigmoid(output["membership"]), (1, 0)).cumsum(-1)
    mean_support = (prefix[:, None, :] - prefix[:, :, None]) / (j - i).clamp(min=1)
    score = start[:, :, None] + end[:, None, :] + membership_weight * mean_support
    legal = legal[None] & batch["edge_mask"][:, :, None] & batch["edge_mask"][:, None, :]
    score = score.masked_fill(~legal, -1e9)
    best = score.flatten(1).argmax(1)
    result = torch.stack([batch["edges"].gather(1, (best // size)[:, None]).squeeze(1),
                          batch["edges"].gather(1, (best % size)[:, None]).squeeze(1)], -1)
    return result
