"""Structured interval quality/completeness; not generic semantic verification.

Coverage supervision refers to the annotated target interval, not exhaustive
absence/presence of all possible actions in a scene. No NULL head is trained.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from model import BoundaryRefiner


def interval_grid(batch):
    edges = batch["edges"]
    a, b = edges[:, :, None], edges[:, None, :]
    valid = batch["edge_mask"][:, :, None] & batch["edge_mask"][:, None, :] & (b > a)
    return a, b, valid


class IntervalQualityRefiner(nn.Module):
    def __init__(self, hidden=128, rank=16, backbone_variant="film"):
        super().__init__()
        self.rank = rank
        if backbone_variant not in ("film", "no_query"):
            raise ValueError(backbone_variant)
        self.backbone = BoundaryRefiner(backbone_variant, hidden=hidden)
        self.left = nn.Linear(hidden, rank * 2)
        self.right = nn.Linear(hidden, rank * 2)
        self.content = nn.Linear(hidden, rank)
        self.content_score = nn.Linear(rank, 2)
        self.geometry = nn.Sequential(nn.Linear(7, 32), nn.GELU(), nn.Linear(32, 2))
        self.outside = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Linear(hidden, 2))

    def forward(self, batch):
        output = self.backbone(batch)
        edge, temporal = output["edge_hidden"], output["temporal_hidden"]
        left = self.left(edge).reshape(*edge.shape[:2], 2, self.rank)
        right = self.right(edge).reshape(*edge.shape[:2], 2, self.rank)
        pair = torch.einsum("bikr,bjkr->bijk", left, right) / math.sqrt(self.rank)
        a, b, valid = interval_grid(batch)
        lo, hi = batch["window"][:, 0, None, None], batch["window"][:, 1, None, None]
        scale = (hi - lo).clamp(min=.1)
        ca, cb = batch["coarse"][:, 0, None, None], batch["coarse"][:, 1, None, None]
        overlap = (torch.minimum(b, cb) - torch.maximum(a, ca)).clamp(min=0)
        width = (b - a).clamp(min=.01)
        union = (torch.maximum(b, cb) - torch.minimum(a, ca)).clamp(min=.01)
        shape = pair.shape[:3]
        geometry = torch.stack([((a - lo) / scale).expand(shape), ((b - lo) / scale).expand(shape),
                                (b - a).expand(shape) / scale, overlap / union,
                                overlap / (cb - ca).clamp(min=.01), overlap / width,
                                ((a + b - ca - cb) / (2 * scale)).abs().expand(shape)], -1)
        content = self.content(temporal) * batch["mask"][..., None]
        prefix = F.pad(content.cumsum(1), (0, 0, 1, 0))
        n = torch.arange(edge.shape[1], device=edge.device)
        counts = (n[None, :] - n[:, None]).clamp(min=1)
        mean_content = (prefix[:, None] - prefix[:, :, None]) / counts[None, :, :, None]
        quality = pair + self.geometry(geometry) + self.content_score(mean_content)
        count = batch["mask"].sum(1)
        pooled = temporal.sum(1) / count[:, None]
        last = temporal[torch.arange(len(count), device=count.device), count - 1]
        output["outside"] = self.outside(torch.cat([pooled, temporal[:, 0], last], -1))
        output["iou_quality"] = quality[..., 0].masked_fill(~valid, -1e4)
        output["coverage_quality"] = quality[..., 1].masked_fill(~valid, -1e4)
        output["pair_mask"] = valid
        return output


def masked_boundary_objective(output, batch):
    valid_edges, edges = batch["edge_mask"], batch["edges"]
    boundary_losses = []
    for i, key in enumerate(("start", "end")):
        truth = batch["target"][:, i:i + 1]
        observed = (truth[:, 0] >= batch["window"][:, 0]) & (truth[:, 0] <= batch["window"][:, 1])
        distribution = (-.5 * ((edges - truth) / .35).square()).masked_fill(~valid_edges, -1e4).softmax(-1)
        loss = -(distribution * output[key].log_softmax(-1)).sum(-1)
        # Missing boundaries are NOT clamped to the crop edge and relabeled complete.
        boundary_losses.append((loss * observed).sum() / observed.sum().clamp(min=1))
    member = F.binary_cross_entropy_with_logits(output["membership"], batch["membership"], reduction="none")
    member = ((member * batch["mask"]).sum(1) / batch["mask"].sum(1)).mean()
    return sum(boundary_losses) + .5 * member, {"membership": float(member.detach())}


def interval_objective(output, batch):
    boundary, _ = masked_boundary_objective(output, batch)
    a, b, valid = interval_grid(batch)
    s, e = batch["target"][:, 0, None, None], batch["target"][:, 1, None, None]
    intersection = (torch.minimum(b, e) - torch.maximum(a, s)).clamp(min=0)
    iou = intersection / (torch.maximum(b, e) - torch.minimum(a, s)).clamp(min=.01)
    coverage = intersection / (e - s)
    denom = valid.sum((1, 2)).clamp(min=1)
    quality = (((output["iou_quality"].sigmoid() - iou).square() * valid).sum((1, 2)) / denom).mean()
    complete = F.binary_cross_entropy_with_logits(output["coverage_quality"], (coverage >= .9).float(), reduction="none")
    complete = ((complete * valid).sum((1, 2)) / denom).mean()
    missing = torch.stack([batch["target"][:, 0] < batch["window"][:, 0],
                           batch["target"][:, 1] > batch["window"][:, 1]], -1).float()
    outside = F.binary_cross_entropy_with_logits(output["outside"], missing)
    total = boundary + 4 * quality + complete + .5 * outside
    return total, {"quality": float(quality.detach()), "complete": float(complete.detach()),
                   "outside": float(outside.detach())}


def quality_decode(output, batch, coverage_weight=.25, margin=.02, outside_threshold=.5, fallback=True):
    iou = output["iou_quality"].sigmoid()
    cover = output["coverage_quality"].sigmoid()
    prior = output["start"].log_softmax(-1)[:, :, None] + output["end"].log_softmax(-1)[:, None, :]
    score = (iou + coverage_weight * cover + .05 * prior).masked_fill(~output["pair_mask"], -1e4)
    size = score.shape[-1]
    best = score.flatten(1).argmax(1)
    i, j = best // size, best % size
    rows = torch.arange(len(best), device=best.device)
    proposal = torch.stack([batch["edges"][rows, i], batch["edges"][rows, j]], -1)
    # Evaluate the unchanged coarse hypothesis on the same map, without true labels.
    a = (batch["edges"] - batch["coarse"][:, :1]).abs().masked_fill(~batch["edge_mask"], 1e6).argmin(-1)
    b = (batch["edges"] - batch["coarse"][:, 1:]).abs().masked_fill(~batch["edge_mask"], 1e6).argmin(-1)
    b = torch.maximum(b, a + 1).minimum(batch["mask"].sum(-1))
    a = torch.minimum(a, b - 1)
    missing = output["outside"].sigmoid().max(-1).values
    use = ((iou[rows, i, j] - iou[rows, a, b]) >= margin) & (cover[rows, i, j] >= cover[rows, a, b])
    use = use & (missing < outside_threshold)
    if not fallback:
        use = torch.ones_like(use)
    prediction = torch.where(use[:, None], proposal, batch["coarse"])
    return prediction, {"intervened": use, "outside_probability": missing,
                        "predicted_coverage_probability": cover[rows, i, j],
                        "predicted_iou": iou[rows, i, j]}
