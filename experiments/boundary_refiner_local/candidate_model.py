"""Query-conditioned proposal-set rankers with capacity/metadata controls."""

import torch
from torch import nn
from torch.nn import functional as F


class CandidateRanker(nn.Module):
    def __init__(self, variant="joint", hidden=128):
        super().__init__()
        if variant not in ("independent", "independent_deep", "joint", "metadata_only"):
            raise ValueError(variant)
        self.variant = variant
        self.visual = nn.Sequential(nn.Linear(128, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.meta = nn.Sequential(nn.Linear(20, hidden), nn.GELU())
        self.query = nn.Sequential(nn.Linear(512, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.condition = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.LayerNorm(hidden), nn.GELU())
        if variant == "joint":
            layer = nn.TransformerEncoderLayer(hidden, 4, hidden * 4, dropout=.1,
                    activation="gelu", batch_first=True, norm_first=True)
            self.trunk = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        elif variant == "independent_deep":
            # ~same active capacity as two Transformer layers, no cross-node mixing.
            self.trunk = nn.Sequential(nn.Linear(hidden, 512), nn.GELU(), nn.Dropout(.1),
                     nn.Linear(512, 512), nn.GELU(), nn.Dropout(.1), nn.Linear(512, hidden), nn.LayerNorm(hidden))
        else:
            self.trunk = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(.1))
        self.score = nn.Linear(hidden, 1)
        self.iou = nn.Linear(hidden, 1)
        self.complete = nn.Linear(hidden, 1)

    def forward(self, batch):
        features, metadata, query = batch["embedding"], batch["meta"], batch["query"]
        if self.variant == "metadata_only":
            features, query, metadata = torch.zeros_like(features), torch.zeros_like(query), metadata.clone()
            # Retain only coarse ranker score and geometry. Erase learned local
            # quality/uncertainty/CLIP semantic scores as well as neural embeddings.
            metadata[..., 11:] = 0
        v = self.visual(features) + self.meta(metadata)
        q = self.query(F.normalize(query, dim=-1))
        expanded = q[:, None].expand_as(v)
        h = self.condition(torch.cat([v, expanded, v * expanded], -1))
        if self.variant == "joint":
            x = torch.cat([q[:, None], h], 1)
            valid = F.pad(batch["mask"], (1, 0), value=True)
            h = self.trunk(x, src_key_padding_mask=~valid)[:, 1:]
        else:
            h = self.trunk(h)
        return {"score": self.score(h).squeeze(-1).masked_fill(~batch["mask"], -1e4),
                "iou": self.iou(h).squeeze(-1).sigmoid(), "complete": self.complete(h).squeeze(-1)}


def ranking_objective(output, batch):
    valid = batch["mask"]
    reward = batch["true_iou"] + .25 * batch["true_complete"]
    target = (reward / .15).masked_fill(~valid, -1e4).softmax(-1)
    usable = (batch["true_iou"] * valid).max(-1).values > 0
    ce = -(target * output["score"].log_softmax(-1)).sum(-1)
    ce = (ce * usable).sum() / usable.sum().clamp(min=1)
    iou = ((output["iou"] - batch["true_iou"]).square() * valid).sum() / valid.sum()
    complete = F.binary_cross_entropy_with_logits(output["complete"], batch["true_complete"], reduction="none")
    complete = (complete * valid).sum() / valid.sum()
    return ce + 2 * iou + .5 * complete


def collate_candidate_sets(rows):
    count, width = len(rows), max(len(x["rank"]) for x in rows)
    batch = {"embedding": torch.zeros(count, width, 128), "meta": torch.zeros(count, width, 20),
             "prediction": torch.zeros(count, width, 2), "coarse": torch.zeros(count, width, 2),
             "true_iou": torch.zeros(count, width), "true_complete": torch.zeros(count, width),
             "rank": torch.zeros(count, width, dtype=torch.long), "mask": torch.zeros(count, width, dtype=torch.bool)}
    for key in ("query", "target"):
        batch[key] = torch.stack([r[key] for r in rows])
    for i, row in enumerate(rows):
        n = len(row["rank"])
        for key in ("embedding", "meta", "prediction", "coarse", "true_iou", "true_complete", "rank"):
            batch[key][i, :n] = row[key]
        batch["mask"][i, :n] = True
    for key in ("case_id", "video_id"):
        batch[key] = [x[key] for x in rows]
    return batch
