"""Small trainable action-conditioned head; the video backbone stays frozen."""

import torch
from torch import nn

from actions import ACTIONS

ACTION_ORDER = list(ACTIONS)


class ActionMatchingHead(nn.Module):
    def __init__(self, feature_dim=384, action_dim=32, hidden_dim=128):
        super().__init__()
        self.action_embedding = nn.Embedding(len(ACTION_ORDER), action_dim)
        self.match = nn.Sequential(
            nn.LayerNorm(feature_dim + action_dim),
            nn.Linear(feature_dim + action_dim, hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features, actions):
        conditions = self.action_embedding(actions)
        return self.match(torch.cat([features, conditions], dim=-1)).squeeze(-1)


def load_head(path, fingerprint, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["encoder_fingerprint"] != fingerprint or checkpoint["actions"] != ACTION_ORDER:
        raise ValueError("Head weights do not match this encoder or action vocabulary")
    model = ActionMatchingHead(feature_dim=checkpoint["feature_dim"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval().to(device), checkpoint
