"""Constrained paired-space metric adaptation with interval-as-bag supervision."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def standardize(values, mask):
    count = mask.sum(-1, keepdim=True).clamp(min=1)
    mean = (values*mask).sum(-1, keepdim=True)/count
    variance = ((values-mean)**2*mask).sum(-1, keepdim=True)/count
    return (values-mean)/variance.clamp(min=.0001).sqrt()*mask


def weighted_cosine(values, query, log_weight):
    weight = log_weight.clamp(-2, 2).exp()
    numerator = (values*query[:, None, :]*weight).sum(-1)
    denominator = ((values.square()*weight).sum(-1)*(query.square()*weight).sum(-1)[:, None]).clamp(min=1e-14).sqrt()
    return numerator/denominator


class MetricMILGrounder(nn.Module):
    def __init__(self, mode, prior):
        super().__init__()
        self.mode, self.prior = mode, dict(prior)
        self.visual_weight = nn.Parameter(torch.zeros(1024))
        self.speech_weight = nn.Parameter(torch.zeros(384))
        alpha = min(.999, max(.001, prior["alpha"]))
        self.gate_logit = nn.Parameter(torch.tensor(math.log(alpha/(1-alpha))))
        self.lags = (-16, -8, -4, -2, 2, 4, 8, 16)
        self.lag_weights = nn.Parameter(torch.zeros(len(self.lags)))

    def forward(self, batch, disable=None):
        visual = standardize(weighted_cosine(batch["visual"], batch["visual_query"], self.visual_weight), batch["mask"])
        available = (batch["curves"][:, :, 7] > 0) & batch["mask"]
        speech = standardize(weighted_cosine(batch["speech"], batch["speech_query"], self.speech_weight), available)
        # Source features are approximately one per second. Zero-padded shifts
        # represent narration leading/lagging the visible event, never wraparound.
        adjusted = speech.clone()
        for seconds, coefficient in zip(self.lags, .25*self.lag_weights.tanh()):
            if seconds > 0:
                shifted = F.pad(speech[:, :-seconds], (seconds, 0))
            else:
                shifted = F.pad(speech[:, -seconds:], (0, -seconds))
            adjusted = adjusted+coefficient*shifted[:, :speech.shape[1]]
        alpha = self.gate_logit.sigmoid() if self.mode == "fusion" else float(self.mode == "visual")
        if self.mode == "speech" or disable == "visual":
            visual = visual*0
        if self.mode == "visual" or disable == "speech":
            adjusted = adjusted*0
        score = alpha*visual+(1-alpha)*adjusted
        distance = score.new_tensor([self.prior["left_seconds"]/30, self.prior["right_seconds"]/30])
        return {"logits": (score*3).masked_fill(~batch["mask"], -1e4),
                "distances": distance.expand(*score.shape, 2)}

    def regularization(self):
        return .02*(self.visual_weight.square().mean()+self.speech_weight.square().mean()) + .01*self.lag_weights.square().mean()


def mil_loss(model, output, batch):
    positive = batch["labels"] & batch["mask"]
    if not positive.any(-1).all():
        raise ValueError("An interval bag has no observed positive bin")
    scores = output["logits"]
    # A human interval says at least some evidence occurs inside it. It does not
    # assert every frame is equally diagnostic or narration is synchronized.
    loss = scores.logsumexp(-1)-scores.masked_fill(~positive, -1e4).logsumexp(-1)
    return loss.mean()+model.regularization()
