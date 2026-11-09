"""Query-conditioned timeline encoder and proposal ranker/boundary refiner."""

import torch
from torch import nn
from torch.nn import functional as F


class TemporalBlock(nn.Module):
    def __init__(self, hidden, dilation):
        super().__init__()
        self.network = nn.Sequential(nn.Conv1d(hidden,hidden,3,padding=dilation,dilation=dilation), nn.GELU(),
                                     nn.Conv1d(hidden,hidden,1), nn.Dropout(0.1))
        self.norm = nn.LayerNorm(hidden)

    def forward(self,x,mask):
        result = self.norm(x + self.network(x.transpose(1,2)).transpose(1,2))
        return result * mask[...,None]


class SpeechGrounder(nn.Module):
    def __init__(self,mode="concat",hidden=96):
        super().__init__()
        if mode not in ("speech","concat","aligned"):
            raise ValueError("Unknown speech/visual ablation")
        self.mode,self.hidden = mode,hidden
        self.visual = nn.Sequential(nn.Linear(1024,hidden),nn.LayerNorm(hidden),nn.GELU())
        self.text = nn.Sequential(nn.Linear(384,hidden),nn.LayerNorm(hidden),nn.GELU())
        self.condition = nn.Sequential(nn.Linear(5*hidden+4,hidden),nn.LayerNorm(hidden),nn.GELU())
        self.temporal = nn.ModuleList([TemporalBlock(hidden,dilation) for dilation in (1,2,4)])
        self.attention = nn.Linear(hidden,1)
        self.candidate = nn.Sequential(nn.Linear(2*hidden+5,hidden),nn.GELU(),nn.Dropout(0.1))
        self.confidence = nn.Linear(hidden,1)
        self.delta = nn.Linear(hidden,2)
        self.visual_alignment = nn.Linear(hidden,384)
        nn.init.zeros_(self.confidence.weight)
        nn.init.zeros_(self.confidence.bias)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        if mode == "speech":
            self.visual.requires_grad_(False)
        if mode != "aligned":
            self.visual_alignment.requires_grad_(False)

    def forward(self,batch,disable_visual=False):
        visual = self.visual(F.normalize(batch["visual"],dim=-1))
        if self.mode == "speech" or disable_visual:
            visual = torch.zeros_like(visual)
        query = self.text(F.normalize(batch["query"],dim=-1))
        q = query[:,None].expand_as(visual)
        speech = self.text(F.normalize(batch["speech"],dim=-1)) * batch["speech_available"][...,None]
        similarity = (F.normalize(batch["speech"],dim=-1)*batch["query"][:,None]).sum(-1)
        t = batch["times"]/batch["duration"][:,None]
        additional = torch.stack([similarity,batch["speech_available"].float(),t,1-t],dim=-1)
        latent = self.condition(torch.cat([visual,speech,q,visual*q,speech*q,additional],dim=-1))
        for layer in self.temporal:
            latent = layer(latent,batch["mask"])
        attention = self.attention(latent).squeeze(-1)[:,None].expand_as(batch["context_mask"])
        attention = attention.masked_fill(~batch["context_mask"],-1e4)
        weights = attention.softmax(-1) * batch["candidate_mask"][...,None]
        pooled = torch.einsum("bkn,bnh->bkh",weights,latent)
        candidate = self.candidate(torch.cat([pooled,query[:,None].expand_as(pooled),batch["metadata"]],dim=-1))
        scores = 10*batch["semantic"] + self.confidence(candidate).squeeze(-1)
        scores = scores.masked_fill(~batch["candidate_mask"],-1e4)
        spans = batch["base"] + 30*self.delta(candidate)
        return {"scores":scores,"spans":spans,"visual_latent":visual}


def objective(model,output,batch):
    indices = torch.arange(len(batch["oracle"]),device=batch["oracle"].device)
    predicted = output["spans"][indices,batch["oracle"]]
    target = batch["target"][indices,batch["oracle"]]
    rank = F.cross_entropy(output["scores"],batch["oracle"])
    localization = F.smooth_l1_loss(predicted/30,target/30,beta=0.25)
    loss = rank + 2*localization
    alignment = torch.tensor(0.,device=loss.device)
    if model.mode == "aligned":
        times = batch["times"]
        mask = (times >= target[:,0,None]) & (times < target[:,1,None]) & batch["mask"]
        pooled = (output["visual_latent"]*mask[...,None]).sum(1)/mask.sum(1).clamp(min=1)[:,None]
        visual = F.normalize(model.visual_alignment(pooled),dim=-1)
        query = F.normalize(batch["query"],dim=-1)
        logits = visual @ query.T / 0.1
        alignment = F.cross_entropy(logits,indices)
        loss = loss + 0.1*alignment
    return loss,{"rank":float(rank.detach()),"localization":float(localization.detach()),"alignment":float(alignment.detach())}
