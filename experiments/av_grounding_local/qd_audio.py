"""Add trainable audio evidence to the existing pretrained temporal localizer."""

import importlib.util

import torch
from torch import nn
from torch.nn import functional as F

from public_zip import ROOT
from train import iou, to_device


def load_teacher(device):
    spec = importlib.util.spec_from_file_location("existing_av_qd_teacher", ROOT.parent / "video_grounding_local/run_local.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    teacher, config = module.load_checkpoint(device)
    if int(config.v_feat_dim) != 514:
        raise ValueError("Expected existing CLIP-only checkpoint")
    from lighthouse.common.qd_detr import build_model
    unused, criterion = build_model(config)
    del unused
    criterion.losses = ["spans", "labels"]
    teacher.eval().requires_grad_(False)
    return teacher, criterion.to(device)


def qd_inputs(batch, visual):
    lengths = batch["mask"].sum(1)
    begin = torch.arange(visual.shape[1], device=visual.device)[None] / lengths[:, None]
    tef = torch.stack([begin, begin + 1 / lengths[:, None]], dim=-1)
    return {"src_vid": torch.cat([visual, tef], dim=-1), "src_vid_mask": batch["mask"],
            "src_txt": F.normalize(batch["query_tokens"], dim=-1), "src_txt_mask": batch["query_mask"]}


class ResidualAudio(nn.Module):
    def __init__(self, mode="query_gated", hidden=128):
        super().__init__()
        if mode not in ("audio_residual", "query_gated"):
            raise ValueError("Unknown audio adaptation method")
        self.mode = mode
        self.audio = nn.Sequential(nn.Linear(2048, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.temporal = nn.Sequential(nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU())
        self.visual = nn.Linear(512, hidden)
        self.query = nn.Linear(512, hidden)
        self.gate = nn.Sequential(nn.Linear(5 * hidden, 64), nn.GELU(), nn.Linear(64, 1))
        self.output = nn.Linear(hidden, 512)
        # Exactly zero initial audio residual; preserve the pretrained starting point.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        if mode == "audio_residual":
            self.visual.requires_grad_(False)
            self.query.requires_grad_(False)
            self.gate.requires_grad_(False)

    def forward(self, batch):
        audio = self.audio(F.normalize(batch["audio"], dim=-1))
        audio = self.temporal(audio.transpose(1, 2)).transpose(1, 2)
        if self.mode == "query_gated":
            visual = self.visual(F.normalize(batch["visual"], dim=-1))
            query = self.query(F.normalize(batch["query"], dim=-1))[:, None].expand_as(audio)
            gate = self.gate(torch.cat([audio, visual, query, audio * query, visual * query], dim=-1)).sigmoid()
        else:
            gate = torch.ones_like(batch["centers"])[..., None]
        delta = 0.1 * self.output(audio).tanh() * gate * batch["audio_mask"][..., None]
        return delta, gate


class AdaptedGrounder(nn.Module):
    def __init__(self, teacher, mode):
        super().__init__()
        self.teacher = teacher
        self.fusion = ResidualAudio(mode)

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()  # Frozen parameters alone do not disable dropout.
        return self

    def forward(self, batch, disable_audio=False, reverse_audio=False):
        if reverse_audio:
            batch = dict(batch)
            batch["audio"] = batch["audio"].clone()
            for i, mask in enumerate(batch["audio_mask"]):
                length = int(mask.sum())
                batch["audio"][i, :length] = batch["audio"][i, :length].flip(0)
        visual = F.normalize(batch["visual"], dim=-1)
        delta, gate = self.fusion(batch)
        if disable_audio:
            delta = torch.zeros_like(delta)
        output = self.teacher(**qd_inputs(batch, F.normalize(visual + delta, dim=-1)))
        return output, delta, gate


def targets(batch, dataset):
    result = []
    for i, index in enumerate(batch["indices"]):
        span = torch.tensor(dataset.rows[index]["relevant_windows"], device=batch["visual"].device, dtype=torch.float32)
        span /= batch["mask"][i].sum() * 2
        cxw = torch.stack([(span[:, 0] + span[:, 1]) / 2, span[:, 1] - span[:, 0]], dim=-1)
        result.append({"spans": cxw})
    return {"span_labels": result}


def decode(output, batch, i, duration):
    scores = output["pred_logits"][i].softmax(-1)[:, 0].cpu().tolist()
    cxw = output["pred_spans"][i].cpu()
    spans = torch.stack([cxw[:, 0] - cxw[:, 1]/2, cxw[:, 0] + cxw[:, 1]/2], dim=-1).clamp(0, 1) * int(batch["mask"][i].sum()) * 2
    spans[:, 1].clamp_(max=duration)
    predictions = []
    for score, span in sorted(zip(scores, spans.tolist()), reverse=True):
        if span[1] <= span[0] or any(iou(span, old["span"]) > 0.6 for old in predictions):
            continue
        predictions.append({"span": span, "score": score})
        if len(predictions) == 5:
            break
    return predictions


@torch.inference_mode()
def evaluate_adapter(model, loader, dataset, device, disable_audio=False, reverse_audio=False):
    model.eval()
    results = []
    for raw in loader:
        batch = to_device(raw, device)
        output, delta, gate = model(batch, disable_audio, reverse_audio)
        for i, index in enumerate(raw["indices"]):
            row = dataset.rows[index]
            predictions = decode(output, batch, i, row["duration"])
            top1 = max((iou(predictions[0]["span"], truth) for truth in row["relevant_windows"]), default=0) if predictions else 0
            top5 = max((iou(item["span"], truth) for item in predictions for truth in row["relevant_windows"]), default=0)
            results.append({"qid": row["qid"], "source_group": row["vid"].rsplit("_", 2)[0], "top1_iou": top1,
                            "top5_iou": top5, "predictions": predictions,
                            "audio_delta_norm": float(delta[i, batch["mask"][i]].norm(dim=-1).mean()),
                            "mean_audio_gate": float(gate[i, batch["mask"][i]].mean())})
    count = len(results)
    metrics = {"n_queries": count, "mean_iou": sum(row["top1_iou"] for row in results)/count,
               **{f"R1_iou{threshold}": sum(row["top1_iou"] >= threshold for row in results)/count for threshold in (0.3, 0.5, 0.7)},
               "R5_iou0.5": sum(row["top5_iou"] >= 0.5 for row in results)/count}
    return metrics, results
