"""Reference the existing QD-DETR so weak new baselines are not mistaken for progress."""

import argparse
import importlib.util
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import AlignedMoments, collate
from public_zip import ROOT
from train import iou, to_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/av_v1")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--selection", type=Path, help="Required before any held-out evaluation")
    args = parser.parse_args()
    if args.split == "test" and (args.selection is None or not args.selection.is_file()):
        raise ValueError("Freeze new-model selection before opening test")
    output_path = args.dataset / f"pretrained_qd_detr_{args.split}.json"
    if output_path.exists():
        raise SystemExit("Reference evaluation already exists; preserving it")
    spec = importlib.util.spec_from_file_location("existing_qd_local", ROOT.parent / "video_grounding_local/run_local.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = module.load_checkpoint(device)
    if int(config.v_feat_dim) != 514:
        raise ValueError("Expected the existing CLIP-only QD-DETR checkpoint")
    dataset = AlignedMoments(args.dataset / f"{args.split}.jsonl")
    loader = DataLoader(dataset, batch_size=16, collate_fn=collate)
    results = []
    with torch.inference_mode():
      for raw in loader:
        batch = to_device(raw, device)
        lengths = batch["mask"].sum(1)
        begin = torch.arange(batch["visual"].shape[1], device=device)[None] / lengths[:, None]
        tef = torch.stack([begin, begin + 1 / lengths[:, None]], dim=-1)
        output = model(src_vid=torch.cat([F.normalize(batch["visual"], dim=-1), tef], dim=-1),
                       src_vid_mask=batch["mask"], src_txt=F.normalize(batch["query_tokens"], dim=-1), src_txt_mask=batch["query_mask"])
        for i, index in enumerate(raw["indices"]):
            row = dataset.rows[index]
            scores = output["pred_logits"][i].softmax(-1)[:, 0].cpu().tolist()
            cxw = output["pred_spans"][i].cpu()
            spans = torch.stack([cxw[:, 0] - cxw[:, 1]/2, cxw[:, 0] + cxw[:, 1]/2], dim=-1).clamp(0, 1) * float(lengths[i]) * 2
            spans[:, 1].clamp_(max=row["duration"])
            predictions = []
            for score, span in sorted(zip(scores, spans.tolist()), reverse=True):
                if span[1] <= span[0] or any(iou(span, old["span"]) > 0.6 for old in predictions):
                    continue
                predictions.append({"span": span, "score": score})
                if len(predictions) == 5:
                    break
            top1 = max((iou(predictions[0]["span"], truth) for truth in row["relevant_windows"]), default=0) if predictions else 0
            top5 = max((iou(item["span"], truth) for item in predictions for truth in row["relevant_windows"]), default=0)
            results.append({"qid": row["qid"], "top1_iou": top1, "top5_iou": top5, "predictions": predictions})
    count = len(results)
    metrics = {"n_queries": count, "mean_iou": sum(row["top1_iou"] for row in results)/count,
               **{f"R1_iou{threshold}": sum(row["top1_iou"] >= threshold for row in results)/count for threshold in (0.3, 0.5, 0.7)},
               "R5_iou0.5": sum(row["top5_iou"] >= 0.5 for row in results)/count}
    report = {"metrics": metrics, "predictions": results,
              "scope": "Existing pretrained CLIP-only QD-DETR on the same source features and NMS; not paper benchmark reproduction",
              "caveat": "QD-DETR was trained on official train, including our new dev subset; reference is not independent on dev"}
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream)
    print(json.dumps({"metrics": metrics, "saved": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
