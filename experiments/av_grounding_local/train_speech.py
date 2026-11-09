"""Develop proposal-based speech/vision models without touching the heldout split."""

import argparse
import hashlib
import json
import random
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_speech_candidates import SpeechCandidates, collate_candidates
from model_speech import SpeechGrounder,objective
from prepare_speech import DATA
from public_zip import ROOT
from train import iou,to_device


@torch.inference_mode()
def evaluate_speech(model,loader,dataset,device,disable_visual=False):
    model.eval()
    predictions = []
    for raw in loader:
        batch = to_device(raw,device)
        output = model(batch,disable_visual)
        for i,index in enumerate(raw["indices"]):
            row = dataset.rows[index]
            order = output["scores"][i].argsort(descending=True).tolist()
            spans = output["spans"][i].cpu().tolist()
            predicted = None
            for j in order:
                if not bool(batch["candidate_mask"][i,j]):
                    continue
                start,end = max(0.,spans[j][0]),min(row["duration"],spans[j][1])
                if start < end:
                    predicted = [start,end]
                    break
            overlap = max((iou(predicted,span) for span in row["relevant_windows"]),default=0.) if predicted else 0.
            predictions.append({"qid":row["qid"],"video":row["video"],"iou":overlap,"span":predicted})
    source_values = {}
    for row in predictions:
        source_values.setdefault(row["video"],[]).append(row["iou"])
    values = np.array([row["iou"] for row in predictions])
    metrics = {"queries":len(predictions),"source_videos":len(source_values),"mean_iou":float(values.mean()),
               **{f"R1_iou{threshold}":float((values>=threshold).mean()) for threshold in (0.3,0.5,0.7)},
               "source_macro_R1_iou0.5":float(np.mean([np.mean(np.array(v)>=0.5) for v in source_values.values()]))}
    return metrics,predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes",nargs="+",choices=["speech","concat","aligned"],default=["speech","concat","aligned"])
    parser.add_argument("--seeds",nargs="+",type=int,default=[42,43,44])
    parser.add_argument("--epochs",type=int,default=12)
    parser.add_argument("--batch-size",type=int,default=8)
    parser.add_argument("--hidden",type=int,default=96)
    parser.add_argument("--lr",type=float,default=0.0003)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train,dev = SpeechCandidates("train"),SpeechCandidates("dev")
    if {r["video"] for r in train.rows}&{r["video"] for r in dev.rows}:
        raise ValueError("Original-video split leakage")
    dev_loader = DataLoader(dev,batch_size=args.batch_size,collate_fn=collate_candidates)
    folder = ROOT/"speech_runs"/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True)
    config = {"modes":args.modes,"seeds":args.seeds,"epochs":args.epochs,"batch_size":args.batch_size,"hidden":args.hidden,"lr":args.lr,
              "plan_sha256":hashlib.sha256((DATA/"split_plan.json").read_bytes()).hexdigest(),
              "train_manifest_sha256":hashlib.sha256((DATA/"train.jsonl").read_bytes()).hexdigest(),
              "dev_manifest_sha256":hashlib.sha256((DATA/"dev.jsonl").read_bytes()).hexdigest(),
              "heldout_accessed":False,"encoder_frozen":True,"frame_sampling":"all supplied 1-fps source features; no additional uniform subsampling",
              "task":"HiREST derived step-query temporal grounding; not original official benchmark task",
              "proposal_policy":"duration-scaled 4..12 diverse speech contexts + full-timeline recovery; no GT in inputs"}
    (folder/"config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    runs = []
    for mode in args.modes:
      for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        model = SpeechGrounder(mode,args.hidden).to(device)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=0.01)
        loader = DataLoader(train,batch_size=args.batch_size,shuffle=True,collate_fn=collate_candidates,
                            generator=torch.Generator().manual_seed(seed))
        baseline,_ = evaluate_speech(model,dev_loader,dev,device)
        best = baseline["source_macro_R1_iou0.5"] + 0.25*baseline["mean_iou"]
        checkpoint_path = folder/f"{mode}_seed{seed}.pt"

        def save(epoch,metrics):
            torch.save({"state_dict":model.state_dict(),"mode":mode,"seed":seed,"epoch":epoch,
                        "development":metrics,"config":config},checkpoint_path)

        save(0,baseline)
        history = [{"epoch":0,"dev":baseline}]
        print(f"{mode} seed{seed} pretrained-speech starting dev: {baseline}",flush=True)
        for epoch in range(args.epochs):
            model.train()
            total = 0.
            for raw in loader:
                batch = to_device(raw,device)
                optimizer.zero_grad(set_to_none=True)
                output = model(batch)
                loss,_ = objective(model,output,batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step()
                total += float(loss.detach())*len(raw["indices"])
            metrics,_ = evaluate_speech(model,dev_loader,dev,device)
            score = metrics["source_macro_R1_iou0.5"]+0.25*metrics["mean_iou"]
            history.append({"epoch":epoch+1,"loss":total/len(train),"dev":metrics})
            print(f"{mode} seed{seed} epoch{epoch+1}: loss={total/len(train):.4f}, devR1@.5={metrics['R1_iou0.5']:.4f}, sourceMacro={metrics['source_macro_R1_iou0.5']:.4f}, mIoU={metrics['mean_iou']:.4f}",flush=True)
            if score > best:
                best = score
                save(epoch+1,metrics)
        checkpoint = torch.load(checkpoint_path,map_location="cpu",weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        controls = {}
        if mode != "speech":
            controls["visual_disabled"],_ = evaluate_speech(model,dev_loader,dev,device,True)
        runs.append({"mode":mode,"seed":seed,"selected_epoch":checkpoint["epoch"],"selected_dev":checkpoint["development"],
                     "controls_dev":controls,"trained_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),"history":history})
        (folder/"development_report.json").write_text(json.dumps({"config":config,"runs":runs},indent=2),encoding="utf-8")
    print(f"Speech development run finished: {folder}; heldout untouched",flush=True)


if __name__ == "__main__":
    main()
