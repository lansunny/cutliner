"""Low-dimensional, semantically aligned speech/vision evidence timelines."""

import hashlib
import json

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset_speech_candidates import video_features
from prepare_speech import DATA
from speech_baselines import rows_for
from train import iou

VERSION = "paired_eva_minilm_curves_1fps_v1"


def cache_path(split):
    return DATA/f"{split}_evidence_curves.npz"


def prepare(split):
    if split not in ("train","dev"):
        raise ValueError("Heldout curve preparation requires a separate frozen-selection evaluation")
    path = cache_path(split)
    if path.exists():
        with np.load(path,allow_pickle=False) as data:
            if str(data["version"])!=VERSION:
                raise ValueError("Existing curve cache has another preprocessing version")
        print(f"Verified existing curves: {split}",flush=True)
        return
    rows=rows_for(split)
    with np.load(DATA/f"{split}_query_features.npz",allow_pickle=False) as data:
        if data["ids"].tolist()!=[r["qid"] for r in rows]:
            raise ValueError("Speech query IDs differ")
        aq=data["features"].copy()
        audio_encoder=str(data["encoder"])
    with np.load(DATA/f"{split}_eva_query_features.npz",allow_pickle=False) as data:
        if data["ids"].tolist()!=[r["qid"] for r in rows]:
            raise ValueError("Paired visual query IDs differ")
        vq=data["features"].copy()
        visual_encoder=str(data["encoder"])
    curves,labels,targets,timestamps,offsets=[],[],[],[],[0]
    for index,(row,a,v) in enumerate(zip(rows,aq,vq),1):
        visual,speech,available,times=video_features(row["video"],row["duration"])
        normalized=torch.nn.functional.normalize(visual,dim=-1)
        vs=normalized@torch.from_numpy(v)
        ss=speech@torch.from_numpy(a)
        vz=(vs-vs.mean())/vs.std(unbiased=False).clamp(min=0.01)
        az=torch.zeros_like(ss)
        if available.any():
            az[available]=(ss[available]-ss[available].mean())/ss[available].std(unbiased=False).clamp(min=0.01)
        dv=torch.cat([torch.zeros(1),vs[1:]-vs[:-1]])
        da=torch.cat([torch.zeros(1),ss[1:]-ss[:-1]])
        novelty=torch.cat([torch.zeros(1),1-(normalized[1:]*normalized[:-1]).sum(-1)])
        value=torch.stack([vs,ss,vz,az,dv,da,novelty,available.float(),times/row["duration"]],dim=-1)
        n=len(times)
        half=row["duration"]/n/2
        inside=torch.zeros(n,dtype=torch.bool)
        distances=torch.zeros(n,2)
        # A bin represents supplied 1-fps evidence. Tiny boundary rounding is
        # quantized by bin overlap, never silently dropping short true events.
        for start,end in sorted(row["relevant_windows"],key=lambda span:span[1]-span[0],reverse=True):
            positive=(times+half>start)&(times-half<end)
            inside[positive]=True
            distances[positive,0]=(times[positive]-start)/30
            distances[positive,1]=(end-times[positive])/30
        if not inside.any():
            raise ValueError("Human interval has no representable source bin")
        curves.append(value.numpy())
        labels.append(inside.numpy())
        targets.append(distances.numpy())
        timestamps.append(times.numpy())
        offsets.append(offsets[-1]+n)
        if index%500==0:
            print(f"Prepared {split} curves {index}/{len(rows)}",flush=True)
    np.savez_compressed(path,curves=np.concatenate(curves),labels=np.concatenate(labels),targets=np.concatenate(targets),
                        times=np.concatenate(timestamps),offsets=np.array(offsets),ids=np.array([r["qid"] for r in rows]),
                        durations=np.array([r["duration"] for r in rows],np.float32),version=VERSION,
                        audio_encoder=audio_encoder,visual_encoder=visual_encoder,
                        manifest_sha256=hashlib.sha256((DATA/f"{split}.jsonl").read_bytes()).hexdigest())
    print(f"Evidence curves ready: {split}, {len(rows)} queries",flush=True)


class EvidenceCurves(Dataset):
    def __init__(self,split):
        self.rows=[json.loads(line) for line in (DATA/f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        with np.load(cache_path(split),allow_pickle=False) as data:
            if data["ids"].tolist()!=[r["qid"] for r in self.rows] or str(data["version"])!=VERSION:
                raise ValueError("Curve cache manifest/version mismatch")
            if str(data["manifest_sha256"])!=hashlib.sha256((DATA/f"{split}.jsonl").read_bytes()).hexdigest():
                raise ValueError("Underlying source manifest changed")
            for key in ("curves","labels","targets","times","offsets","durations"):
                setattr(self,key,data[key].copy())
    def __len__(self):
        return len(self.rows)
    def __getitem__(self,index):
        start,end=self.offsets[index:index+2]
        return {"curves":torch.from_numpy(self.curves[start:end]),"labels":torch.from_numpy(self.labels[start:end]),
                "targets":torch.from_numpy(self.targets[start:end]),"times":torch.from_numpy(self.times[start:end]),
                "duration":float(self.durations[index]),"index":index}


def collate_curves(items):
    n,b=max(len(item["curves"]) for item in items),len(items)
    result={"curves":torch.zeros(b,n,9),"labels":torch.zeros(b,n,dtype=torch.bool),"targets":torch.zeros(b,n,2),
            "times":torch.zeros(b,n),"mask":torch.zeros(b,n,dtype=torch.bool),
            "duration":torch.tensor([item["duration"] for item in items]),"indices":[item["index"] for item in items]}
    for i,item in enumerate(items):
        size=len(item["curves"])
        for key in ("curves","labels","targets","times"):
            result[key][i,:size]=item[key]
        result["mask"][i,:size]=True
    return result


def baseline_metrics(dataset,alpha,left,right):
    values=[]
    for i,row in enumerate(dataset.rows):
        start,end=dataset.offsets[i:i+2]
        curve=dataset.curves[start:end]
        if alpha==0 and not curve[:,7].any():
            values.append(0.)
            continue
        index=int((alpha*curve[:,2]+(1-alpha)*curve[:,3]).argmax())
        center=float(dataset.times[start+index])
        span=(max(0.,center-left),min(row["duration"],center+right))
        values.append(max(iou(span,target) for target in row["relevant_windows"]) if span[0]<span[1] else 0.)
    return reduce_metrics(dataset.rows,values)


def reduce_metrics(rows,values):
    groups={}
    for row,value in zip(rows,values):
        groups.setdefault(row["video"],[]).append(value)
    values=np.array(values)
    return {"queries":len(rows),"source_videos":len(groups),"mean_iou":float(values.mean()),
            **{f"R1_iou{threshold}":float((values>=threshold).mean()) for threshold in (0.3,0.5,0.7)},
            "source_macro_R1_iou0.5":float(np.mean([np.mean(np.array(v)>=0.5) for v in groups.values()]))}


def main():
    torch.set_num_threads(4)
    for split in ("train","dev"):
        prepare(split)
    output=DATA/"curve_baselines_dev.json"
    if output.exists():
        print("Preserving existing curve baseline report",flush=True)
        return
    train,dev=EvidenceCurves("train"),EvidenceCurves("dev")
    report={"version":VERSION,"fit_split":"train_only","heldout_not_accessed":True,"modes":{}}
    for mode,alphas in (("visual",[1.]),("speech",[0.]),("fusion",[0.,0.25,0.5,0.75,1.])):
        best=None
        for alpha in alphas:
            for left in (0,5,10,20):
                for right in (5,10,20,30,45):
                    metrics=baseline_metrics(train,alpha,left,right)
                    candidate=(metrics["source_macro_R1_iou0.5"]+0.25*metrics["mean_iou"],-left-right,alpha,left,right,metrics)
                    if best is None or candidate[:5]>best[:5]:
                        best=candidate
        _,_,alpha,left,right,metrics=best
        report["modes"][mode]={"alpha":alpha,"left_seconds":left,"right_seconds":right,"train":metrics,
                               "dev":baseline_metrics(dev,alpha,left,right)}
        print(json.dumps({"mode":mode,**report["modes"][mode]}),flush=True)
    with output.open("x",encoding="utf-8") as stream:
        json.dump(report,stream,indent=2)


if __name__=="__main__":
    main()
