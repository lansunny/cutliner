"""Real audio/video feature data and native pretrained class evidence for Agent tools."""

import csv
import hashlib
import io
import json
import sys

import numpy as np
import requests
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from audio_encoder import MD5, WEIGHT, FrozenAudioEncoder
from extract_ave import CLIP_ROOT, VISUAL_PROFILE
from prepare_ave import DATA

PROMPTS = {
    "Accordion": "A person is playing an accordion.",
    "Acoustic guitar": "A person is playing an acoustic guitar.",
    "Baby cry, infant cry": "A baby is crying.",
    "Banjo": "A person is playing a banjo.",
    "Bark": "A dog is barking.",
    "Bus": "A bus is driving.",
    "Cat": "A cat is meowing.",
    "Chainsaw": "A person is cutting wood with a chainsaw.",
    "Church bell": "A church bell is ringing.",
    "Clock": "A clock is ticking.",
    "Female speech, woman speaking": "A woman is speaking.",
    "Fixed-wing aircraft, airplane": "An airplane is flying.",
    "Flute": "A person is playing a flute.",
    "Frying (food)": "Food is frying in a pan.",
    "Goat": "A goat is bleating.",
    "Helicopter": "A helicopter is flying.",
    "Horse": "A horse is neighing.",
    "Male speech, man speaking": "A man is speaking.",
    "Mandolin": "A person is playing a mandolin.",
    "Motorcycle": "A motorcycle is driving.",
    "Race car, auto racing": "A race car is racing.",
    "Rodents, rats, mice": "A rodent is moving and squeaking.",
    "Shofar": "A person is blowing a shofar.",
    "Toilet flush": "A toilet is flushing.",
    "Train horn": "A train is sounding its horn.",
    "Truck": "A truck is driving.",
    "Ukulele": "A person is playing a ukulele.",
    "Violin, fiddle": "A person is playing a violin.",
}


def prepare_assets():
    path = DATA/"native_classifiers.npz"
    plan = json.loads((DATA/"split_plan.json").read_text(encoding="utf-8"))
    names = [name for name, label in sorted(plan["classes"].items(), key=lambda x: x[1])]
    if set(names) != set(PROMPTS):
        raise ValueError("Explicit event prompts do not cover source taxonomy exactly")
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if data["classes"].tolist() != names or data["prompts"].tolist() != [PROMPTS[name] for name in names]:
                raise ValueError("Native classifier labels/prompts changed")
        return
    csv_path = DATA/"metadata/audioset_class_labels_indices.csv"
    if not csv_path.exists():
        url = "https://raw.githubusercontent.com/qiuqiangkong/audioset_tagging_cnn/master/metadata/class_labels_indices.csv"
        response = requests.get(url, timeout=(10, 30))
        response.raise_for_status()
        with csv_path.open("xb") as stream:
            stream.write(response.content)
    rows = list(csv.DictReader(io.StringIO(csv_path.read_text(encoding="utf-8"))))
    mapping = {row["display_name"]: int(row["index"]) for row in rows}
    if any(name not in mapping for name in names):
        raise ValueError("AVE class has no exact AudioSet class; do not invent mapping")
    if hashlib.md5(WEIGHT.read_bytes()).hexdigest() != MD5:
        raise ValueError("Cnn6 classifier source checkpoint differs")
    checkpoint = torch.load(WEIGHT, map_location="cpu", weights_only=True)["model"]
    indices = [mapping[name] for name in names]
    sys.path.insert(0, str(CLIP_ROOT/"vendor/UniVTG"))
    from run_on_video import clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load(str(CLIP_ROOT/"weights/ViT-B-32.pt"), device=device, jit=False)
    model.eval().requires_grad_(False)
    with torch.inference_mode():
        encoded = model.encode_text(clip.tokenize([PROMPTS[name] for name in names]).to(device))
        text = F.normalize(encoded["pooler_output"].float(), dim=-1)
    np.savez_compressed(path, audio_weight=checkpoint["fc_audioset.weight"][indices].numpy(),
                        audio_bias=checkpoint["fc_audioset.bias"][indices].numpy(), visual_text=text.cpu().numpy(),
                        classes=np.array(names), prompts=np.array([PROMPTS[name] for name in names]),
                        audio_model_md5=MD5, audio_taxonomy_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                        visual_profile=VISUAL_PROFILE, audio_profile=FrozenAudioEncoder.profile)
    print("Prepared native CLIP/PANN classifier controls", flush=True)


class EventFeatures(Dataset):
    def __init__(self, split, frozen_selection=None):
        if split == "test" and frozen_selection is None:
            raise ValueError("Heldout event feature access requires a frozen selection artifact")
        plan = json.loads((DATA/"split_plan.json").read_text(encoding="utf-8"))
        self.rows = plan["selected"][split]
        self.visual, self.audio, self.labels = [], [], []
        with np.load(DATA/"native_classifiers.npz", allow_pickle=False) as data:
            vw, aw, ab = [torch.from_numpy(data[key].copy()) for key in ("visual_text", "audio_weight", "audio_bias")]
        for row in self.rows:
            with np.load(DATA/"features"/f"{row['video']}.npz", allow_pickle=False) as data:
                if str(data["visual_profile"]) != VISUAL_PROFILE or str(data["audio_profile"]) != FrozenAudioEncoder.profile:
                    raise ValueError("Source feature encoder/preprocessing differs")
                visual, audio = data["visual"].copy(), data["audio"].copy()
                if visual.shape != (10, 512) or audio.shape != (10, 512):
                    raise ValueError("Invalid event feature shapes")
                self.visual.append(visual)
                self.audio.append(audio)
                self.labels.append(row["labels"])
        raw_visual, raw_audio = torch.from_numpy(np.stack(self.visual)), torch.from_numpy(np.stack(self.audio))
        self.visual, self.audio = F.normalize(raw_visual, dim=-1), F.normalize(raw_audio, dim=-1)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.float32)
        self.visual_prior = self.visual@vw.T
        self.audio_prior = raw_audio@aw.T+ab
        if not all(x.isfinite().all() for x in (self.visual, self.audio, self.labels, self.visual_prior, self.audio_prior)):
            raise ValueError("Non-finite event training data")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return {key: getattr(self, key)[index] for key in ("visual", "audio", "labels", "visual_prior", "audio_prior")}


def contiguous(binary):
    padded = np.pad(np.asarray(binary, dtype=np.int8), (1, 1))
    return list(zip(np.where(np.diff(padded) == 1)[0].tolist(), np.where(np.diff(padded) == -1)[0].tolist()))


def event_metrics(probabilities, labels, threshold):
    probabilities, labels = np.asarray(probabilities), np.asarray(labels, dtype=bool)
    active = probabilities >= threshold
    tp = (active & labels).sum((0, 1))
    fp, fn = (active & ~labels).sum((0, 1)), (~active & labels).sum((0, 1))
    macro = (2*tp/np.maximum(1, 2*tp+fp+fn)).mean()
    query_values, source_values, partial_values, unannotated_alarms = [], [], [], []
    from train import iou
    for probability, truth in zip(probabilities, labels):
        per_source = []
        for cls in range(28):
            ground_truth = contiguous(truth[:, cls])
            proposals = contiguous(probability[:, cls] >= threshold)
            prediction = max(proposals, key=lambda span: float((probability[span[0]:span[1], cls]-threshold).sum())) if proposals else None
            if ground_truth:
                overlap = max((iou(prediction, target) for target in ground_truth), default=0.) if prediction else 0.
                query_values.append(overlap)
                per_source.append(overlap)
                if sum(b-a for a, b in ground_truth) < 10:
                    partial_values.append(overlap)
            else:
                unannotated_alarms.append(bool(prediction))
        if per_source:
            source_values.append(float(np.mean(np.array(per_source) >= .5)))
    values = np.asarray(query_values)
    return {"source_videos": len(labels), "positive_event_queries": len(values), "threshold": float(threshold),
            "frame_macro_f1": float(macro), "frame_micro_f1": float(2*tp.sum()/max(1, (2*tp+fp+fn).sum())),
            "frame_exact_eventset_accuracy": float(np.all(active == labels, axis=-1).mean()),
            "event_query_R1_iou0.5": float(np.mean(values >= .5)), "event_query_mean_iou": float(values.mean()),
            "source_macro_event_R1_iou0.5": float(np.mean(source_values)),
            "partial_event_query_R1_iou0.5": float(np.mean(np.array(partial_values) >= .5)) if partial_values else None,
            "unannotated_query_alarm_rate": float(np.mean(unannotated_alarms)),
            "absence_limit": "Non-annotated classes are benchmark negatives, not exhaustively human-verified no-target queries"}


def select_threshold(probabilities, labels):
    probabilities, labels = np.asarray(probabilities), np.asarray(labels, dtype=bool)
    best = None
    for threshold in np.arange(.1, .91, .05):
        active = probabilities >= threshold
        tp = (active & labels).sum((0, 1))
        fp, fn = (active & ~labels).sum((0, 1)), (~active & labels).sum((0, 1))
        score = (float((2*tp/np.maximum(1, 2*tp+fp+fn)).mean()), float(threshold))
        if best is None or score > best:
            best = score
    return event_metrics(probabilities, labels, best[1])
