"""Freeze development decisions, then evaluate audiovisual event heads once by source."""

import argparse
import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from torch.utils.data import DataLoader

from audio_encoder import FrozenAudioEncoder
from event_dataset import EventFeatures, contiguous, event_metrics
from extract_ave import FrozenEventFeatures, VISUAL_PROFILE, decode_media
from model_events import EventGrounder, ScoreCalibration
from prepare_ave import DATA, ensure_video
from public_zip import ROOT
from train import iou
from train_events import probabilities_for


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(folder):
    path = folder/"frozen_selection.json"
    if path.exists():
        print(f"Preserving existing frozen event selection: {path}", flush=True)
        return
    report = json.loads((folder/"development_report.json").read_text(encoding="utf-8"))
    seeds = report["config"]["seeds"]
    if len(set(seeds)) < 3:
        raise ValueError("At least three training seeds required before event heldout evaluation")
    by_mode = {}
    for run in report["runs"]:
        by_mode.setdefault(run["mode"], []).append(run)
    required = {"visual", "visual_wide", "audio", "concat", "gated"}
    if set(by_mode) != required or any({r["seed"] for r in v} != set(seeds) for v in by_mode.values()):
        raise ValueError("Planned control runs are incomplete")
    means = {mode: float(np.mean([r["selected_dev"]["frame_macro_f1"] for r in runs])) for mode, runs in by_mode.items()}
    selected = max(("concat", "gated"), key=means.get)
    strongest = max([means[m] for m in ("visual", "visual_wide", "audio")]+[v["frame_macro_f1"] for v in report["static_dev"].values()])
    recall = float(np.mean([r["selected_dev"]["event_query_R1_iou0.5"] for r in by_mode[selected]]))
    control_recall = max([float(np.mean([r["selected_dev"]["event_query_R1_iou0.5"] for r in by_mode[m]])) for m in ("visual", "visual_wide", "audio")]
                         +[v["event_query_R1_iou0.5"] for v in report["static_dev"].values()])
    if means[selected] < strongest+.03 or recall < control_recall-.005:
        raise SystemExit(f"No heldout spent: dev fusion evidence insufficient. F1={means[selected]:.4f}, strongest control={strongest:.4f}, queryR1={recall:.4f}, control={control_recall:.4f}")
    frozen = {"selected_mode": selected, "seeds": seeds, "development_means_f1": means,
              "rule": "Best mean dev F1 joint mode; >=3pp over strongest control and queryR1 no worse by>0.5pp",
              "plan_sha256": sha(DATA/"split_plan.json"), "calibration_sha256": sha(DATA/"calibration_train_only.pt"),
              "native_assets_sha256": sha(DATA/"native_classifiers.npz"),
              "evaluation_source_sha256": sha(ROOT/"evaluate_events.py"),
              "checkpoints": {f"{r['mode']}_seed{r['seed']}.pt": sha(folder/f"{r['mode']}_seed{r['seed']}.pt") for r in report["runs"]},
              "static_thresholds": {k: v["threshold"] for k, v in report["static_dev"].items()},
              "thresholds": {f"{r['mode']}_seed{r['seed']}": r["selected_dev"]["threshold"] for r in report["runs"]},
              "heldout_performance_seen_at_freeze": False,
              "interpretation": "Auxiliary closed-taxonomy event tool; no claim of arbitrary query, dialogue reasoning, or long-video retrieval gains"}
    with path.open("x", encoding="utf-8") as stream:
        json.dump(frozen, stream, indent=2)
    print(f"Frozen event selection: {selected}; heldout not yet evaluated", flush=True)


def validate_frozen(folder):
    path = folder/"frozen_selection.json"
    frozen = json.loads(path.read_text(encoding="utf-8"))
    for key, source in (("plan_sha256", DATA/"split_plan.json"), ("calibration_sha256", DATA/"calibration_train_only.pt"),
                        ("native_assets_sha256", DATA/"native_classifiers.npz"), ("evaluation_source_sha256", ROOT/"evaluate_events.py")):
        if frozen[key] != sha(source):
            raise ValueError("Frozen evaluation source/assets changed")
    for name, expected in frozen["checkpoints"].items():
        if sha(folder/name) != expected:
            raise ValueError("Frozen event checkpoint changed")
    return frozen


def prepare_heldout(folder):
    validate_frozen(folder)
    plan = json.loads((DATA/"split_plan.json").read_text(encoding="utf-8"))
    entries = json.loads((DATA/"video_zip_index.json").read_text(encoding="utf-8"))["entries"]
    by_name = {e["name"]: e for e in entries}
    rows = plan["selected"]["test"]
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(ensure_video, row, by_name[row["zip_member"]]) for row in rows]
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            if i%25 == 0 or i == len(rows):
                print(f"Verified frozen heldout media {i}/{len(rows)}", flush=True)
    encoder = FrozenEventFeatures("cuda" if torch.cuda.is_available() else "cpu")
    for i, row in enumerate(rows, 1):
        media, path = DATA/"videos"/f"{row['video']}.mp4", DATA/"features"/f"{row['video']}.npz"
        signature = sha(media)
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                if str(data["media_sha256"]) != signature or str(data["visual_profile"]) != VISUAL_PROFILE or str(data["audio_profile"]) != FrozenAudioEncoder.profile:
                    raise ValueError("Existing heldout cache source/profile differs")
        else:
            visual, audio = encoder.encode(decode_media(media))
            np.savez_compressed(path, visual=visual, audio=audio, media_sha256=signature,
                                visual_profile=VISUAL_PROFILE, audio_profile=FrozenAudioEncoder.profile)
        if i%25 == 0 or i == len(rows):
            print(f"Prepared frozen heldout features {i}/{len(rows)}", flush=True)


def per_source_statistics(probability, labels, threshold):
    truth, active = labels.astype(bool), probability >= threshold
    tp = (truth & active).sum(1)
    fp, fn = (~truth & active).sum(1), (truth & ~active).sum(1)
    hits, counts = [], []
    for p, y in zip(probability, truth):
        hit, count = 0, 0
        for cls in range(28):
            targets = contiguous(y[:, cls])
            if not targets:
                continue
            count += 1
            proposals = contiguous(p[:, cls] >= threshold)
            best = max(proposals, key=lambda span: float((p[span[0]:span[1], cls]-threshold).sum())) if proposals else None
            hit += bool(best and max(iou(best, target) for target in targets) >= .5)
        hits.append(hit)
        counts.append(count)
    return tp, fp, fn, np.array(hits), np.array(counts)


def bootstrap_pair(joint, control, indices):
    def scores(group, picked):
        f1, recall = [], []
        for tp, fp, fn, hits, counts in group:
            a, b, c = (x[picked].sum(1) for x in (tp, fp, fn))
            f1.append((2*a/np.maximum(1, 2*a+b+c)).mean(-1))
            recall.append(hits[picked].sum(-1)/np.maximum(1, counts[picked].sum(-1)))
        return np.mean(f1, axis=0), np.mean(recall, axis=0)
    differences = [[], []]
    for start in range(0, len(indices), 100):
        picked = indices[start:start+100]
        j, c = scores(joint, picked), scores(control, picked)
        for i in range(2):
            differences[i].extend((j[i]-c[i]).tolist())
    return {name: {"paired_source_bootstrap95ci_pp": (np.quantile(values, [.025, .975])*100).tolist(),
                   "resamples": len(indices)} for name, values in zip(("frame_macro_f1", "event_query_R1_iou0.5"), differences)}


def evaluate(folder):
    frozen = validate_frozen(folder)
    output = folder/"heldout_results.json"
    if output.exists():
        raise SystemExit("Event heldout results already exist; no retuning or silent re-evaluation")
    data = EventFeatures("test", frozen_selection=frozen)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = DataLoader(data, batch_size=64)
    calibration = torch.load(DATA/"calibration_train_only.pt", map_location="cpu", weights_only=True)
    models = {}
    for mode in ("visual", "audio"):
        state = calibration[mode]
        models[mode] = ScoreCalibration(state["mean"], state["std"])
        models[mode].load_state_dict(state)
    arrays, results, stats = {}, {}, {}
    truth = data.labels.numpy()
    with torch.inference_mode():
        v, a = [models[mode](getattr(data, f"{mode}_prior")) for mode in ("visual", "audio")]
        alpha = calibration["alpha"]
        native = {"visual": v.sigmoid().numpy(), "audio": a.sigmoid().numpy(),
                  "late_fusion": (alpha*v+(1-alpha)*a).sigmoid().numpy()}
    for name, probability in native.items():
        key, threshold = f"static_{name}", frozen["static_thresholds"][name]
        arrays[key] = probability
        results[key] = event_metrics(probability, truth, threshold)
        stats[key] = [per_source_statistics(probability, truth, threshold)]
    for filename in frozen["checkpoints"]:
        checkpoint = torch.load(folder/filename, map_location="cpu", weights_only=True)
        mode, seed = checkpoint["mode"], checkpoint["seed"]
        model = EventGrounder(mode, copy.deepcopy(models["visual"]), copy.deepcopy(models["audio"]), checkpoint["alpha"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        key = f"{mode}_seed{seed}"
        probability = probabilities_for(model, loader, device)
        threshold = frozen["thresholds"][key]
        arrays[key] = probability
        results[key] = event_metrics(probability, truth, threshold)
        stats.setdefault(mode, []).append(per_source_statistics(probability, truth, threshold))
        print(f"Heldout {key}: F1={results[key]['frame_macro_f1']:.4f}, queryR1={results[key]['event_query_R1_iou0.5']:.4f}", flush=True)
    indices = np.random.default_rng(20261107).integers(0, len(data), size=(2000, len(data)))
    selected = frozen["selected_mode"]
    comparisons = {name: bootstrap_pair(stats[selected], values, indices) for name, values in stats.items()
                   if name not in (selected, "concat", "gated")}
    np.savez_compressed(folder/"heldout_predictions.npz", **arrays, labels=truth,
                        ids=np.array([r["video"] for r in data.rows]), frozen_selection_sha256=sha(folder/"frozen_selection.json"))
    report = {"selection": frozen, "results": results, "paired_comparisons": comparisons,
              "uncertainty": "Resample original source videos, paired across methods; seed-average effects. Not frame-independent intervals.",
              "limits": "Derived class-capped, grouped AVE task; AudioSet-pretrained audio backbone. Not arbitrary-query or long-video Agent evaluation."}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(f"Frozen event evaluation complete: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=lambda p: (ROOT/p).resolve(), required=True)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--prepare-heldout", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(3)
    if args.freeze:
        freeze(args.run)
    if args.prepare_heldout:
        prepare_heldout(args.run)
    if args.evaluate:
        evaluate(args.run)


if __name__ == "__main__":
    main()
