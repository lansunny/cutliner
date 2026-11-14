"""Revalidate frozen AV deployment on dev caches and deterministic real-media replays.

No training, threshold selection, heldout-feature access, cloud calls or exports.
Cached-head metrics are separate from the small raw-media replay audit.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parents[2]
AV = PROJECT / "experiments/av_grounding_local"
sys.path.insert(0, str(AV))

from evaluate_events import bootstrap_pair, per_source_statistics, validate_frozen
from event_dataset import EventFeatures, event_metrics
from model_events import EventGrounder, ScoreCalibration
from predict_events import EventRuntime, decode_window
from prepare_ave import DATA
from train_events import probabilities_for


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_head(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]
    calibrations = []
    for prefix in ("visual_calibration.", "audio_calibration."):
        parameters = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        calibration = ScoreCalibration(parameters["mean"], parameters["std"])
        calibration.load_state_dict(parameters)
        calibrations.append(calibration)
    model = EventGrounder(checkpoint["mode"], *calibrations, checkpoint["alpha"])
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False).to(device), checkpoint


@torch.inference_mode()
def run(args):
    if args.output.exists():
        raise ValueError("Preserve existing validation results; choose a new output directory")
    manifest_path = AV / "approved_event_model.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    folder = AV / manifest["run_dir"]
    frozen = validate_frozen(folder)
    selected = f"{manifest['selected_mode']}_seed{manifest['deployment_seed']}"
    if manifest["deployment_seed"] != frozen["seeds"][0] or manifest["selected_mode"] != frozen["selected_mode"]:
        raise ValueError("Deployment does not match the frozen first-seed rule")
    for path, expected in ((folder / f"{selected}.pt", manifest["checkpoint_sha256"]),
                           (folder / "frozen_selection.json", manifest["frozen_selection_sha256"]),
                           (folder / "heldout_results.json", manifest["heldout_results_sha256"])):
        if sha(path) != expected:
            raise ValueError(f"Frozen deployment asset changed: {path.name}")
    torch.set_num_threads(3)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    status = args.output / "status.json"
    save(status, {"status": "running", "phase": "audit_dev_sources"})
    try:
        plan = json.loads((DATA / "split_plan.json").read_text(encoding="utf-8"))
        dev_ids = [row["video"] for row in plan["selected"]["dev"]]
        if len(dev_ids) != len(set(dev_ids)) or set(dev_ids) & {r["video"] for r in plan["selected"]["train"]}:
            raise ValueError("Duplicate development source or train/dev leakage")
        audited = []
        for name in dev_ids:
            media, cache = DATA / "videos" / f"{name}.mp4", DATA / "features" / f"{name}.npz"
            with np.load(cache, allow_pickle=False) as data:
                if str(data["media_sha256"]) != sha(media):
                    raise ValueError(f"Development media differs from cache: {name}")
            audited.append({"video": name, "media_sha256": sha(media), "cache_sha256": sha(cache)})
        sources = [Path(__file__), PROJECT / "experiments/neural_retrieval/retrieve_worker.py"] + [AV / name for name in
            ("predict_events.py", "event_dataset.py", "model_events.py", "audio_encoder.py", "extract_ave.py", "train_events.py", "evaluate_events.py")]
        hashes = {str(path.relative_to(PROJECT)): sha(path) for path in sources}
        for path in sources:
            target = args.output / "source_snapshot" / path.relative_to(PROJECT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        save(args.output / "configuration.json", {"scope": "AVE-derived 28-class development set, frozen single-seed deployment",
            "manifest": manifest, "manifest_sha256": sha(manifest_path), "split_plan_sha256": sha(DATA / "split_plan.json"),
            "device": args.device, "seed": manifest["deployment_seed"], "thresholds": frozen["thresholds"],
            "source_sha256": hashes, "source_audit": audited, "raw_replay_count": args.replay_sources,
            "raw_replay_selection": "SHA256 order using av-dev-replay-v1 prefix; no performance selection",
            "test_features_accessed": False, "models_or_thresholds_tuned": False,
            "development_data_previously_used_for_selection": True, "paid_api_calls": 0})
        dataset = EventFeatures("dev")
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        truth = dataset.labels.numpy()
        models, arrays, metrics, statistics = {}, {}, {}, {}
        controls = ("visual", "visual_wide", "audio", manifest["selected_mode"])
        for mode in controls:
            key = f"{mode}_seed{manifest['deployment_seed']}"
            model, checkpoint = load_head(folder / f"{key}.pt", args.device)
            models[mode] = model
            probabilities = probabilities_for(model, loader, args.device)
            threshold = frozen["thresholds"][key]
            arrays[mode] = probabilities
            metrics[mode] = event_metrics(probabilities, truth, threshold)
            statistics[mode] = [per_source_statistics(probabilities, truth, threshold)]
            if checkpoint["development"] != metrics[mode]:
                # Float kernels can differ slightly across devices; report rather than tune.
                metrics[mode]["saved_dev_f1_absolute_difference"] = abs(metrics[mode]["frame_macro_f1"] - checkpoint["development"]["frame_macro_f1"])
            print(json.dumps({"mode": mode, "metrics": metrics[mode]}), flush=True)
        joint = manifest["selected_mode"]
        threshold = frozen["thresholds"][selected]
        arrays["joint_audio_disabled"] = probabilities_for(models[joint], loader, args.device, "audio")
        metrics["joint_audio_disabled"] = event_metrics(arrays["joint_audio_disabled"], truth, threshold)
        statistics["joint_audio_disabled"] = [per_source_statistics(arrays["joint_audio_disabled"], truth, threshold)]
        calibrations = torch.load(DATA / "calibration_train_only.pt", map_location="cpu", weights_only=True)
        priors = {}
        for mode in ("visual", "audio"):
            state = calibrations[mode]
            calibration = ScoreCalibration(state["mean"], state["std"])
            calibration.load_state_dict(state)
            priors[mode] = calibration(getattr(dataset, f"{mode}_prior"))
        alpha = calibrations["alpha"]
        arrays["static_late_fusion"] = (alpha * priors["visual"] + (1 - alpha) * priors["audio"]).sigmoid().numpy()
        late_threshold = frozen["static_thresholds"]["late_fusion"]
        metrics["static_late_fusion"] = event_metrics(arrays["static_late_fusion"], truth, late_threshold)
        statistics["static_late_fusion"] = [per_source_statistics(arrays["static_late_fusion"], truth, late_threshold)]
        indices = np.random.default_rng(20261107).integers(0, len(dataset), size=(2000, len(dataset)))
        comparisons = {key: bootstrap_pair(statistics[joint], value, indices) for key, value in statistics.items() if key != joint}
        np.savez_compressed(args.output / "dev_predictions.npz", **arrays, labels=truth, ids=np.array(dev_ids))
        save(args.output / "cached_head_results.json", {"methods": metrics, "paired_source_bootstrap95ci": comparisons,
            "scope": "Cached development features only; raw-media consistency is audited separately"})
        replays = []
        if args.replay_sources:
            save(status, {"status": "running", "phase": "raw_media_replay", "dev_sources_complete": len(dataset)})
            runtime = EventRuntime(folder, args.device)
            if runtime.seed != manifest["deployment_seed"] or runtime.mode != joint or runtime.threshold != threshold:
                raise ValueError("Production runtime differs from the evaluated head")
            for index in sorted(range(len(dev_ids)), key=lambda i: hashlib.sha256(("av-dev-replay-v1:" + dev_ids[i]).encode()).hexdigest())[:args.replay_sources]:
                probability = runtime.probabilities(decode_window(DATA / "videos" / f"{dev_ids[index]}.mp4", 0.))
                difference = float(np.abs(probability - arrays[joint][index]).max())
                agreement = bool(np.array_equal(probability >= threshold, arrays[joint][index] >= threshold))
                replays.append({"video": dev_ids[index], "max_probability_absolute_difference": difference,
                    "all_frame_class_decisions_identical": agreement,
                    "passed_fixed_tolerance": difference <= .003 and agreement})
                save(args.output / "raw_media_replays.json", replays)
                if difference > .003 or not agreement:
                    raise ValueError(f"Raw-media replay differs from cached head: {dev_ids[index]}, max difference {difference}, decision agreement {agreement}")
                print(json.dumps({"raw_replays_complete": len(replays), "raw_replays_total": min(args.replay_sources, len(dev_ids))}), flush=True)
        if hashes != {str(path.relative_to(PROJECT)): sha(path) for path in sources}:
            raise ValueError("Source changed during validation")
        report = {"scope": "AVE-derived 28-class development event-head evaluation; not full Agent delivery",
            "source_videos": len(dataset), "seed": manifest["deployment_seed"], "deployed_mode": joint,
            "methods": metrics, "paired_source_bootstrap95ci": comparisons, "raw_media_replays": replays,
            "failed_or_excluded_sources": 0, "models_or_thresholds_tuned": False, "test_features_accessed": False,
            "paid_api_calls": 0, "elapsed_seconds": time.monotonic() - started,
            "limits": ["Dev used previously for checkpoint/threshold selection; not independent generalization evidence",
                "Cached-head event decoder ranks by excess-probability sum; production candidate ranking/padding is not measured here",
                "Raw-media replay is a numerical consistency audit, not another independent accuracy set",
                "Closed 28-class taxonomy, one-second labels, no arbitrary query/dialogue or Agent/export evaluation",
                "AudioSet pretrained backbone may have seen AVE source collection",
                "Unannotated classes are benchmark negatives, not exhaustively verified absences"]}
        save(args.output / "results.json", report)
        save(status, {"status": "complete", "sources_complete": len(dataset), "raw_replays_complete": len(replays)})
        print(json.dumps({"results": str(args.output / "results.json"), "methods": metrics}), flush=True)
    except BaseException as error:
        save(status, {"status": "failed", "reason": str(error)[:1000]})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    parser.add_argument("--replay-sources", type=int, default=3)
    arguments = parser.parse_args()
    if arguments.replay_sources < 0:
        parser.error("replay-sources must be nonnegative")
    run(arguments)
