"""Evaluate deployed visual retrieval/refinement on strict DEVELOPMENT data.

Uses actual production inference functions with shared frozen models and source
observations. No training, threshold search, test access, Agent API or exporting.
Shared observation caches make elapsed time unsuitable as a cold-start benchmark.
"""

import argparse
from collections import defaultdict
import contextlib
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
BOUNDARY = PROJECT / "experiments/boundary_refiner_local"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(BOUNDARY))

from mini_agent.video.boundary_refiner import merge_refinements
from calibrate_policy import bootstrap_delta
from predict_boundary import decode_windows, encode_query, encode_visual, load_models, refine_features
from prepare_data import DEFAULT_ROOT, dump, file_hash, jsonl, read_jsonl
from train import interval_metrics, summarize
import retrieve_worker


METHODS = ("coarse_top1", "fine_top1", "deployed_top3")


def interval(candidate):
    return [candidate["start_seconds"], candidate["end_seconds"]]


@torch.inference_mode()
def run(args):
    if args.output.exists():
        raise ValueError("Output already exists; preserve previous validation results")
    manifest_path = BOUNDARY / "approved_boundary_model.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = read_jsonl(args.root / "manifests/dev.complete.jsonl")
    if args.limit_cases:
        rows = rows[:args.limit_cases]
    if not rows or len(rows) != len({r["case_id"] for r in rows}) or any(r["split"] != "dev" for r in rows):
        raise ValueError("Expected unique strict development queries")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(3)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["video_id"]].append(row)
    args.output.mkdir(parents=True)
    source_paths = [Path(__file__), Path(retrieve_worker.__file__), BOUNDARY / "predict_boundary.py",
        PROJECT / "mini_agent/video/boundary_refiner.py", PROJECT / "mini_agent/video/neural_retriever.py",
        PROJECT / "mini_agent/video/local_search.py", PROJECT / "mini_agent/video/schemas.py",
        PROJECT / "experiments/video_grounding_local/run_local.py", BOUNDARY / "model.py",
        BOUNDARY / "interval_model.py", BOUNDARY / "candidate_model.py", BOUNDARY / "train.py",
        BOUNDARY / "calibrate_policy.py", PROJECT / "experiments/video_grounding_local/vendor/UniVTG/run_on_video/clip/clip.py",
        PROJECT / "experiments/video_grounding_local/vendor/UniVTG/run_on_video/clip/model.py",
        PROJECT / "experiments/video_grounding_local/vendor/lighthouse/lighthouse/common/qd_detr.py"]
    hashes = {str(p.relative_to(PROJECT)): file_hash(p) for p in source_paths}
    for path in source_paths:
        dest = args.output / "source_snapshot" / path.relative_to(PROJECT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    config = {"scope": "strict Charades-STA positive development subset, deployed visual route only",
        "created_at": datetime.now(timezone.utc).isoformat(), "queries": len(rows), "videos": len(grouped),
        "pilot": bool(args.limit_cases), "device": device, "deployment_seed": manifest["deployment_seed"],
        "coarse_review_budget": 12, "refinement_budget": 3, "context_seconds": 6., "decoder_quantile": .3,
        "model_manifest": manifest, "model_manifest_sha256": file_hash(manifest_path),
        "dev_manifest_sha256": file_hash(args.root / "manifests/dev.complete.jsonl"),
        "qd_checkpoint_sha256": file_hash(PROJECT / "experiments/video_grounding_local/weights/clip_qd_detr_qvhighlight.ckpt"),
        "source_sha256": hashes, "models_or_thresholds_tuned": False, "test_accessed": False,
        "development_data_previously_used_for_model_selection": True,
        "observation_reuse": "source-global real-frame features shared between queries; query-dependent model inputs unchanged",
        "not_evaluated": ["Agent semantic decisions", "export delivery", "audio/hybrid route", "cold-start latency", "absence detection"]}
    dump(args.output / "configuration.json", config)
    status = args.output / "status.json"
    dump(status, {"status": "running", "phase": "loading_models", "queries_total": len(rows)})
    started = time.monotonic()
    all_rows = {method: [] for method in METHODS}
    completed, encoded_frames = 0, 0
    with (args.output / "pipeline.log").open("w", encoding="utf-8") as log, \
         (args.output / "query_records.jsonl").open("w", encoding="utf-8") as records:
        try:
            with contextlib.redirect_stdout(log):
                local, ranker = load_models(manifest, args.root, device)
                sys.path.insert(0, str(PROJECT / "experiments/video_grounding_local"))
                import run_local as qd
                coarse_model, coarse_config = qd.load_checkpoint(device)
                encoder, _ = qd.clip.load(str(qd.ROOT / "weights/ViT-B-32.pt"), device=device, jit=False)
                encoder.eval().requires_grad_(False)
            runtime = (coarse_model, coarse_config, encoder)
            budget, oracle_context, changed_lengths = [], [], []
            for video_index, cases in enumerate(grouped.values(), 1):
                source = Path(cases[0]["video_path"])
                if file_hash(source) != cases[0]["video_sha256"]:
                    raise ValueError("Development source differs from the audited version")
                duration = cases[0]["duration_seconds"]
                info = retrieve_worker.metadata(source)
                # Training/production share this decoder, crop and quantization;
                # compute once per source instead of re-encoding overlapping crops.
                frames, times = decode_windows(source, [(0., duration)])
                visual = encode_visual(encoder, frames, device)
                del frames
                encoded_frames += len(times)
                for case in cases:
                    request = {"route": "visual", "video": str(source), "start_seconds": 0.,
                        "duration_seconds": duration, "visual_query_en": case["query"], "top_k": 12,
                        "padding_seconds": 1., "device": device, "audio_event_classes": [], "audio_match_mode": "all"}
                    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                        raw = retrieve_worker.retrieve(request, visual_runtime=runtime, source_info=info)
                        proposals = raw["predictions"]
                        if not proposals:
                            raise ValueError("No coarse candidates; cannot silently exclude this query")
                        for i, proposal in enumerate(proposals, 1):
                            proposal["candidate_id"] = f"neural_{i:04d}"
                        text = encode_query(encoder, case["query"], device)
                        refinement = refine_features(manifest, local, ranker, visual, times, text,
                            proposals[:3], duration, case["query"], device)
                        effective = merge_refinements(proposals, refinement, duration)
                    local_first = next((r for r in refinement["refinements"] if r["coarse_index"] == 1), proposals[0])
                    predictions = {"coarse_top1": interval(proposals[0]), "fine_top1": interval(local_first),
                                   "deployed_top3": interval(effective[0])}
                    # Ground truth first enters AFTER every inference/selection.
                    target = case["target_interval_seconds"]
                    per_method = {}
                    for name, pred in predictions.items():
                        if not 0 <= pred[0] < pred[1] <= duration + 1e-5:
                            raise ValueError("A predicted interval exceeds source extent")
                        item = {"case_id": case["case_id"], "video_id": case["video_id"], "kind": "deployed_dev",
                                "prediction": pred, "target": target, "metrics": interval_metrics(pred, target)}
                        all_rows[name].append(item)
                        per_method[name] = item
                    observed = np.zeros(len(times), dtype=bool)
                    for proposal in proposals[:3]:
                        lo, hi = max(0., proposal["start_seconds"] - 6), min(duration, proposal["end_seconds"] + 6)
                        observed |= (times >= lo) & (times < hi)
                    budget.append(int(observed.sum()))
                    oracle_context.append(any(max(0., p["start_seconds"] - 6) <= target[0] < target[1] <= min(duration, p["end_seconds"] + 6)
                                              for p in proposals[:3]))
                    changed_lengths.append((predictions["deployed_top3"][1] - predictions["deployed_top3"][0]) -
                                           (predictions["coarse_top1"][1] - predictions["coarse_top1"][0]))
                    records.write(json.dumps({"case_id": case["case_id"], "video_id": case["video_id"],
                        "query": case["query"], "methods": per_method, "coarse_predictions": proposals,
                        "selected_candidate_id": effective[0]["candidate_id"], "refinement": refinement}, ensure_ascii=False) + "\n")
                    records.flush()
                    completed += 1
                log.flush()
                state = {"status": "running", "phase": "inference", "queries_complete": completed,
                         "queries_total": len(rows), "videos_complete": video_index, "videos_total": len(grouped),
                         "elapsed_seconds": round(time.monotonic() - started, 1), "real_encoded_frames": encoded_frames}
                dump(status, state)
                if video_index % 10 == 0 or video_index == len(grouped):
                    print(json.dumps(state), flush=True)
            if completed != len(rows) or any(len(r) != len(rows) for r in all_rows.values()):
                raise ValueError("Incomplete development evaluation; no silent exclusions")
            for name, values in all_rows.items():
                jsonl(args.output / f"{name}.jsonl", values)
            compared = {baseline: {metric: bootstrap_delta(all_rows["deployed_top3"], all_rows[baseline], metric)
                                  for metric in ("iou", "coverage_at_90", "event_coverage", "redundancy_seconds", "boundary_mae_seconds")}
                        for baseline in ("coarse_top1", "fine_top1")}
            report = {"configuration_sha256": file_hash(args.output / "configuration.json"), "scope": config["scope"],
                "pilot": config["pilot"], "deployment_seed": manifest["deployment_seed"], "queries": completed,
                "videos": len(grouped), "failed_or_silently_excluded_queries": 0,
                "methods": {name: summarize(values) for name, values in all_rows.items()}, "paired_deltas": compared,
                "diagnostics": {"top3_context_contains_target_fraction_ORACLE_NOT_MODEL_ACCURACY": float(np.mean(oracle_context)),
                    "mean_local_union_frames_used": float(np.mean(budget)), "source_frames_encoded_once": encoded_frames,
                    "fraction_outputs_shorter_than_coarse": float(np.mean(np.array(changed_lengths) < -.1)),
                    "fraction_outputs_longer_than_coarse": float(np.mean(np.array(changed_lengths) > .1))},
                "test_accessed": False, "models_or_thresholds_tuned": False, "paid_api_calls": 0,
                "bootstrap": "2000 source-video cluster draws, seed 187", "limits": config["not_evaluated"],
                "elapsed_seconds_with_shared_observations_NOT_COLD_START": time.monotonic() - started}
            for path in source_paths:
                if file_hash(path) != hashes[str(path.relative_to(PROJECT))]:
                    raise ValueError("Inference source changed during validation")
            dump(args.output / "results.json", report)
            dump(status, {"status": "complete", "queries_complete": completed, "queries_total": len(rows),
                          "videos_complete": len(grouped), "elapsed_seconds": round(time.monotonic() - started, 1)})
            print(json.dumps({"queries": completed, "methods": report["methods"], "results": str(args.output / "results.json")}, ensure_ascii=False), flush=True)
        except BaseException as error:
            dump(status, {"status": "failed", "queries_complete": completed, "queries_total": len(rows),
                          "reason": str(error)[:1000]})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--limit-cases", type=int, help="Pilot only; omit for the complete development set")
    run(parser.parse_args())
