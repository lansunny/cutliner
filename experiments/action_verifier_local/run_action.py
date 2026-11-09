"""Inspect a short clip with the pretrained baseline, or a legitimately trained head."""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

from actions import ACTIONS
from encoder import FrozenVideoEncoder
from prepare_model import ROOT, REPOSITORY
from verifier import ACTION_ORDER, load_head


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=ROOT / "data/ssv2_pilot_videos/205868.webm")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, help="Default: entire short input clip (maximum 12s)")
    parser.add_argument("--action", default="push_left_to_right", help="Canonical supported action ID; no silent text translation")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--hwaccel", choices=["auto", "cuda", "none"], default="auto")
    parser.add_argument("--head", type=Path)
    parser.add_argument("--compare-order", action="store_true", help="Diagnostic only; reversed clips are not automatically labeled negatives")
    args = parser.parse_args()
    if args.action not in ACTIONS:
        print(json.dumps({"status": "unsupported_action", "action": args.action, "supported": ACTIONS}, ensure_ascii=False, indent=2))
        return
    started = time.perf_counter()
    encoder = FrozenVideoEncoder(args.device, args.hwaccel)
    print(f"Loaded official VideoMAE-Small on {encoder.device}; parameters={sum(p.numel() for p in encoder.model.parameters()):,}", flush=True)
    feature, logits, metadata, cache_hit = encoder.encode(args.video, args.start, args.end)
    probabilities = torch.from_numpy(logits).softmax(-1)
    scores, ids = probabilities.topk(5)
    target_id = encoder.action_labels[args.action]
    report = {
        "mode": "pretrained_174_class_baseline", "video": str(args.video.resolve()), "action": args.action,
        "action_zh": ACTIONS[args.action]["zh"], "input": metadata, "cache_hit": cache_hit,
        "model_repository": REPOSITORY, "target_class_score": float(probabilities[target_id]),
        "top_classes": [{"label": encoder.model.config.id2label[int(index)], "score": float(score)} for score, index in zip(scores, ids)],
        "decision": "unvalidated_baseline", "training_performed_this_run": False,
        "uses_trained_head": args.head is not None,
        "limitations": ["Closed action vocabulary, not arbitrary-text verification.",
                        "Classification scores do not establish that an action occurred.",
                        "Scores are diagnostic; quantitative ground-truth comparisons use the labeled manifest.",
                        "No calibrated abstention or full-process/identity guarantee."],
    }
    if args.head:
        head, checkpoint = load_head(args.head, encoder.fingerprint)
        with torch.inference_mode():
            score = head(torch.from_numpy(feature).unsqueeze(0), torch.tensor([ACTION_ORDER.index(args.action)])).sigmoid().item()
        report.update({"mode": "trained_action_matching_head", "head_score": score,
                       "decision": "needs_review", "threshold_selected_on_val": checkpoint["threshold"],
                       "validation_metrics": checkpoint["validation_metrics"],
                       "note": "Threshold is supplied for diagnostics, not an OOD/uncertainty guarantee."})
    if args.compare_order:
        _, reversed_logits, _, _ = encoder.encode(args.video, args.start, args.end, order="reverse")
        reversed_prob = torch.from_numpy(reversed_logits).softmax(-1)
        report["reverse_diagnostic"] = {"target_score": float(reversed_prob[target_id]),
                                        "note": "No correctness claim without action-specific ground truth."}
    report["elapsed_seconds"] = time.perf_counter() - started
    report["peak_allocated_gpu_mib"] = torch.cuda.max_memory_allocated() / 1024**2 if encoder.device == "cuda" else None
    folder = ROOT / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True)
    output = folder / "result.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Result: {output}")


if __name__ == "__main__":
    main()
