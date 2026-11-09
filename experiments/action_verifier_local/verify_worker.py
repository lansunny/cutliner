"""Local JSON tool process, isolated from the application's Conda environment."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from actions import ACTIONS, OPPOSITES
from encoder import FrozenVideoEncoder
from prepare_model import sha256_file
from rejection_policy import decide
from verifier import ACTION_ORDER, load_head


def inspect(encoder, head, checkpoint, head_digest, video, start, end, action, closed_domain, use_cache=True):
    if action not in ACTIONS:
        return {"decision": "unsupported_action", "action": action}
    started = time.perf_counter()
    feature, logits, metadata, hit = encoder.encode(video, start, end, use_cache=use_cache)
    probabilities = torch.from_numpy(logits).softmax(-1)
    target_id = encoder.action_labels[action]
    opposite = OPPOSITES[action]
    opposite_id = encoder.action_labels[opposite]
    with torch.inference_mode():
        score = head(torch.from_numpy(feature)[None], torch.tensor([ACTION_ORDER.index(action)])).sigmoid().item()
    result = decide(score, float(probabilities[opposite_id]), int(probabilities.argmax()) == opposite_id,
                    checkpoint["rejection_policy"], closed_domain)
    result.update({"schema_version": 1, "action": action, "opposite_action": opposite,
                   "video": str(Path(video).resolve()), "input": metadata, "head_score": score,
                   "target_backbone_score": float(probabilities[target_id]),
                   "opposite_backbone_score": float(probabilities[opposite_id]),
                   "opposite_is_top1": int(probabilities.argmax()) == opposite_id,
                   "head_sha256": head_digest, "encoder_fingerprint": encoder.fingerprint,
                   "training_manifest_sha256": checkpoint["manifest_sha256"],
                   "policy": checkpoint["rejection_policy"], "cache_hit": hit,
                   "elapsed_seconds": time.perf_counter() - started,
                   "scope": "primary_action_match_not_exhaustive_presence",
                   "warning": "Experimental SSV2-like short-action triage; no OOD or multi-action rejection guarantee"})
    return result


def emit(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float)
    parser.add_argument("--action")
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--closed-domain", action="store_true")
    parser.add_argument("--serve", action="store_true", help="JSON-lines on stdin/stdout; keep one encoder loaded")
    parser.add_argument("--no-cache", action="store_true", help="Benchmark actual decode/encode instead of feature cache")
    args = parser.parse_args()
    if not args.serve and (args.video is None or args.action is None):
        parser.error("One-shot mode requires --video and --action")
    encoder = FrozenVideoEncoder()
    head, checkpoint = load_head(args.head, encoder.fingerprint)
    head_digest = sha256_file(args.head)
    if checkpoint.get("purpose") == "execution_pilot" or "rejection_policy" not in checkpoint:
        raise ValueError("Pilot weights cannot reject candidates; use an evaluated experimental checkpoint")
    if not args.serve:
        emit(inspect(encoder, head, checkpoint, head_digest, args.video, args.start, args.end, args.action, args.closed_domain, not args.no_cache))
        return
    emit({"event": "ready", "encoder_fingerprint": encoder.fingerprint, "head_sha256": head_digest})
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                request = {}
                raise ValueError("Expected a JSON object")
            result = inspect(encoder, head, checkpoint, head_digest, request["video"], float(request["start"]),
                             request["end"], request["action"], request.get("closed_domain") is True,
                             request.get("use_cache", True) is True)
        except (ValueError, KeyError, OSError, RuntimeError) as error:
            result = {"decision": "uncertain", "reason": "local_inference_failed",
                      "error_type": type(error).__name__, "certifies_full_query": False}
        result["request_id"] = request.get("request_id")
        emit(result)


if __name__ == "__main__":
    main()
