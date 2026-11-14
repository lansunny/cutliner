"""Multi-seed, capacity-controlled proposal interaction study. Final test sealed."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from calibrate_policy import bootstrap_delta
from candidate_model import collate_candidate_sets
from prepare_data import dump, jsonl, read_jsonl
from train import summarize
from train_candidate import selected_rows


def baseline_rows(data, kind):
    result = []
    for start in range(0, len(data), 256):
        batch = collate_candidate_sets(data[start:start + 256])
        if kind == "top1":
            idx = torch.zeros(len(batch["mask"]), dtype=torch.long)
        elif kind in ("fixed_rank2", "fixed_rank3"):
            rank = int(kind[-1]) - 1
            idx = (batch["mask"].sum(-1) - 1).clamp(max=rank)
        else:
            if kind == "clip":
                scores = batch["meta"][..., 17]
            elif kind == "frozen_quality":
                meta = batch["meta"]
                scores = meta[..., 11] + .25 * meta[..., 12] - .25 * meta[..., 13:15].max(-1).values
            else:
                # Explicit oracle upper bound over fixed predictions, not a model.
                scores = batch["true_iou"] + .25 * batch["true_complete"]
            idx = scores.masked_fill(~batch["mask"], -1e4).argmax(-1)
        result.extend(selected_rows(idx, batch))
    return result


def averaged_metrics(predictions):
    lookup = [{x["case_id"]: x for x in rows} for rows in predictions]
    if any(x.keys() != lookup[0].keys() for x in lookup):
        raise ValueError("Seed evaluation cases changed")
    result = []
    for case, reference in lookup[0].items():
        result.append({"case_id": case, "video_id": reference["video_id"], "kind": "proposal_set",
                       "metrics": {key: float(np.mean([x[case]["metrics"][key] for x in lookup])) for key in reference["metrics"]}})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--variants", nargs="+", default=["metadata_only", "independent", "independent_deep", "joint"])
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("New study folder required")
    args.output.mkdir(parents=True)
    torch.set_num_threads(3)
    data = torch.load(args.features / "dev.pt", map_location="cpu", weights_only=True)
    baselines = {}
    for name in ("top1", "fixed_rank2", "fixed_rank3", "clip", "frozen_quality", "oracle"):
        rows = baseline_rows(data, name)
        baselines[name] = summarize(rows)
        jsonl(args.output / f"baseline_{name}.jsonl", rows)
    dump(args.output / "baselines.json", {"scope": "oracle is an upper bound, others are achieved dev results", "baselines": baselines})
    started = time.monotonic()
    here = Path(__file__).resolve().parent
    for variant in args.variants:
        for seed in args.seeds:
            name = f"{variant}_s{seed}"
            command = [sys.executable, "-u", str(here / "train_candidate.py"), "--features", str(args.features),
                       "--output", str(args.output / name), "--variant", variant, "--seed", str(seed)]
            with (args.output / f"{name}.log").open("w", encoding="utf-8") as log:
                child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, encoding="utf-8", errors="replace")
                dump(args.output / "status.json", {"status": "running", "phase": name, "pid": os.getpid(),
                     "child_pid": child.pid, "test_accessed": False})
                for line in child.stdout:
                    log.write(line)
                    log.flush()
                    print(f"[{name}] {line.rstrip()}", flush=True)
                code = child.wait()
                if code:
                    dump(args.output / "status.json", {"status": "failed", "phase": name, "exit_code": code})
                    raise RuntimeError(f"Study failed: {name}")
    summary, average = {}, {}
    for variant in args.variants:
        metrics, predictions = [], []
        for seed in args.seeds:
            run = args.output / f"{variant}_s{seed}"
            if json.loads((run / "status.json").read_text())["status"] != "complete":
                raise ValueError("Cannot summarize a live/nonterminal run")
            metrics.append(json.loads((run / "best_dev.json").read_text())["dev"])
            predictions.append(read_jsonl(run / "best_dev_predictions.jsonl"))
        summary[variant] = {key: {"mean": float(np.mean([x[key] for x in metrics])),
                                      "seed_std": float(np.std([x[key] for x in metrics]))}
                            for key in ("iou", "coverage_at_90", "event_coverage", "redundancy_seconds")}
        average[variant] = averaged_metrics(predictions)
    paired = {}
    if "joint" in average and "independent_deep" in average:
        paired = {m: bootstrap_delta(average["joint"], average["independent_deep"], m)
                  for m in ("iou", "coverage_at_90", "redundancy_seconds")}
    dump(args.output / "study_results.json", {"scope": "development-only architecture search; not independent final-test significance",
         "seeds": args.seeds, "baselines": baselines, "models": summary, "joint_minus_capacity_matched_independent": paired,
         "test_accessed": False, "local_encoder_frozen": True})
    dump(args.output / "status.json", {"status": "complete", "test_accessed": False,
         "elapsed_seconds": time.monotonic() - started})
    print(json.dumps({"models": summary, "paired_interaction_effect": paired}, indent=2), flush=True)


if __name__ == "__main__":
    main()
