"""Replicate the selected full localizer+independent-ranker pipeline across seeds."""

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
from prepare_data import DEFAULT_ROOT, dump, read_jsonl
from run_candidate_study import averaged_metrics, baseline_rows
from train import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[29, 43])
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("New replication directory required")
    args.output.mkdir(parents=True)
    here, exp = Path(__file__).resolve().parent, args.root / "experiments"
    started = time.monotonic()
    for seed in args.seeds:
        local = args.output / f"local_s{seed}"
        features = args.output / f"features_s{seed}"
        ranker = args.output / f"ranker_s{seed}"
        jobs = [
            (f"local_s{seed}", [str(here / "train_interval.py"), "--root", str(args.root),
              "--actual-candidates", str(exp / "real_coarse_train_v1"), "--output", str(local), "--seed", str(seed)]),
            (f"features_s{seed}", [str(here / "cache_rerank_features.py"), "--root", str(args.root), "--run", str(local),
              "--train-candidates", str(exp / "real_coarse_train_topk_v2"), "--dev-candidates", str(exp / "real_coarse_dev_topk_v2"),
              "--output", str(features)]),
            (f"ranker_s{seed}", [str(here / "train_candidate.py"), "--features", str(features), "--output", str(ranker),
              "--variant", "independent", "--seed", str(seed)])]
        for name, command in jobs:
            print("Starting " + name, flush=True)
            with (args.output / f"{name}.log").open("w", encoding="utf-8") as log:
                child = subprocess.Popen([sys.executable, "-u", *command], stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
                dump(args.output / "status.json", {"status": "running", "phase": name, "pid": os.getpid(),
                     "child_pid": child.pid, "test_accessed": False})
                for line in child.stdout:
                    log.write(line)
                    log.flush()
                    print(f"[{name}] {line.rstrip()}", flush=True)
                code = child.wait()
                if code:
                    dump(args.output / "status.json", {"status": "failed", "phase": name, "exit_code": code})
                    raise RuntimeError(f"Replication stopped: {name}")
    sources = [(17, exp / "rerank_features_v1", exp / "round_06/independent_s17")]
    sources += [(seed, args.output / f"features_s{seed}", args.output / f"ranker_s{seed}") for seed in args.seeds]
    selected, top1, qualities, per_seed = [], [], [], {}
    for seed, folder, run in sources:
        if json.loads((run / "status.json").read_text())["status"] != "complete":
            raise ValueError("Nonterminal pipeline component")
        data = torch.load(folder / "dev.pt", map_location="cpu", weights_only=True)
        predictions = read_jsonl(run / "best_dev_predictions.jsonl")
        original = baseline_rows(data, "top1")
        quality = baseline_rows(data, "frozen_quality")
        selected.append(predictions)
        top1.append(original)
        qualities.append(quality)
        per_seed[seed] = {"pipeline": summarize(predictions), "fine_top1": summarize(original),
                          "untrained_frozen_quality_ranking": summarize(quality)}
    mean = {}
    for method in ("pipeline", "fine_top1", "untrained_frozen_quality_ranking"):
        mean[method] = {key: {"mean": float(np.mean([v[method][key] for v in per_seed.values()])),
                             "seed_std": float(np.std([v[method][key] for v in per_seed.values()]))}
                        for key in ("iou", "coverage_at_90", "event_coverage", "redundancy_seconds")}
    avg, ref = averaged_metrics(selected), averaged_metrics(top1)
    paired = {key: bootstrap_delta(avg, ref, key) for key in ("iou", "coverage_at_90", "redundancy_seconds")}
    report = {"scope": "full local encoder AND selected ranker seed replication on development only",
              "test_accessed": False, "seeds": list(per_seed), "per_seed": per_seed, "averages": mean,
              "selected_pipeline_minus_fine_top1": paired,
              "remaining": "freeze inference protocol before independent final-test extraction/evaluation"}
    dump(args.output / "replication_results.json", report)
    dump(args.output / "status.json", {"status": "complete", "test_accessed": False,
         "elapsed_seconds": time.monotonic() - started})
    print(json.dumps({"averages": mean, "paired": paired}, indent=2), flush=True)


if __name__ == "__main__":
    main()
