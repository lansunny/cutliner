"""Paired architecture ablations under the same representative training protocol."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from prepare_data import DEFAULT_ROOT, dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve previous ablation rounds; choose a new output")
    args.output.mkdir(parents=True)
    here = Path(__file__).resolve().parent
    train_candidates = args.root / "experiments/real_coarse_train_v1"
    dev_candidates = args.root / "experiments/real_coarse_dev_v1"
    jobs = []
    for name, extra, is_v1 in ((f"marginal_s{args.seed}", ["--marginal-only"], True),
                                (f"no_query_s{args.seed}", ["--backbone-variant", "no_query"], False)):
        run = args.output / name
        cache = args.output / (name + "_actual_cache")
        jobs.append((name, [str(here / "train_interval.py"), "--root", str(args.root),
                     "--actual-candidates", str(train_candidates), "--output", str(run),
                     "--epochs", str(args.epochs), "--seed", str(args.seed), *extra]))
        jobs.append((name + "_eval", [str(here / "cache_interval_eval.py"), "--root", str(args.root),
                     "--run", str(run), "--candidates", str(dev_candidates), "--output", str(cache),
                     *(["--v1"] if is_v1 else [])]))
        jobs.append((name + "_policy", [str(here / "calibrate_policy.py"), "--cache", str(cache),
                     "--output", str(args.output / (name + "_policy"))]))
    started = time.monotonic()
    for name, command in jobs:
        print("Starting " + name, flush=True)
        with (args.output / (name + ".log")).open("w", encoding="utf-8") as log:
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
                raise RuntimeError(f"Ablation failed; inspect {name}.log")
    report = {}
    for name in (f"marginal_s{args.seed}", f"no_query_s{args.seed}"):
        report[name] = {
            "configuration": json.loads((args.output / name / "configuration.json").read_text(encoding="utf-8")),
            "actual": json.loads((args.output / (name + "_actual_cache") / "results.json").read_text(encoding="utf-8")),
            "policy": json.loads((args.output / (name + "_policy") / "calibration.json").read_text(encoding="utf-8"))}
    dump(args.output / "ablation_results.json", report)
    dump(args.output / "status.json", {"status": "complete", "test_accessed": False,
         "elapsed_seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()
