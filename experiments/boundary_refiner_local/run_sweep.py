"""Run a reproducible first architecture round, serially, with final test sealed."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from extract_features import PROFILE
from prepare_data import DEFAULT_ROOT, dump, read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    report = json.loads((args.root / "features" / PROFILE / "extraction_report.json").read_text(encoding="utf-8"))
    rows = [x for s in ("train", "dev") for x in read_jsonl(args.root / "manifests" / f"{s}.complete.jsonl")]
    if report["status"] != "complete" or report["videos"] != len({x["video_id"] for x in rows}) or report["queries"] != len(rows):
        raise ValueError("Complete train/dev feature cache required; do not silently train on a partial corpus")
    if args.output.exists():
        raise ValueError("Use a new sweep directory, preserving all previous evidence")
    args.output.mkdir(parents=True)
    here = Path(__file__).resolve().parent
    baseline = args.output / "baselines"
    runs = [args.output / f"{v}_s{args.seed}" for v in ("no_query", "film", "cross")]
    tasks = [([str(here / "train.py"), "--root", str(args.root), "--baselines-only", "--output", str(baseline)], "baselines")]
    for variant, path in zip(("no_query", "film", "cross"), runs):
        tasks.append(([str(here / "train.py"), "--root", str(args.root), "--variant", variant,
                       "--seed", str(args.seed), "--epochs", str(args.epochs), "--output", str(path)], path.name))
    tasks.append(([str(here / "compare.py"), "--root", str(args.root), "--baselines", str(baseline),
                   "--runs", *map(str, runs), "--output", str(args.output / "comparison")], "comparison"))
    started = time.monotonic()
    for command, name in tasks:
        print(f"Starting {name}", flush=True)
        with (args.output / f"{name}.log").open("w", encoding="utf-8") as log:
            child = subprocess.Popen([sys.executable, "-u", *command], stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            dump(args.output / "sweep_status.json", {"status": "running", "phase": name,
                 "pid": os.getpid(), "child_pid": child.pid, "test_accessed": False})
            for line in child.stdout:
                log.write(line)
                log.flush()
                print(f"[{name}] {line.rstrip()}", flush=True)
            code = child.wait()
            if code:
                dump(args.output / "sweep_status.json", {"status": "failed", "phase": name,
                     "exit_code": code, "test_accessed": False})
                raise RuntimeError(f"{name} exited {code}; see {args.output / (name + '.log')}")
    dump(args.output / "sweep_status.json", {"status": "complete", "test_accessed": False,
         "elapsed_seconds": round(time.monotonic() - started, 1)})


if __name__ == "__main__":
    main()
