"""Execute the frozen held-out pipeline without any optimizer or model selection."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from final_protocol import load_frozen_protocol
from prepare_data import DEFAULT_ROOT, dump, file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_frozen_protocol(args.protocol, args.root)
    if args.output.exists():
        raise ValueError("Preserve all previous final-test artifacts; output must be new")
    args.output.mkdir(parents=True)
    here = Path(__file__).resolve().parent
    coarse = args.output / "coarse_candidates"
    jobs = [
        ("raw_features", [str(here / "extract_features.py"), "--root", str(args.root), "--splits", "test", "--final-protocol", str(args.protocol)]),
        ("coarse_candidates", [str(here / "retrieval_dev.py"), "--root", str(args.root), "--build-candidates", "--keep-all-proposals",
          "--candidate-split", "test", "--final-protocol", str(args.protocol), "--candidates", str(coarse)])]
    for pipeline in protocol["pipelines"]:
        seed = pipeline["seed"]
        jobs.append((f"features_s{seed}", [str(here / "cache_rerank_features.py"), "--root", str(args.root),
                     "--run", str(Path(pipeline["local_checkpoint"]).parent), "--test-candidates", str(coarse),
                     "--splits", "test", "--max-k", str(protocol["top_k"]), "--final-protocol", str(args.protocol),
                     "--output", str(args.output / f"features_s{seed}")]))
    jobs.append(("evaluation", [str(here / "evaluate_final.py"), "--root", str(args.root), "--protocol", str(args.protocol),
                 "--features-root", str(args.output), "--output", str(args.output / "evaluation")]))
    started = time.monotonic()
    for name, command in jobs:
        print("Frozen test stage: " + name, flush=True)
        with (args.output / f"{name}.log").open("w", encoding="utf-8") as log:
            child = subprocess.Popen([sys.executable, "-u", *command], stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            dump(args.output / "status.json", {"status": "running", "stage": name, "pid": os.getpid(),
                 "child_pid": child.pid, "protocol_sha256": file_hash(args.protocol)})
            for line in child.stdout:
                log.write(line)
                log.flush()
                print(f"[{name}] {line.rstrip()}", flush=True)
            code = child.wait()
            if code:
                dump(args.output / "status.json", {"status": "failed", "stage": name, "exit_code": code})
                raise RuntimeError(f"Frozen final stage failed: {name}. Do not change parameters using test scores.")
    dump(args.output / "status.json", {"status": "complete", "test_accessed": True,
         "protocol_sha256": file_hash(args.protocol), "elapsed_seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()
