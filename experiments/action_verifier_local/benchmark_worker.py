"""Compare identical local inference with/without model-process reuse; no paid API."""

import argparse
import asyncio
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))

from mini_agent.video.action_verifier import LocalActionVerifier
from mini_agent.video.reader import VideoReader


async def run(args):
    # Three actual short videos; every measured call re-decodes and re-encodes.
    sources = [ROOT / f"data/ssv2_pilot_videos/{video_id}.webm" for video_id in (87436, 101859, 205868)]
    intervals = [(path, VideoReader().read_metadata(path).duration_seconds) for path in sources]
    runs = {}
    for mode in ("one-shot", "resident"):
        tool = LocalActionVerifier(args.head, closed_domain=True, process_mode=mode, feature_cache=False)
        measured = []
        try:
            for video, end in intervals:
                result = await tool.inspect("从左向右推动物体", video, 0.0, end)
                if "head_score" not in result or result.get("cache_hit"):
                    raise RuntimeError(f"Invalid benchmark: {result}")
                measured.append(result)
                print(f"{mode} / {video.stem}: {result['tool_wall_seconds_including_process_start']:.3f}s, cache={result['cache_hit']}", flush=True)
        finally:
            await tool.close()
        runs[mode] = measured
    differences = [abs(a["head_score"] - b["head_score"]) for a, b in zip(runs["one-shot"], runs["resident"])]
    same_decisions = all(a["decision"] == b["decision"] for a, b in zip(runs["one-shot"], runs["resident"]))
    old = [r["tool_wall_seconds_including_process_start"] for r in runs["one-shot"]]
    new = [r["tool_wall_seconds_including_process_start"] for r in runs["resident"]]
    report = {"feature_cache_disabled_in_both_modes": True, "identical_source_intervals": True,
              "one_shot_total_seconds": sum(old), "resident_total_seconds_including_first_start": sum(new),
              "resident_first_call_seconds": new[0], "resident_warm_median_seconds": statistics.median(new[1:]),
              "one_shot_median_seconds": statistics.median(old),
              "batch_time_reduction_fraction": 1 - sum(new) / sum(old),
              "max_absolute_head_score_difference": max(differences), "identical_decisions": same_decisions,
              "runs": runs,
              "limitations": "Three short local clips, one timing pass; not long-video throughput or end-to-end Agent speedup"}
    folder = ROOT / "benchmarks" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True)
    (folder / "worker_benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}, indent=2))
    print(f"Benchmark: {folder}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
