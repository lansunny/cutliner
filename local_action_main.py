"""Run only the local candidate-triage step; never call paid APIs or export clips."""

import argparse
import asyncio
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from mini_agent.video.action_verifier import LocalActionVerifier
from mini_agent.video.local_search import deliver_first_match
from mini_agent.video.reader import VideoReader
from mini_agent.video.schemas import VideoTaskSpec


@dataclass
class ProposedInterval:
    # A user-supplied candidate, not fabricated visual/locator evidence.
    candidate_id: str
    start_seconds: float
    end_seconds: float
    relevance_score: float = 0.0


async def run(args):
    metadata = VideoReader().read_metadata(args.video)
    end = metadata.duration_seconds if args.end is None else args.end
    if not 0 <= args.start < end <= metadata.duration_seconds + 1e-6:
        raise ValueError("Candidate interval is outside the video")
    folder = Path(__file__).resolve().parent / "video_runs" / f"local_action_{datetime.now():%Y%m%d_%H%M%S_%f}"
    folder.mkdir(parents=True)
    await deliver_first_match(VideoTaskSpec(query=args.query), metadata,
                              [ProposedInterval("user_interval_001", args.start, end)], folder,
                              local_verifier=LocalActionVerifier(args.head, closed_domain=args.closed_domain),
                              allow_remote_review=False)
    print(f"本地工具及控制器记录：{folder / 'local_search.json'}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--query", default="从左向右推动物体")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float)
    parser.add_argument("--closed-domain", action="store_true",
                        help="Opt into experimental rejection ONLY for SSV2-like short single-action clips")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
