"""Offline wiring checks: real FFmpeg exports, MOCK semantic reviews, no API calls."""

import argparse
import asyncio
import json
from pathlib import Path
import sys
from unittest.mock import patch
from uuid import uuid4

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

import studio_server as studio
import mini_agent.video.local_search as search
from mini_agent.video.review_agent import ClipReview
from mini_agent.video.studio_history import read_json, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Existing local neural retrieval run; never edited")
    args = parser.parse_args()
    folder = PROJECT / "runs" / f"scope_check_{uuid4().hex[:12]}"
    folder.mkdir()
    originals = {}
    for name in ("metadata.json", "task.json", "candidates.json", "retrieval.json", "retrieval_state.json"):
        source = args.run / name
        originals[name] = source.read_bytes()
        save_json(folder / name, json.loads(originals[name].decode("utf-8")))
    run = studio.load_run(folder)
    candidate = run["candidates"][0]
    start, end = candidate["start_seconds"] + .3, candidate["end_seconds"] - .3
    assert start < end
    calls = []

    class MockAgent:
        verdict = "pass"

        def __init__(self, *args, **kwargs):
            pass

        async def review_interval(self, task, video, candidate_id, directory, a, b, **kwargs):
            calls.append(("source", a, b))
            return ClipReview(candidate_id=candidate_id, file_path=video, verdict=self.verdict,
                reason="MOCK_ONLY scope wiring check, not semantic evidence", sampled_frames=0)

        async def review_clip(self, task, clip, directory, **kwargs):
            calls.append(("output", clip.source_start_seconds, clip.source_end_seconds))
            assert clip.file_path.is_file()
            return ClipReview(candidate_id=clip.candidate_id, file_path=clip.file_path, verdict=self.verdict,
                reason="MOCK_ONLY output review; actual FFmpeg file exists", sampled_frames=0)

        async def close(self):
            pass

    def key():
        value = uuid4().hex
        studio.JOBS[value] = {"events": [], "status": "running"}
        return value

    body = {"run_id": run["id"], "candidate_id": candidate["candidate_id"], "start": start, "end": end, "allow_cloud": True}
    meta = studio.VideoReader().read_metadata(run["path"] and read_json(folder / "metadata.json")["file_path"])
    invalid = [{"start": None}, {"end": None}, {"start": -1}, {"end": meta.duration_seconds + 1},
               {"start": float("nan")}, {"start": True}, {"candidate_id": "unknown"}, {"candidate_id": None}]
    for mutation in invalid:
        try:
            studio.operation_candidates(run, {**body, **mutation}, meta)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid scope accepted: {mutation}")
    try:
        asyncio.run(studio.review(key(), {**body, "allow_cloud": False}))
    except ValueError:
        pass
    else:
        raise AssertionError("Cloud review accepted without consent")
    assert not calls

    with patch.object(search, "VideoReviewAgent", MockAgent), patch.object(search.Settings, "from_env", return_value=None):
        first = asyncio.run(studio.review(key(), body))
        assert first["reviewed"] and calls == [("source", start, end), ("output", start, end)]
        current = first["run"]["candidates"][0]
        assert current["start_seconds"] == start and current["model_start_seconds"] == candidate["start_seconds"]
        assert current["review_verdict"] == "pass" and current["review_scope"]["start"] == start
        restored = studio.load_run(folder)
        assert restored["delivery"]["reviewed"] and restored["delivery"]["start"] == start

        MockAgent.verdict = "reject"
        second_start = start + .2
        second = asyncio.run(studio.review(key(), {**body, "start": second_start}))
        assert not second["reviewed"] and second["run"]["candidates"][0]["review_verdict"] == "reject"
        assert second["run"]["delivery"]["start"] == start  # Prior file is retained, not relabelled.
        assert second["run"]["candidates"][0]["review_scope"]["start"] == second_start

    manual = studio.export_trim(key(), {**body, "start": start + .4})
    assert manual["reviewed"] is False and manual["run"]["review_status"] == "manual_exported"
    assert manual["run"]["delivery"]["reviewed"] is False
    assert manual["run"]["candidates"][0]["review_verdict"] == "pending"
    assert studio.load_run(folder)["delivery"]["start"] == start + .4
    for name, content in originals.items():
        assert (args.run / name).read_bytes() == content
    assert (folder / "candidates.json").read_text(encoding="utf-8") == json.dumps(json.loads(originals["candidates.json"]), ensure_ascii=False, indent=2)
    signature = read_json(folder / "retrieval.json")
    signature["source_signature"]["mtime_ns"] += 1
    save_json(folder / "retrieval.json", signature)
    try:
        studio.operation_candidates(manual["run"], body, meta)
    except ValueError:
        pass
    else:
        raise AssertionError("Changed source accepted")
    summary = {"status": "passed", "review_scope_matches_manual_range": True, "review_and_export_share_scope": True,
        "original_candidate_evidence_unchanged": True, "review_history_survives_reload": True,
        "old_pass_not_inherited_by_new_scope": True, "manual_export_not_marked_verified": True,
        "invalid_ranges_and_missing_consent_rejected": True, "source_version_checked": True,
        "paid_api_calls": 0, "semantic_reviews": "MOCK ONLY", "folder": str(folder)}
    save_json(folder / "wiring_check.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
