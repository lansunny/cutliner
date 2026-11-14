"""Version/gallery checks with real FFmpeg, MOCK semantic reviews and no API calls."""

import argparse
import asyncio
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.request import urlopen
from unittest.mock import patch
from uuid import uuid4

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

import studio_server as studio
import mini_agent.video.local_search as search
from mini_agent.video.review_agent import ClipReview, VideoReviewAgent
from mini_agent.video.schemas import ExportedVideoClip, VideoTaskSpec
from mini_agent.video.studio_history import read_json, save_json, source_signature
from studio_jobs import StudioJobs


def main(args):
    folder = PROJECT / "runs" / f"version_check_{uuid4().hex[:12]}"
    original = {name: (args.run / name).read_bytes() for name in
                ("metadata.json", "task.json", "candidates.json", "retrieval.json", "retrieval_state.json")}
    case = folder / "case"
    case.mkdir(parents=True)
    for name, content in original.items():
        save_json(case / name, json.loads(content))
    task_record = read_json(case / "task.json")
    task_record["request"] = task_record["task"]["query"] = "离线成片版本验证（MOCK，不代表语义识别结果）"
    save_json(case / "task.json", task_record)
    studio.JOB_MANAGER = StudioJobs(folder / "jobs", studio.JOBS, studio.LOCK, studio.BUSY)
    run = studio.load_run(case)
    candidate = run["candidates"][0]
    task = VideoTaskSpec.model_validate(run["task"])

    def key():
        value = uuid4().hex
        studio.JOBS[value] = {"id": value, "events": [], "status": "running", "started": time.time()}
        return value

    class MockAgent:
        change_output = False

        def __init__(self, *args, **kwargs):
            pass

        async def review_interval(self, task, video, candidate_id, directory, start, end, **kwargs):
            return ClipReview(candidate_id=candidate_id, file_path=video, verdict="pass", sampled_frames=2,
                              reason="MOCK_ONLY source verdict, not semantic accuracy")

        async def review_clip(self, task, clip, directory, **kwargs):
            if self.change_output:
                stat = clip.file_path.stat()
                os.utime(clip.file_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
            return ClipReview(candidate_id=clip.candidate_id, file_path=clip.file_path, verdict="pass", sampled_frames=2,
                              reason="MOCK_ONLY output verdict, not semantic accuracy")

        async def close(self):
            pass

    body = {"run_id": run["id"], "candidate_id": candidate["candidate_id"], "allow_cloud": True,
            "start": candidate["start_seconds"] + .3, "end": candidate["end_seconds"] - .3}
    with patch.object(search, "VideoReviewAgent", MockAgent), patch.object(search.Settings, "from_env", return_value=None):
        reviewed = asyncio.run(studio.review(key(), body))
    first = reviewed["run"]["deliveries"][0]
    assert first["version"] == 1 and first["reviewed"] and first["file_status"] == "current"
    manual = studio.export_trim(key(), {**body, "start": body["start"] + .5})
    current = manual["run"]
    second, first = current["deliveries"]
    assert [second["version"], first["version"]] == [2, 1]
    assert not second["reviewed"] and first["reviewed"] and current["delivery"]["id"] == second["id"]
    assert first["start"] != second["start"] and first["media"]["url"] != second["media"]["url"]
    stable_ids = [item["id"] for item in current["deliveries"]]
    assert [item["id"] for item in studio.load_run(case)["deliveries"]] == stable_ids

    # A clean, isolated preview snapshot, before any destructive fault simulation.
    preview_run = folder / "preview_run"
    shutil.copytree(case, preview_run)
    save_json(folder / "preview/studio_data/jobs/versions_preview.json", {
        "id": "versions_preview", "kind": "export", "status": "complete", "stage": "MOCK版本验证完成",
        "started": time.time(), "events": current["events"], "source_signature": source_signature(read_json(case / "metadata.json")["file_path"]),
        "root_run_path": str(preview_run), "media_id": current["media_id"], "run_id": "preview_run", "result": manual})

    server = studio.ThreadingHTTPServer(("127.0.0.1", 0), studio.Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def get(path):
        with urlopen(base + path, timeout=8) as response:
            return json.load(response)

    try:
        assert [item["id"] for item in get(f"/api/runs/{run['id']}")["deliveries"]] == stable_ids
        first_path = Path(first["media"]["path"])
        first_stat = first_path.stat()
        try:
            os.utime(first_path, ns=(first_stat.st_atime_ns, first_stat.st_mtime_ns + 1_000_000))
            changed = get(f"/api/runs/{run['id']}")
            record = next(item for item in changed["deliveries"] if item["id"] == first["id"])
            assert record["file_status"] == "changed" and record["media"] is None and not record["reviewed"]
            assert changed["delivery"]["id"] == second["id"]
            try:
                urlopen(base + first["media"]["url"], timeout=8)
            except HTTPError as error:
                assert error.code == 400
            else:
                raise AssertionError("An old media URL served a changed file")
            clip = ExportedVideoClip.model_validate(read_json(case / first["operation_id"] / "exports.json")["clips"][0])
            reviewer = VideoReviewAgent.__new__(VideoReviewAgent)
            blocked = asyncio.run(reviewer.review_clip(task, clip, folder / "never_opened"))
            assert blocked.verdict == "uncertain" and blocked.sampled_frames == 0
        finally:
            os.utime(first_path, ns=(first_stat.st_atime_ns, first_stat.st_mtime_ns))

        second_path = Path(second["media"]["path"])
        backup = second_path.with_suffix(".test-backup")
        try:
            second_path.replace(backup)
            missing = get(f"/api/runs/{run['id']}")
            assert missing["delivery"] is None and missing["latest_artifact_id"] == second["id"]
            assert missing["deliveries"][0]["file_status"] == "missing"
            assert missing["deliveries"][1]["available"]  # Do not silently substitute the old output.
        finally:
            backup.replace(second_path)

        # Legacy JSON has no file binding. Keep it viewable, never invent proof.
        report_path = case / first["operation_id"] / "exports.json"
        state_path = case / first["operation_id"] / "local_search.json"
        report, state = read_json(report_path), read_json(state_path)
        report["clips"][0].pop("file_signature", None)
        for attempt in state["attempts"]:
            for clip in attempt.get("exports", {}).get("clips", []):
                clip.pop("file_signature", None)
        save_json(report_path, report); save_json(state_path, state)
        legacy = get(f"/api/runs/{run['id']}")
        old = next(item for item in legacy["deliveries"] if item["id"] == first["id"])
        assert old["file_status"] == "legacy" and old["available"] and old["was_reviewed"] and not old["reviewed"]

        MockAgent.change_output = True
        with patch.object(search, "VideoReviewAgent", MockAgent), patch.object(search.Settings, "from_env", return_value=None):
            changed_during_review = asyncio.run(studio.review(key(), {**body, "start": body["start"] + .2}))
        assert not changed_during_review["reviewed"] and len(changed_during_review["run"]["deliveries"]) == 2
        assert changed_during_review["run"]["review_status"] == "output_changed"
        assert all((args.run / name).read_bytes() == content for name, content in original.items())
        summary = {"status": "passed", "distinct_versions_and_ranges_survive_reload": True,
            "historical_review_does_not_transfer_to_manual_export": True, "changed_files_lose_review_and_old_media_url": True,
            "missing_latest_output_does_not_silently_fall_back": True, "legacy_versions_not_silently_certified": True,
            "file_changed_during_review_never_delivered": True, "changed_file_rejected_before_api_call": True,
            "semantic_reviews": "MOCK ONLY", "paid_api_calls": 0, "folder": str(folder)}
        save_json(folder / "version_check.json", summary)
        print(json.dumps(summary), flush=True)
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    main(parser.parse_args())
