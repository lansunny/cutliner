"""Isolated HTTP/restart checks; MOCK planning/reviews, real FFmpeg exports.

All mutable state and copied media live under runs/http_check_*. No cloud calls,
model training, real semantic verdicts, or edits to the source retrieval run.
"""

import argparse
import asyncio
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from mini_agent.video.studio_history import read_json, save_json, source_signature


def serve_fixture(folder, port):
    import studio_server as studio
    from studio_jobs import StudioJobs

    def trace(kind, **details):
        path = folder / "mock_calls.json"
        save_json(path, [*read_json(path, []), {"kind": kind, "at": time.time(), **details}])

    class OfflinePlanner:
        def __init__(self, settings):
            pass

        async def parse_task(self, request, metadata):
            trace("MOCK_planner")
            return studio.VideoTaskSpec(query=request, visual_query_en="A person pours into a glass.")

        async def close(self):
            pass

    class OfflineRetriever:
        def __init__(self, **kwargs):
            pass

        async def retrieve(self, task, metadata, **kwargs):
            trace("MOCK_retriever", query=task.query)
            await asyncio.sleep(read_json(folder / "controls.json", {}).get("retrieval_delay_seconds", 12))
            candidates = [studio.ClipCandidate.model_validate(item) for item in read_json(folder / "template_candidates.json")]
            return candidates, {"MOCK_ONLY": True, "source_signature": source_signature(metadata.file_path)}

    async def offline_review(task, metadata, candidates, directory, **kwargs):
        trace("MOCK_review", ranges=[[c.start_seconds, c.end_seconds] for c in candidates])
        await asyncio.sleep(.1)
        if read_json(folder / "controls.json", {}).get("review") != "no_match":
            raise ValueError("MOCK_ONLY intentional review failure; no model API was called")
        save_json(directory / "local_search.json", {"status": "no_match", "attempts": [], "MOCK_ONLY": True})
        return None

    studio.ROOT, studio.DATA = folder, folder / "studio_data"
    studio.MEDIA.clear(); studio.RUNS.clear(); studio.JOBS.clear()
    studio.Settings.from_env = staticmethod(lambda: None)
    studio.VideoTaskPlanner, studio.NeuralVideoRetriever = OfflinePlanner, OfflineRetriever
    studio.deliver_first_match = offline_review
    studio.ALLOWED_ORIGINS.add(f"http://127.0.0.1:{port}")
    studio.JOB_MANAGER = StudioJobs(studio.DATA / "jobs", studio.JOBS, studio.LOCK, studio.BUSY,
                                    on_terminal=studio.terminal_job, on_restore=studio.restore_job)
    studio.seed_library()
    if not any((studio.DATA / "jobs").glob("*.json")):
        studio.load_run(folder / "runs" / "archived_fixture")
    studio.JOB_MANAGER.restore()
    server = studio.ThreadingHTTPServer(("127.0.0.1", port), studio.Handler)
    server.daemon_threads = True
    print(f"MOCK semantic fixture; real local exports: http://127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        studio.JOB_MANAGER.close()


def until(check, seconds=25):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        result = check()
        if result:
            return result
        time.sleep(.08)
    raise AssertionError("HTTP lifecycle verification timed out")


def run_checks(source):
    folder = PROJECT / "runs" / f"http_check_{uuid4().hex[:12]}"
    run = folder / "runs" / "archived_fixture"
    run.mkdir(parents=True)
    metadata = read_json(source / "metadata.json")
    media = folder / "OFFLINE_FIXTURE.mp4"
    shutil.copy2(metadata["file_path"], media)
    metadata["file_path"] = str(media)
    for name in ("task.json", "candidates.json", "retrieval_state.json"):
        shutil.copy2(source / name, run / name)
    task = read_json(run / "task.json")
    task["request"] = "离线接口验证（MOCK，不代表语义识别结果）"
    task["task"]["query"] = task["request"]
    save_json(run / "task.json", task)
    save_json(run / "metadata.json", metadata)
    save_json(run / "retrieval.json", {"source_signature": source_signature(media), "MOCK_ONLY": True})
    shutil.copy2(run / "candidates.json", folder / "template_candidates.json")
    # A completed historical job is the only anchor for this archived run;
    # seed_library deliberately does not include its folder name.
    save_json(folder / "studio_data/jobs" / "archive_anchor.json", {
        "id": "archive_anchor", "kind": "export", "status": "complete", "started": 0,
        "root_run_path": str(run), "source_signature": source_signature(media), "events": []})
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    base, token = f"http://127.0.0.1:{port}", ""
    process, log = None, None

    def request(path, body=None, *, headers=None):
        data = json.dumps(body).encode() if body is not None else None
        fields = {"Content-Type": "application/json", "X-Studio-Token": token, **(headers or {})}
        try:
            with urlopen(Request(base + path, data=data, headers=fields), timeout=8) as response:
                return response.status, json.load(response)
        except HTTPError as error:
            return error.code, json.load(error)

    def start():
        nonlocal process, log, token
        log = (folder / "http.log").open("a", encoding="utf-8")
        process = subprocess.Popen([sys.executable, "-X", "utf8", "-u", str(Path(__file__).resolve()),
            "--serve-fixture", str(folder), "--port", str(port)], cwd=PROJECT,
            stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

        def ready():
            if process.poll() is not None:
                raise AssertionError("Fixture server exited: " + (folder / "http.log").read_text(encoding="utf-8")[-2000:])
            try:
                status, bootstrap = request("/api/bootstrap")
                return bootstrap if status == 200 else None
            except (URLError, TimeoutError):
                return None

        bootstrap = until(ready)
        token = bootstrap["token"]
        return bootstrap

    def stop():
        if process is not None and process.poll() is None:
            process.terminate()  # Exact owned fixture PID, never a user service.
            process.wait(timeout=15)
        if log:
            log.close()

    def terminal(key):
        status, record = request(f"/api/jobs/{key}")
        assert status == 200
        return record if record["status"] not in {"running", "cancelling"} else None

    try:
        bootstrap = start()
        source_media = next(m for m in bootstrap["media"] if m["path"] == str(media))
        current = next(r for r in bootstrap["runs"] if r["id"] == "archived_fixture")
        candidate = current["candidates"][0]
        local = {"media_id": source_media["id"], "query": "离线重启检查", "route": "visual",
                 "visual_query_en": "A person pours into a glass.", "cloud_planning": False}
        assert request("/api/retrieve", local, headers={"X-Studio-Token": "wrong"})[0] == 403
        assert request("/api/retrieve", local, headers={"Origin": "https://untrusted.example"})[0] == 403
        assert request("/api/retrieve", {**local, "cloud_planning": True})[0] == 400
        assert request("/api/auto-edit", {**local, "allow_cloud": False})[0] == 400

        status, submitted = request("/api/retrieve", local)
        assert status == 202
        cancelled = submitted["job_id"]
        until(lambda: any(row["kind"] == "MOCK_retriever" for row in read_json(folder / "mock_calls.json", [])))
        assert request(f"/api/jobs/{cancelled}/cancel", {})[0] == 200
        cancelled_record = until(lambda: terminal(cancelled))
        assert cancelled_record["status"] == "cancelled" and cancelled_record["run"]["review_status"] == "cancelled"

        # Crash only while a MOCK asyncio delay is active: no child model/FFmpeg.
        status, submitted = request(f"/api/jobs/{cancelled}/retry", {})
        assert status == 202 and submitted["job"]["retry_of"] == cancelled
        interrupted = submitted["job_id"]
        until(lambda: len([r for r in read_json(folder / "mock_calls.json", []) if r["kind"] == "MOCK_retriever"]) == 2)
        old_token, old_instance = token, bootstrap["instance_id"]
        count = len(read_json(folder / "mock_calls.json", []))
        stop(); bootstrap = start()
        assert token != old_token and bootstrap["instance_id"] != old_instance
        recovered = terminal(interrupted)
        assert recovered["status"] == "interrupted" and recovered["can_retry"]
        assert recovered["run"]["review_status"] == "interrupted"
        assert bootstrap["active_job"] is None and len(read_json(folder / "mock_calls.json", [])) == count
        assert request(f"/api/jobs/{interrupted}/retry", {}, headers={"X-Studio-Token": old_token})[0] == 403

        # The original run is deliberately outside seed_library's recent-run glob.
        # Its references must return from persisted job state, not a seed shortcut.
        body = {"run_id": "archived_fixture", "candidate_id": candidate["candidate_id"], "allow_cloud": True}
        # Register one export job so archive/root and actual file are durable.
        status, submitted = request("/api/export", {**body, "start": 7.2, "end": 18.4})
        assert status == 202
        export_id = submitted["job_id"]
        output = until(lambda: terminal(export_id))
        assert output["status"] == "complete" and output["result"]["reviewed"] is False
        exported = output["result"]["media"]
        assert Path(exported["path"]).is_file() and abs(exported["duration"] - 11.2) < .06
        stop(); bootstrap = start()
        restored_output = terminal(export_id)
        assert restored_output["status"] == "complete" and restored_output["result"]["media"]["id"] == exported["id"]
        with urlopen(base + exported["url"], timeout=8) as response:
            assert response.status == 200 and response.read(16)

        # No start/end in the request: retry must nevertheless use captured ranges.
        status, submitted = request("/api/review", body)
        assert status == 202
        review_id = submitted["job_id"]
        failed = until(lambda: terminal(review_id))
        original_ranges = failed["scope"]["ranges"]
        assert failed["status"] == "failed" and failed["can_retry"] and failed["requires_cloud"]
        assert "input" not in failed and "allow_cloud" not in read_json(folder / "studio_data/jobs" / f"{review_id}.json")["input"]
        calls = len(read_json(folder / "mock_calls.json", []))
        assert request(f"/api/jobs/{review_id}/retry", {})[0] == 400
        assert len(read_json(folder / "mock_calls.json", [])) == calls
        edits = read_json(run / "edits.json", {})
        edits[candidate["candidate_id"]]["start_seconds"] += 1
        save_json(run / "edits.json", edits)
        save_json(folder / "controls.json", {"review": "no_match"})
        status, submitted = request(f"/api/jobs/{review_id}/retry", {"allow_cloud": True, "start": 0., "end": 1.})
        assert status == 202 and submitted["job"]["scope"]["ranges"] == original_ranges
        retried = until(lambda: terminal(submitted["job_id"]))
        assert retried["status"] == "complete" and retried["result"]["outcome"] == "no_match"
        reviews = [row for row in read_json(folder / "mock_calls.json") if row["kind"] == "MOCK_review"]
        assert reviews[0]["ranges"] == reviews[1]["ranges"]
        assert request("/api/export", {**body, "start": -1, "end": 1})[0] == 400

        # A changed source cannot reuse the old retrieval/retry authority.
        timestamp = media.stat().st_mtime_ns
        import os
        os.utime(media, ns=(media.stat().st_atime_ns, timestamp + 1_000_000))
        assert request(f"/api/jobs/{review_id}/retry", {"allow_cloud": True})[0] == 400
        os.utime(media, ns=(media.stat().st_atime_ns, timestamp))
        summary = {"status": "passed", "real_http_auth_and_consent_gates": True,
            "cancel_and_retry_use_distinct_jobs": True, "real_process_restart_rotates_token_and_interrupts_job": True,
            "restart_does_not_replay_operations": True, "archived_run_and_actual_output_rebound": True,
            "real_ffmpeg_export_survives_restart": True, "retry_requires_fresh_cloud_consent": True,
            "retry_keeps_original_ranges_despite_edits_or_request_overrides": True,
            "changed_source_rejected": True, "semantic_reviews": "MOCK ONLY", "paid_api_calls": 0,
            "folder": str(folder)}
        save_json(folder / "http_check.json", summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    finally:
        stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--serve-fixture", type=Path)
    parser.add_argument("--port", type=int, default=7862)
    args = parser.parse_args()
    if args.serve_fixture:
        serve_fixture(args.serve_fixture.resolve(), args.port)
    elif args.source_run:
        run_checks(args.source_run.resolve())
    else:
        parser.error("Provide --source-run, or --serve-fixture for isolated browser QA")
