"""Offline lifecycle checks using real owned processes; no model API calls."""

import asyncio
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from uuid import uuid4
from unittest.mock import patch

import imageio_ffmpeg

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from mini_agent.video.job_control import run_process, run_sync, terminate_process
from mini_agent.video.studio_history import read_json, save_json
from studio_jobs import ACTIVE, StudioJobs


def until(check, seconds=25):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = check()
        if value:
            return value
        time.sleep(.03)
    raise AssertionError("Lifecycle check timed out")


def alive(pid):
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.restype = ctypes.c_void_p
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))
        return code.value == 259
    finally:
        kernel.CloseHandle(ctypes.c_void_p(handle))


def main():
    folder = PROJECT / "runs" / f"job_check_{uuid4().hex[:12]}"
    records, lock, busy = {}, threading.RLock(), threading.Lock()
    manager = StudioJobs(folder / "jobs", records, lock, busy)
    manager.restore()

    def terminal(key):
        return manager.public(key) if records[key]["status"] not in ACTIVE else None

    async def complete(key):
        return {"message": "offline lifecycle check"}

    finished = manager.submit("check", complete, {"query": "local"})["job_id"]
    assert until(lambda: terminal(finished))["status"] == "complete"

    # A Python model-worker analogue owns a child, like a decoder/FFmpeg child.
    owned = []
    entered, cleaning = threading.Event(), threading.Event()

    async def model_worker(key):
        code = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); print(p.pid,flush=True); time.sleep(60)"
        process = await asyncio.create_subprocess_exec(sys.executable, "-c", code,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            owned.extend([process.pid, int(await process.stdout.readline())])
            entered.set()
            await process.wait()
        finally:
            cleaning.set()
            await terminate_process(process)
            await asyncio.sleep(.25)  # Exercise the stopping-but-not-yet-free state.

    model_id = manager.submit("check", model_worker, {"query": "model process cleanup"})["job_id"]
    assert entered.wait(10) and all(alive(pid) for pid in owned)
    assert manager.cancel(model_id)["status"] == "cancelling"
    assert cleaning.wait(5)
    try:
        manager.submit("check", complete, {})
    except ValueError:
        pass
    else:
        raise AssertionError("New job admitted before cancellation cleanup")
    assert until(lambda: terminal(model_id))["status"] == "cancelled"
    assert all(not alive(pid) for pid in owned)

    # A real, deliberately real-time FFmpeg subprocess runs inside a sync exporter.
    ffmpeg_started = threading.Event()
    ffmpeg_pids = []
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    real_popen = subprocess.Popen

    def tracked_popen(command, *args, **kwargs):
        process = real_popen(command, *args, **kwargs)
        if command[0] == ffmpeg:
            ffmpeg_pids.append(process.pid)
            ffmpeg_started.set()
        return process

    async def export_worker(key):
        command = [ffmpeg, "-nostdin", "-v", "error", "-re", "-f", "lavfi",
                   "-i", "testsrc=duration=60:size=128x72:rate=10", "-f", "null", "-"]
        await run_sync(run_process, command, timeout=75, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        raise AssertionError("Cancelled FFmpeg job reached success")

    with patch("mini_agent.video.job_control.subprocess.Popen", tracked_popen):
        ffmpeg_id = manager.submit("export", export_worker, {"query": "real FFmpeg cancellation"})["job_id"]
        assert ffmpeg_started.wait(10)
        manager.cancel(ffmpeg_id)
        assert until(lambda: terminal(ffmpeg_id))["status"] == "cancelled"
    assert ffmpeg_pids and all(not alive(pid) for pid in ffmpeg_pids)

    async def failed(key):
        raise ValueError("offline failure sk-not-a-real-key")

    failed_id = manager.submit("review", failed, {"query": "original request", "allow_cloud": True}, requires_cloud=True)["job_id"]
    public = until(lambda: terminal(failed_id))
    assert public["status"] == "failed" and public["can_retry"]
    assert "sk-" not in public["error"] and "input" not in public
    persisted = read_json(folder / "jobs" / f"{failed_id}.json")
    assert "allow_cloud" not in persisted["input"]

    # Restoring state never starts an operation or reuses a previous consent.
    interrupted_id = uuid4().hex
    save_json(folder / "jobs" / f"{interrupted_id}.json", {"id": interrupted_id,
        "status": "running", "kind": "review", "started": time.time(), "input": {"query": "not replayed"},
        "requires_cloud": True})
    recovered = {}
    restored = StudioJobs(folder / "jobs", recovered, threading.RLock(), threading.Lock())
    restored.restore()
    assert restored.public(interrupted_id)["status"] == "interrupted"
    assert restored.public(interrupted_id)["can_retry"] and not restored.runtimes
    assert restored.public(finished)["status"] == "complete"
    assert not busy.locked()
    rebound = []
    replay = StudioJobs(folder / "jobs", {}, threading.RLock(), threading.Lock(),
                        on_restore=lambda record: rebound.append(record["id"]))
    replay.restore()
    assert set(rebound) == set(recovered) and not replay.runtimes

    # Disk/hook failures must not leave a phantom active job with the lock free.
    with patch("studio_jobs.save_json", side_effect=OSError("offline disk fault")):
        try:
            manager.submit("check", complete, {"query": "never started"})
        except OSError:
            pass
        else:
            raise AssertionError("Unpersisted job was started")
    assert not busy.locked() and not manager.runtimes
    assert all(record["status"] not in ACTIVE for record in records.values())

    def broken_hook(record):
        raise OSError("offline terminal hook fault")

    guarded = StudioJobs(folder / "hook_fault", {}, threading.RLock(), threading.Lock(), on_terminal=broken_hook)
    guarded_id = guarded.submit("check", complete, {"query": "bookkeeping failure"})["job_id"]
    until(lambda: guarded.public(guarded_id)["status"] not in ACTIVE)
    assert guarded.public(guarded_id)["persistence_error"] and not guarded.public(guarded_id)["can_retry"]
    assert read_json(folder / "hook_fault" / f"{guarded_id}.json")["status"] == "failed"

    waiting = threading.Event()

    async def long_task(key):
        waiting.set()
        await asyncio.sleep(60)

    closing_id = manager.submit("check", long_task, {"query": "graceful close"})["job_id"]
    assert waiting.wait(5)
    manager.close()
    assert manager.public(closing_id)["status"] == "cancelled" and not busy.locked()
    summary = {"status": "passed", "owned_model_and_child_stopped": owned,
        "real_ffmpeg_stopped": ffmpeg_pids, "busy_held_until_cleanup": True,
        "restart_marks_orphan_interrupted_without_replay": True, "saved_consent_not_reusable": True,
        "errors_redacted": True, "paid_api_calls": 0, "folder": str(folder)}
    summary.update(terminal_references_rebound_on_restore=True, persistence_faults_do_not_leave_active_jobs=True,
                   failed_bookkeeping_prevents_blind_retry=True, graceful_shutdown_cleans_active_jobs=True)
    save_json(folder / "lifecycle_check.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
