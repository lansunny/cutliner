"""Optional local action tool. No torch dependency or paid API in this adapter."""

import asyncio
import json
import os
import time
from uuid import uuid4
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
EXPERIMENT = PROJECT / "experiments" / "action_verifier_local"
DEFAULT_PYTHON = PROJECT / "experiments" / "video_grounding_local" / ".venv" / "Scripts" / "python.exe"
# Exact phrases only. Never silently strip qualifiers, negation or conjunctions.
QUERY_ACTIONS = {
    "push_left_to_right": "push_left_to_right", "从左向右推动物体": "push_left_to_right",
    "push_right_to_left": "push_right_to_left", "从右向左推动物体": "push_right_to_left",
    "cover": "cover", "覆盖物体": "cover",
    "uncover": "uncover", "揭开覆盖物": "uncover",
}


class LocalActionVerifier:
    def __init__(self, head: Path, python: Path = DEFAULT_PYTHON, closed_domain: bool = False,
                 process_mode="resident", feature_cache=True):
        self.head, self.python = Path(head).resolve(), Path(python).resolve()
        self.closed_domain = closed_domain
        if process_mode not in ("resident", "one-shot"):
            raise ValueError("Unknown worker mode")
        self.process_mode = process_mode
        self.feature_cache = feature_cache
        self._worker = None
        self._lock = asyncio.Lock()

    @classmethod
    def from_env(cls):
        head = os.getenv("LOCAL_ACTION_HEAD", "").strip()
        if not head:
            return None  # Legacy behavior unchanged unless explicitly configured.
        return cls(Path(head), Path(os.getenv("LOCAL_ACTION_PYTHON", str(DEFAULT_PYTHON))),
                   os.getenv("LOCAL_ACTION_DOMAIN", "") == "ssv2_single_action")

    async def _close_worker(self):
        worker, self._worker = self._worker, None
        if worker is None:
            return
        if worker.returncode is None:
            if worker.stdin:
                worker.stdin.close()
            try:
                await asyncio.wait_for(worker.wait(), timeout=5)
            except asyncio.TimeoutError:
                worker.kill()
                await worker.wait()

    async def close(self):
        async with self._lock:
            await self._close_worker()

    async def _resident_request(self, request):
        cold_start = self._worker is None or self._worker.returncode is not None
        if cold_start:
            await self._close_worker()
            self._worker = await asyncio.create_subprocess_exec(
                str(self.python), str(EXPERIMENT / "verify_worker.py"), "--head", str(self.head), "--serve",
                cwd=str(EXPERIMENT), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL)
            ready = json.loads((await self._worker.stdout.readline()).decode("utf-8"))
            if ready.get("event") != "ready":
                raise ValueError("Worker did not become ready")
        self._worker.stdin.write((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))
        await self._worker.stdin.drain()
        result = json.loads((await self._worker.stdout.readline()).decode("utf-8"))
        if result.get("request_id") != request["request_id"]:
            raise ValueError("Worker response identity mismatch")
        result["worker_cold_start"] = cold_start
        return result

    async def inspect(self, query: str, video: Path, start: float, end: float) -> dict:
        # Serialize requests so cancellation/response ordering cannot mix candidates.
        async with self._lock:
            return await self._inspect(query, video, start, end)

    async def _inspect(self, query: str, video: Path, start: float, end: float) -> dict:
        action = QUERY_ACTIONS.get(query.strip().rstrip("。.!！"))
        if action is None:
            return {"decision": "unsupported_action", "reason": "query_outside_exact_closed_vocabulary",
                    "certifies_full_query": False}
        if not 0 < end - start <= 12:
            return {"decision": "uncertain", "reason": "interval_outside_local_short_action_scope",
                    "certifies_full_query": False}
        if not self.python.is_file() or not self.head.is_file():
            return {"decision": "uncertain", "reason": "local_tool_not_configured", "certifies_full_query": False}
        command = [str(self.python), str(EXPERIMENT / "verify_worker.py"), "--video", str(Path(video).resolve()),
                   "--start", str(start), "--end", str(end), "--action", action, "--head", str(self.head)]
        if self.closed_domain:
            command.append("--closed-domain")
        if not self.feature_cache:
            command.append("--no-cache")
        process = None
        started = time.perf_counter()
        try:
            if self.process_mode == "resident":
                request = {"request_id": uuid4().hex, "video": str(Path(video).resolve()), "start": start, "end": end,
                           "action": action, "closed_domain": self.closed_domain, "use_cache": self.feature_cache}
                result = await asyncio.wait_for(self._resident_request(request), timeout=180)
            else:
                process = await asyncio.create_subprocess_exec(*command, cwd=str(EXPERIMENT),
                                                              stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                stdout, _stderr = await asyncio.wait_for(process.communicate(), timeout=180)
                if process.returncode != 0:
                    return {"decision": "uncertain", "reason": "local_tool_failed", "exit_code": process.returncode,
                            "certifies_full_query": False}
                result = json.loads(stdout.decode("utf-8"))
            if result.get("decision") not in ("reject", "uncertain", "unsupported_action"):
                raise ValueError("Invalid local tool verdict")
            if result["decision"] == "reject" and (not self.closed_domain or not result.get("closed_domain_opt_in")):
                raise ValueError("Rejection requires explicit domain opt-in")
            result["tool_wall_seconds_including_process_start"] = time.perf_counter() - started
            result["worker_mode"] = self.process_mode
            return result
        except asyncio.CancelledError:
            await self._close_worker()
            raise
        except (OSError, ValueError, asyncio.TimeoutError):
            await self._close_worker()
            return {"decision": "uncertain", "reason": "local_tool_unavailable", "certifies_full_query": False}
        finally:
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()
