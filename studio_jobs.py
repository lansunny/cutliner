"""Persistent local job records, separate from non-serializable live runtimes."""

import asyncio
import copy
import re
import threading
import time
from uuid import uuid4

from openai import APIError

from mini_agent.video.job_control import Cancellation, current_cancellation
from mini_agent.video.studio_history import read_json, save_json


ACTIVE = {"running", "cancelling"}
RETRYABLE = {"failed", "cancelled", "interrupted"}


def safe_error(error):
    # Never serialize provider response bodies, headers, Settings or env values.
    name = type(error).__name__
    if name in {"PermissionDeniedError", "AuthenticationError", "BadRequestError"}:
        return "模型服务拒绝了本次请求，请检查权限、配置或内容限制后手动重试"
    if isinstance(error, APIError):
        return "模型服务调用失败，请检查网络、配额或配置后明确重试"
    return re.sub(r"sk-[\w-]+", "[hidden]", str(error))[:900] or name


class StudioJobs:
    def __init__(self, directory, records, lock, busy, *, on_terminal=None, on_restore=None):
        self.directory, self.records, self.lock, self.busy = directory, records, lock, busy
        self.on_terminal = on_terminal
        self.on_restore = on_restore
        self.runtimes = {}

    def _save(self, key):
        save_json(self.directory / f"{key}.json", self.records[key])

    def restore(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.lock:
            for path in sorted(self.directory.glob("*.json")):
                record = read_json(path)
                if not isinstance(record, dict) or record.get("id") != path.stem:
                    continue
                self.records[record["id"]] = record
                if record.get("status") in ACTIVE:
                    self._finalize(record["id"], "interrupted", stage="服务重启，任务已中断",
                                   error="任务没有自动重跑；可以检查已有结果后明确重试")
                elif self.on_restore:
                    try:
                        self.on_restore(record)
                    except (OSError, ValueError, KeyError):
                        record.update(source_unavailable=True, restoration_warning="历史素材暂不可用，请检查文件后重新导入")
                        record.pop("run", None)
                        if record.get("result"):
                            record["result"].pop("run", None)

    def public(self, key):
        with self.lock:
            record = copy.deepcopy(self.records[key])
        original_input = record.pop("input", None)
        record.pop("source_signature", None)
        record["can_retry"] = (record.get("status") in RETRYABLE and bool(original_input)
                               and not record.get("source_unavailable") and not record.get("persistence_error"))
        return record

    def _finalize(self, key, status, **details):
        """A failed bookkeeping hook must not strand a running in-memory job."""
        record = self.records[key]
        record.update(status=status, finished=time.time(), **details)
        try:
            if self.on_terminal:
                self.on_terminal(record)
            self._save(key)
        except Exception:
            record.update(status="failed", persistence_error=True, stage="任务记录未完整保存",
                          error="已有文件已保留。请重新连接并检查结果，不要重复提交云端任务")
            try:
                self._save(key)
            except OSError:
                pass  # Disk may be unavailable; live clients still see a terminal state.

    def update(self, key, **values):
        with self.lock:
            self.records[key].update(values, updated=time.time())
            self._save(key)

    def submit(self, kind, operation, body, *, source_signature=None, scope=None, requires_cloud=False, retry_of=None):
        if not self.busy.acquire(blocking=False):
            raise ValueError("另一个任务正在运行或停止中，请等待完成后再开始")
        key = uuid4().hex
        token = Cancellation()
        runtime = {"token": token, "loop": None, "task": None, "cancel_sent": False}
        try:
            with self.lock:
                self.records[key] = {"id": key, "kind": kind, "status": "running", "stage": "准备任务",
                    "started": time.time(), "events": [], "input": copy.deepcopy(body),
                    "source_signature": source_signature, "scope": scope or {}, "requires_cloud": requires_cloud,
                    "retry_of": retry_of, "media_id": body.get("media_id"), "run_id": body.get("run_id"),
                    "request": body.get("query", "")}
                # Stored consent is not reusable authority. Retry requires a new request.
                self.records[key]["input"].pop("allow_cloud", None)
                self.runtimes[key] = runtime
                self._save(key)
        except BaseException:
            with self.lock:
                self.runtimes.pop(key, None)
                if key in self.records:
                    self.records[key].update(status="failed", finished=time.time(), stage="任务未能启动",
                                             persistence_error=True, error="任务记录保存失败，未启动工作进程")
            self.busy.release()
            raise

        def worker():
            binding = current_cancellation.set(token)
            status, details = "failed", {}
            try:
                with asyncio.Runner() as runner:
                    loop = runner.get_loop()

                    async def execute():
                        token.check()
                        return await operation(key)

                    task = loop.create_task(execute())
                    with self.lock:
                        runtime.update(loop=loop, task=task)
                        if token.event.is_set():
                            runtime["cancel_sent"] = True
                            task.cancel()
                    async def wait_task():
                        return await task

                    result = runner.run(wait_task())
                    status, details = "complete", {"result": result, "stage": result.get("message", "任务完成")}
            except asyncio.CancelledError:
                status, details = "cancelled", {"stage": "已停止", "error": None}
            except Exception as error:
                status, details = "failed", {"stage": "任务未完成", "error": safe_error(error)}
            finally:
                # Runner has drained executor threads and subprocess cleanup before unlock.
                current_cancellation.reset(binding)
                with self.lock:
                    try:
                        self._finalize(key, status, **details)
                    finally:
                        self.runtimes.pop(key, None)
                        self.busy.release()

        thread = threading.Thread(target=worker, daemon=False, name=f"studio-{kind}-{key[:8]}")
        runtime["thread"] = thread
        try:
            thread.start()
        except BaseException:
            with self.lock:
                self._finalize(key, "failed", stage="任务未能启动", error="工作线程启动失败")
                self.runtimes.pop(key, None)
                self.busy.release()
            raise
        return {"job_id": key, "job": self.public(key)}

    def close(self):
        with self.lock:
            keys = list(self.runtimes)
            threads = [runtime.get("thread") for runtime in self.runtimes.values()]
        for key in keys:
            self.cancel(key)
        for thread in threads:
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=30)

    def cancel(self, key):
        with self.lock:
            record = self.records[key]
            if record.get("status") not in ACTIVE:
                return self.public(key)
            runtime = self.runtimes.get(key)
            if runtime is None:
                raise ValueError("任务已不在当前服务中执行，请重新连接")
            runtime["token"].cancel()
            record.update(status="cancelling", stage="正在停止并清理工作进程")
            self._save(key)
            if runtime["loop"] is not None and not runtime["cancel_sent"]:
                runtime["cancel_sent"] = True
                loop, task = runtime["loop"], runtime["task"]

                def request_cancel():
                    if not task.done():
                        task.cancel()

                if not loop.is_closed():
                    loop.call_soon_threadsafe(request_cancel)
            return self.public(key)
