"""Cooperative cancellation shared by one studio job and its owned workers."""

import asyncio
from contextvars import ContextVar
import subprocess
import threading
import time


class JobCancelled(asyncio.CancelledError):
    pass


class Cancellation:
    def __init__(self):
        self.event = threading.Event()

    def cancel(self):
        self.event.set()

    def check(self):
        if self.event.is_set():
            raise JobCancelled("任务已取消")


current_cancellation = ContextVar("studio_cancellation", default=None)


def checkpoint():
    token = current_cancellation.get()
    if token is not None:
        token.check()


def stop_owned_process(pid):
    """Only call for a subprocess created and still owned by this operation."""
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       creationflags=subprocess.CREATE_NO_WINDOW, timeout=20)


async def terminate_process(process, *, stderr_owned=False):
    if process.returncode is None:
        if hasattr(subprocess, "CREATE_NO_WINDOW"):
            await asyncio.to_thread(stop_owned_process, process.pid)
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
    if stderr_owned:
        await process.stdout.read()
        await process.wait()
    else:
        await process.communicate()


def run_process(command, *, timeout, **kwargs):
    """subprocess.run equivalent that waits for cancellation cleanup before returning."""
    checkpoint()
    started = time.monotonic()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
    try:
        while True:
            checkpoint()
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            try:
                stdout, stderr = process.communicate(timeout=min(.2, remaining))
                checkpoint()
                return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
            except subprocess.TimeoutExpired:
                if time.monotonic() - started >= timeout:
                    raise
    finally:
        if process.poll() is None:
            try:
                stop_owned_process(process.pid)
            finally:
                if process.poll() is None:
                    process.kill()
                process.communicate()


async def run_sync(function, *args, **kwargs):
    """Cancel an await, but never leave its synchronous exporter running behind it."""
    checkpoint()
    work = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        result = await asyncio.shield(work)
        checkpoint()
        return result
    except asyncio.CancelledError:
        token = current_cancellation.get()
        if token is not None:
            token.cancel()
        try:
            await asyncio.shield(work)
        except (Exception, asyncio.CancelledError):
            pass
        raise
