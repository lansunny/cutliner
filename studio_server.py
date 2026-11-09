r"""Loopback-only cutliner UI/API bridge. No keys are ever sent to the browser.

Start: D:\CONDA\envs\vlm_agent\python.exe studio_server.py
Build frontend first with `npm run build` in frontend/.
"""
import asyncio
import hashlib
import json
import math
import mimetypes
import os
import re
import secrets
import subprocess
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import cv2
import imageio_ffmpeg
import numpy as np

from mini_agent.config import PROJECT_ROOT, Settings
from mini_agent.video.audio_visual_events import EVENT_NAMES
from mini_agent.video.clip_exporter import VideoClipExporter
from mini_agent.video.clip_validator import ClipSelectionValidator
from mini_agent.video.local_search import deliver_first_match
from mini_agent.video.neural_retriever import NeuralVideoRetriever, route_task
from mini_agent.video.reader import VideoReader
from mini_agent.video.schemas import ClipCandidate, ClipChoice, ClipSelection, VideoTaskSpec
from mini_agent.video.task_planner import VideoTaskPlanner

ROOT = PROJECT_ROOT
DATA = ROOT / "studio_data"
DIST = ROOT / "frontend/dist"
TOKEN = secrets.token_urlsafe(32)
MEDIA, RUNS, JOBS = {}, {}, {}
LOCK = threading.RLock()
BUSY = threading.Lock()
ALLOWED_ORIGINS = {"http://127.0.0.1:7861", "http://localhost:7861", "http://127.0.0.1:5173", "http://localhost:5173"}
VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}


def read_json(path, fallback=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return fallback


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def register_media(path, label=None, category=None):
    path = Path(path).resolve()
    if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError("请选择存在的 MP4 / MOV / MKV / WebM 视频文件")
    signature = f"{path}|{path.stat().st_size}|{path.stat().st_mtime_ns}"
    key = hashlib.sha256(signature.encode()).hexdigest()[:20]
    with LOCK:
        if key in MEDIA:
            return MEDIA[key]
    meta = VideoReader().read_metadata(path)
    item = {"id": key, "name": label or path.name, "filename": path.name,
            "category": category, "path": str(path), "duration": meta.duration_seconds,
            "width": meta.width, "height": meta.height, "fps": meta.fps,
            "size": path.stat().st_size, "url": f"/api/media/{key}",
            "poster": f"/api/thumb/{key}?t={min(2, meta.duration_seconds / 3):.2f}"}
    with LOCK:
        MEDIA[key] = item
    return item


def load_run(folder):
    meta, task = read_json(folder / "metadata.json"), read_json(folder / "task.json", {})
    if not meta or not Path(meta["file_path"]).is_file():
        return None
    media = register_media(meta["file_path"])
    raw = read_json(folder / "candidates.json", [])
    candidates = [c for c in raw if isinstance(c, dict) and c.get("model_evidence")]
    if not candidates:
        return None
    review = read_json(folder / "local_search.json", {})
    verdicts = {}
    for attempt in review.get("attempts", []):
        r = attempt.get("output_review") or attempt.get("local_review") or {}
        verdicts[attempt["candidate_id"]] = r.get("verdict", "pending")
    for c in candidates:
        c["review_verdict"] = verdicts.get(c["candidate_id"], "pending")
    state = read_json(folder / "retrieval_state.json", {})
    item = {"id": folder.name, "media_id": media["id"], "request": task.get("request", ""),
            "task": task.get("task", task), "candidates": candidates,
            "route": state.get("route", "visual"), "review_status": review.get("status", "not_reviewed"),
            "created": datetime.fromtimestamp(folder.stat().st_mtime).strftime("%m.%d %H:%M"),
            "path": str(folder)}
    with LOCK:
        RUNS[item["id"]] = item
    return item


def seed_library():
    for folder in sorted((ROOT / "runs").glob("neural_video_*"), reverse=True)[:24]:
        try:
            load_run(folder)
        except (OSError, ValueError, KeyError):
            continue
    base = ROOT / "experiments/av_grounding_local/data/ave_events_v1"
    # Real local research samples; these labels are dataset metadata, not model verdicts.
    wanted = {"Race car, auto racing": "赛道 / 引擎与速度", "Acoustic guitar": "现场 / 吉他演奏", "Motorcycle": "公路 / 摩托车"}
    annotations = base / "metadata/Annotations.txt"
    for line in annotations.read_text(encoding="utf-8").splitlines() if annotations.is_file() else []:
        fields = line.split("&")
        if len(fields) < 2 or fields[0] not in wanted:
            continue
        path = base / "videos" / (fields[1] + ".mp4")
        if path.is_file():
            register_media(path, wanted.pop(fields[0]), fields[0])
        if not wanted:
            break
    for path in (DATA / "imports").glob("*") if (DATA / "imports").exists() else []:
        try:
            register_media(path)
        except (OSError, ValueError):
            continue


def update_job(key, **values):
    with LOCK:
        JOBS[key].update(values)


def spawn_job(kind, operation):
    if not BUSY.acquire(blocking=False):
        raise ValueError("另一个任务正在运行，请等待完成后再开始")
    key = uuid4().hex
    with LOCK:
        JOBS[key] = {"id": key, "kind": kind, "status": "running", "stage": "准备任务", "started": time.time()}

    def worker():
        try:
            result = operation(key)
            update_job(key, status="complete", stage="任务完成", result=result)
        except Exception as error:
            # Never return repr(Settings), environment variables or API response bodies.
            message = str(error)
            message = re.sub(r"sk-[\w-]+", "[hidden]", message)[:900]
            update_job(key, status="failed", stage="任务停止", error=message)
        finally:
            BUSY.release()

    threading.Thread(target=worker, daemon=True, name=f"studio-{kind}").start()
    return {"job_id": key}


async def retrieve(key, body):
    media = MEDIA[body["media_id"]]
    meta = await asyncio.to_thread(VideoReader().read_metadata, media["path"])
    request = str(body.get("query", "")).strip()
    if not request or len(request) > 2000:
        raise ValueError("请填写 1–2000 字的检索需求")
    if body.get("cloud_planning"):
        update_job(key, stage="正在解析需求")
        planner = VideoTaskPlanner(Settings.from_env())
        try:
            task = await planner.parse_task(request, meta)
        finally:
            await planner.close()
    else:
        route = body.get("route", "visual")
        task = VideoTaskSpec(query=request, visual_query_en=body.get("visual_query_en") or None,
                             audio_condition_type="none" if route == "visual" else "event_activity",
                             audio_event_classes=body.get("events", []) if route != "visual" else [],
                             requires_extra_visual_grounding=route == "hybrid")
    route, _ = route_task(task)
    folder = ROOT / "runs" / f"neural_video_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    folder.mkdir(parents=True)
    save_json(folder / "metadata.json", meta.model_dump(mode="json"))
    save_json(folder / "task.json", {"request": request, "task": task.model_dump(mode="json")})
    save_json(folder / "retrieval_state.json", {"status": "running", "route": route})
    update_job(key, stage={"visual": "正在定位画面", "audio_visual": "正在分析声音与画面", "hybrid": "正在联合检索"}[route])
    candidates, report = await NeuralVideoRetriever(top_k=12).retrieve(task, meta)
    save_json(folder / "candidates.json", [c.model_dump(mode="json") for c in candidates])
    save_json(folder / "retrieval.json", report)
    save_json(folder / "retrieval_state.json", {"status": "complete", "route": route, "candidate_count": len(candidates)})
    run = load_run(folder)
    return {"run": run, "candidate_count": len(candidates), "message": f"已找到 {len(candidates)} 个候选片段，可预览或进一步审核"}


async def review(key, body):
    if body.get("allow_cloud") is not True:
        raise ValueError("Agent 复核会调用已配置的模型 API，需要先明确勾选授权")
    run = RUNS[body["run_id"]]
    source = Path(run["path"])
    meta = VideoReader().read_metadata(MEDIA[run["media_id"]]["path"])
    task = VideoTaskSpec.model_validate(run["task"])
    selected = body.get("candidate_id")
    candidates = [ClipCandidate.model_validate({k: v for k, v in c.items() if k != "review_verdict"})
                  for c in run["candidates"] if not selected or c["candidate_id"] == selected]
    if not candidates:
        raise ValueError("找不到待复核的候选片段")
    folder = source / f"studio_review_{uuid4().hex[:8]}"
    folder.mkdir()
    update_job(key, stage="正在复核片段内容")
    result = await deliver_first_match(task, meta, candidates, folder, use_legacy_action_verifier=False)
    state = read_json(folder / "local_search.json", {})
    if result and result.valid and state.get("status") == "satisfied":
        clip = register_media(result.clips[0].file_path)
        return {"media": clip, "reviewed": True, "message": "片段与成片审核通过", "path": str(folder)}
    return {"reviewed": False, "message": "未获通过：" + state.get("status", "no_match"), "path": str(folder)}


def export_trim(key, body):
    run = RUNS[body["run_id"]]
    candidate = next(c for c in run["candidates"] if c["candidate_id"] == body["candidate_id"])
    raw = {k: v for k, v in candidate.items() if k != "review_verdict"}
    raw.update(start_seconds=float(body["start"]), end_seconds=float(body["end"]), shot_start_seconds=None, shot_end_seconds=None)
    clip = ClipCandidate.model_validate(raw)
    meta = VideoReader().read_metadata(MEDIA[run["media_id"]]["path"])
    task = VideoTaskSpec.model_validate(run["task"])
    selection = ClipSelection(choices=[ClipChoice(candidate_id=clip.candidate_id, reason="用户在时间线中手动确认裁剪范围；未进行语义复核")], summary="手动导出")
    checked = ClipSelectionValidator().validate(task, meta, [clip], selection)
    update_job(key, stage="正在导出成片")
    folder = DATA / "exports" / uuid4().hex
    folder.parent.mkdir(parents=True, exist_ok=True)
    report = VideoClipExporter().export(task, meta, checked, folder)
    save_json(folder / "export.json", report.model_dump(mode="json"))
    if not report.valid or not report.clips:
        raise ValueError("导出校验未通过：" + "；".join(report.errors))
    return {"media": register_media(report.clips[0].file_path), "reviewed": False, "message": "已按当前范围导出，未经内容审核"}


class Handler(BaseHTTPRequestHandler):
    server_version = "Cutliner/0.1"

    def log_message(self, fmt, *args):
        if self.path.startswith(("/api/jobs/", "/api/thumb/")):
            return
        print(fmt % args)

    def valid_host(self):
        return self.headers.get("Host", "").split(":")[0] in {"127.0.0.1", "localhost"}

    def send_json(self, data, status=200):
        raw = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if not self.valid_host():
            return self.send_json({"error": "Invalid host"}, 403)
        parsed = urlparse(self.path)
        parts = parsed.path.strip("/").split("/")
        try:
            if parsed.path == "/api/bootstrap":
                with LOCK:
                    runs = list(RUNS.values())
                    media = list(MEDIA.values())
                return self.send_json({"token": TOKEN, "media": media, "runs": runs,
                    "events": [{"value": k, "label": v} for k, v in EVENT_NAMES.items()],
                    "active_job": next((j for j in JOBS.values() if j["status"] == "running"), None)})
            if len(parts) == 3 and parts[:2] == ["api", "jobs"]:
                with LOCK:
                    return self.send_json(dict(JOBS[parts[2]]))
            if len(parts) == 3 and parts[:2] == ["api", "media"]:
                return self.send_file(Path(MEDIA[parts[2]]["path"]))
            if len(parts) == 3 and parts[:2] == ["api", "thumb"]:
                media = MEDIA[parts[2]]
                at = float(parse_qs(parsed.query).get("t", ["1"])[0])
                if not math.isfinite(at):
                    raise ValueError("Invalid time")
                at = max(0, min(at, media["duration"] - 0.04))
                folder = DATA / "thumbs"
                folder.mkdir(parents=True, exist_ok=True)
                target = folder / f"{media['id']}_{at:.2f}.jpg"
                if not target.exists():
                    capture = cv2.VideoCapture(media["path"])
                    try:
                        capture.set(cv2.CAP_PROP_POS_MSEC, at * 1000)
                        ok, frame = capture.read()
                        if not ok:
                            raise ValueError("无法解码缩略图")
                        h, w = frame.shape[:2]
                        frame = cv2.resize(frame, (640, max(1, round(h * 640 / w))))
                        cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, 83])
                    finally:
                        capture.release()
                return self.send_file(target)
            if len(parts) == 3 and parts[:2] == ["api", "waveform"]:
                media = MEDIA[parts[2]]
                folder = DATA / "waveforms"
                folder.mkdir(parents=True, exist_ok=True)
                target = folder / (media["id"] + ".json")
                result = read_json(target)
                if result is None:
                    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin", "-i", media["path"], "-t", "7200", "-vn", "-ac", "1", "-ar", "1000", "-f", "s16le", "pipe:1"]
                    process = subprocess.run(command, capture_output=True, timeout=120, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    samples = np.frombuffer(process.stdout, dtype=np.int16).astype(np.float32)
                    peaks = [float(np.sqrt(np.mean(x * x))) if x.size else 0. for x in np.array_split(samples, 220)]
                    maximum = max(peaks) or 1.
                    result = {"values": [round(x / maximum, 4) for x in peaks], "has_audio": bool(samples.size), "duration": min(7200, media["duration"])}
                    save_json(target, result)
                return self.send_json(result)
            if parsed.path.startswith("/api/"):
                return self.send_json({"error": "Not found"}, 404)
            requested = (DIST / parsed.path.lstrip("/")).resolve()
            if not requested.is_relative_to(DIST.resolve()):
                return self.send_json({"error": "Forbidden"}, 403)
            return self.send_file(requested if requested.is_file() else DIST / "index.html")
        except (KeyError, StopIteration):
            self.send_json({"error": "资源不存在，请刷新素材库"}, 404)
        except (ValueError, OSError) as error:
            self.send_json({"error": str(error)[:500]}, 400)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        if not self.valid_host() or self.headers.get("X-Studio-Token") != TOKEN or (self.headers.get("Origin") and self.headers["Origin"] not in ALLOWED_ORIGINS):
            return self.send_json({"error": "本地请求验证失败，请刷新页面"}, 403)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if self.path == "/api/import/upload":
                if not 0 < length <= 4 * 1024**3:
                    raise ValueError("文件须小于 4 GB；更大素材请通过本地路径导入")
                name = Path(self.headers.get("X-Filename", "video.mp4")).name
                suffix = Path(name).suffix.lower()
                if suffix not in VIDEO_SUFFIXES:
                    raise ValueError("不支持的视频格式")
                folder = DATA / "imports"
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / (uuid4().hex + suffix)
                remaining = length
                with path.open("xb") as stream:
                    while remaining:
                        chunk = self.rfile.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise ValueError("上传中断")
                        stream.write(chunk)
                        remaining -= len(chunk)
                return self.send_json({"media": register_media(path, name)})
            if not 0 < length <= 32768:
                raise ValueError("请求内容无效")
            body = json.loads(self.rfile.read(length))
            if self.path == "/api/import/path":
                return self.send_json({"media": register_media(str(body.get("path", "")).strip().strip('"'))})
            if self.path == "/api/retrieve":
                if body.get("media_id") not in MEDIA:
                    raise ValueError("请先导入素材")
                return self.send_json(spawn_job("retrieval", lambda key: asyncio.run(retrieve(key, body))), 202)
            if self.path in {"/api/review", "/api/export"}:
                if body.get("run_id") not in RUNS:
                    raise ValueError("请先选择有效的检索结果")
                op = (lambda key: asyncio.run(review(key, body))) if self.path.endswith("review") else (lambda key: export_trim(key, body))
                return self.send_json(spawn_job("review" if self.path.endswith("review") else "export", op), 202)
            self.send_json({"error": "Not found"}, 404)
        except (ValueError, OSError, KeyError) as error:
            self.send_json({"error": str(error)[:500]}, 400)

    def send_file(self, path):
        if not path.is_file():
            return self.send_json({"error": "请先在 frontend 中执行 npm run build"}, 404)
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        header = self.headers.get("Range", "")
        if header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", header)
            if not match or not any(match.groups()):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            a, b = match.groups()
            if a:
                start, end = int(a), min(int(b), end) if b else end
            else:
                start = max(0, size - int(b))
            if start > end or start >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            status = 206
        self.send_response(status)
        self.send_header("Content-Type", mimetypes.guess_type(str(path))[0] or "application/octet-stream")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "private, max-age=3600" if path.suffix in {".jpg", ".mp4"} else "no-cache")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        try:
            with path.open("rb") as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining:
                    data = stream.read(min(256 * 1024, remaining))
                    if not data:
                        break
                    self.wfile.write(data)
                    remaining -= len(data)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    seed_library()
    server = ThreadingHTTPServer(("127.0.0.1", 7861), Handler)
    server.daemon_threads = True
    print("cutliner: http://127.0.0.1:7861", flush=True)
    print("Local media stays on this computer. Cloud planning/review requires explicit UI opt-in.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
