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
from mini_agent.video.media_identity import file_version_status, media_id
from mini_agent.video.job_control import checkpoint, run_sync
from mini_agent.video.neural_retriever import NeuralVideoRetriever, route_task
from mini_agent.video.reader import VideoReader
from mini_agent.video.schemas import ClipCandidate, ClipChoice, ClipSelection, VideoTaskSpec
from mini_agent.video.studio_history import (
    begin_operation, finish_operation, load_history, read_json, save_edit, save_json,
    scoped_candidate, source_signature,
)
from mini_agent.video.task_planner import VideoTaskPlanner
from studio_jobs import ACTIVE, RETRYABLE, StudioJobs

ROOT = PROJECT_ROOT
DATA = ROOT / "studio_data"
DIST = ROOT / "frontend/dist"
TOKEN = secrets.token_urlsafe(32)
INSTANCE_ID = uuid4().hex
MEDIA, RUNS, JOBS = {}, {}, {}
LOCK = threading.RLock()
BUSY = threading.Lock()
ALLOWED_ORIGINS = {"http://127.0.0.1:7861", "http://localhost:7861", "http://127.0.0.1:5173", "http://localhost:5173"}
VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}


def register_media(path, label=None, category=None):
    path = Path(path).resolve()
    if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError("请选择存在的 MP4 / MOV / MKV / WebM 视频文件")
    key = media_id(path)
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


def registered_media(key):
    item = MEDIA[key]
    if media_id(item["path"]) != key:
        raise ValueError("文件版本已改变，请刷新素材后再操作")
    return item


def load_run(folder):
    meta, task = read_json(folder / "metadata.json"), read_json(folder / "task.json", {})
    if not meta or not Path(meta["file_path"]).is_file():
        return None
    media = register_media(meta["file_path"])
    raw = read_json(folder / "candidates.json", [])
    metadata = VideoReader().read_metadata(meta["file_path"])
    candidates, delivery, review_status, events, updated, artifacts = load_history(folder, raw, metadata)
    edits = read_json(folder / "edits.json", {})
    active = max(edits, key=lambda key: edits[key]["updated_at"], default=None)
    state = read_json(folder / "retrieval_state.json", {})
    if review_status == "not_reviewed" and state.get("status") in {"cancelled", "interrupted", "failed"}:
        review_status = "operation_failed" if state["status"] == "failed" else state["status"]
    item = {"id": folder.name, "media_id": media["id"], "request": task.get("request", ""),
            "task": task.get("task", task), "candidates": candidates,
            "route": state.get("route", "visual"), "review_status": review_status,
            "created": datetime.fromtimestamp((folder / "metadata.json").stat().st_mtime).strftime("%m.%d %H:%M"),
            "updated_at": updated, "active_candidate_id": active,
            "path": str(folder), "events": events,
            "delivery": None, "deliveries": [], "latest_artifact_id": artifacts[0]["id"] if artifacts else None}
    for artifact in artifacts:
        public = {**{k: v for k, v in artifact.items() if k != "file_path"},
                  "media": register_media(artifact["file_path"]) if artifact["available"] else None}
        item["deliveries"].append(public)
        if delivery is not None and artifact["id"] == delivery["id"]:
            item["delivery"] = public
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
    JOB_MANAGER.update(key, **values)


def job_event(key, event):
    """Stream bounded, factual action summaries; no model reasoning or credentials."""
    with LOCK:
        job = JOBS[key]
        events = list(job.get("events", []))
        safe = {k: v for k, v in event.items() if k in {
            "stage_id", "kind", "message", "tool", "candidate_id", "candidate_index", "candidate_total", "verdict"}}
        safe.update(seq=job.get("event_seq", 0) + 1, at=time.time())
        events.append(safe)
        job.update(events=events[-120:], event_seq=safe["seq"], stage_id=safe.get("stage_id", job.get("stage_id")))
        if safe.get("message"):
            job["stage"] = safe["message"]
        if safe.get("candidate_id"):
            job["candidate_progress"] = {"id": safe["candidate_id"], "current": safe.get("candidate_index"), "total": safe.get("candidate_total")}
        folder = job.get("run_path")
        if folder:
            save_json(Path(folder) / "agent_events.json", job["events"])
        JOB_MANAGER.update(key)


def terminal_job(record):
    folder = Path(record["run_path"]) if record.get("run_path") else None
    status = record["status"]
    if folder and status != "complete":
        finish_operation(folder, status=status)
        for name in ("retrieval_state.json", "local_search.json"):
            state = read_json(folder / name)
            if state and state.get("status") == "running":
                save_json(folder / name, {**state, "status": status})
        messages = {"cancelled": "已停止；已发送的 API 请求可能仍计费", "interrupted": "服务重启，任务中断，未自动重跑",
                    "failed": "本次任务未完成，已有文件和记录已保留"}
        job_event(record["id"], {"kind": "attention", "message": messages.get(status, "任务未完成")})
    restore_job(record)


def restore_job(record):
    """Rebind references after restart without replaying actions or duplicating logs."""
    signature = record.get("source_signature")
    if signature:
        try:
            if source_signature(signature["path"]) != signature:
                record["source_unavailable"] = True
                return
            register_media(signature["path"])
            record.pop("source_unavailable", None)
        except (OSError, ValueError):
            record["source_unavailable"] = True
            return
    result = record.get("result", {})
    output = result.get("media")
    if output:
        try:
            restored = register_media(output["path"])
            if restored["id"] != output["id"]:
                raise ValueError("输出已改变")
            result["media"] = restored
        except (OSError, ValueError):
            record["output_unavailable"] = True
            result.pop("media", None)
            result["reviewed"] = False
            result["message"] = "历史成片已改变或不可用，任务记录已保留"
    folder = Path(record["run_path"]) if record.get("run_path") else None
    source = record.get("root_run_path") or (str(folder) if folder and (folder / "metadata.json").is_file() else None)
    if source:
        run = load_run(Path(source))
        if run:
            record["run"] = run
            if record.get("result") is not None:
                record["result"]["run"] = run
                output = record["result"].get("media")
                artifact = next((item for item in run.get("deliveries", []) if output and item.get("media", {}) and item["media"]["id"] == output["id"]), None)
                if artifact is not None:
                    record["result"]["reviewed"] = artifact["reviewed"]


JOB_MANAGER = StudioJobs(DATA / "jobs", JOBS, LOCK, BUSY, on_terminal=terminal_job, on_restore=restore_job)


def spawn_job(kind, body, *, retry_of=None, frozen_ranges=None):
    fields = {"retrieval": {"media_id", "query", "route", "visual_query_en", "events", "cloud_planning"},
              "auto_edit": {"media_id", "query"},
              "review": {"run_id", "candidate_id", "start", "end"},
              "export": {"run_id", "candidate_id", "start", "end"}}
    if kind not in fields:
        raise ValueError("不支持的任务类型")
    request = {name: body[name] for name in fields[kind] if name in body}
    requires_cloud = kind in {"auto_edit", "review"} or bool(request.get("cloud_planning"))
    if requires_cloud and body.get("allow_cloud") is not True:
        raise ValueError("请明确授权本次云端解析或画面复核")
    if kind in {"retrieval", "auto_edit"}:
        media = MEDIA[request["media_id"]]
        if not 0 < len(str(request.get("query", "")).strip()) <= 2000:
            raise ValueError("请填写 1–2000 字的剪辑需求")
        scope = {"media_name": media["name"], "query": str(request["query"]).strip()}
    else:
        run = RUNS[request["run_id"]]
        media = MEDIA[run["media_id"]]
        metadata = VideoReader().read_metadata(media["path"])
        candidates = operation_candidates(run, {**request, "_frozen_ranges": frozen_ranges}, metadata)
        if kind == "export" and not request.get("candidate_id"):
            raise ValueError("请选择要导出的片段")
        scope = {"media_name": media["name"], "query": run["request"],
                 "ranges": [{"candidate_id": c.candidate_id, "start": c.start_seconds, "end": c.end_seconds} for c in candidates]}
    actual = {**request, "allow_cloud": body.get("allow_cloud") is True}
    if kind in {"review", "export"}:
        actual["_frozen_ranges"] = scope["ranges"]

    async def operation(key):
        if kind == "export":
            return await run_sync(export_trim, key, actual)
        return await {"retrieval": retrieve, "auto_edit": auto_edit, "review": review}[kind](key, actual)

    return JOB_MANAGER.submit(kind, operation, request, source_signature=source_signature(media["path"]),
                              scope=scope, requires_cloud=requires_cloud, retry_of=retry_of)


def retry_job(key, body):
    with LOCK:
        record = dict(JOBS[key])
    if record.get("status") not in RETRYABLE or not record.get("input"):
        raise ValueError("该任务当前不能重试")
    signature = record.get("source_signature")
    if not signature or source_signature(signature["path"]) != signature:
        raise ValueError("原素材已改变或不可用，请重新导入与检索")
    register_media(signature["path"])
    request = {**record["input"], "allow_cloud": body.get("allow_cloud") is True}
    if record.get("root_run_path"):
        load_run(Path(record["root_run_path"]))
    return spawn_job(record["kind"], request, retry_of=key, frozen_ranges=record.get("scope", {}).get("ranges"))


async def retrieve(key, body):
    media = MEDIA[body["media_id"]]
    meta = await run_sync(VideoReader().read_metadata, media["path"])
    request = str(body.get("query", "")).strip()
    if not request or len(request) > 2000:
        raise ValueError("请填写 1–2000 字的检索需求")
    if body.get("cloud_planning"):
        job_event(key, {"stage_id": "planning", "message": "解析剪辑目标与声画条件，选择检索路径"})
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
    update_job(key, run_path=str(folder), root_run_path=str(folder), media_id=media["id"], request=request)
    job_event(key, {"stage_id": "retrieval", "message": {"visual": "调用本地视觉定位器，检索画面候选", "audio_visual": "调用本地声画网络，定位有声事件", "hybrid": "联合视觉定位与声画网络，检索候选"}[route]})
    candidates, report = await NeuralVideoRetriever(top_k=12).retrieve(
        task, meta, on_progress=lambda event: job_event(key, event))
    save_json(folder / "candidates.json", [c.model_dump(mode="json") for c in candidates])
    save_json(folder / "retrieval.json", report)
    save_json(folder / "retrieval_state.json", {"status": "complete", "route": route, "candidate_count": len(candidates)})
    run = load_run(folder)
    update_job(key, run=run)
    stage_id = "refinement" if report.get("details", {}).get("boundary_refinement", {}).get("status") == "ok" else "retrieval"
    job_event(key, {"stage_id": stage_id, "message": f"已取得 {len(candidates)} 个候选片段", "kind": "result"})
    return {"run": run, "candidate_count": len(candidates), "message": f"已找到 {len(candidates)} 个候选片段，可预览或进一步审核"}


async def auto_edit(key, body):
    # Consent is checked again here so direct callers cannot trigger cloud work accidentally.
    if body.get("allow_cloud") is not True:
        raise ValueError("自动剪辑需要授权需求解析与画面复核")
    update_job(key, media_id=body["media_id"], request=str(body.get("query", "")).strip())
    found = await retrieve(key, {**body, "cloud_planning": True})
    run = found["run"]
    if not run["candidates"]:
        save_json(Path(run["path"]) / "local_search.json", {"status": "no_match", "stop_after_first_pass": True, "attempts": []})
        job_event(key, {"stage_id": "done", "kind": "no_match", "message": "未找到可供复核的候选，未生成成片"})
        return {**found, "run": load_run(Path(run["path"])), "outcome": "no_match", "reviewed": False,
                "message": "没有找到候选片段，可修改需求后重试"}
    return await review(key, {"run_id": run["id"], "allow_cloud": True}, persist_in_run=True)


def operation_candidates(run, body, metadata):
    selected = body.get("candidate_id")
    if ("start" in body or "end" in body) and not selected:
        raise ValueError("调整范围时需要指定候选片段")
    if ("start" in body or "end" in body) and (body.get("start") is None or body.get("end") is None):
        raise ValueError("请填写有效的入点和出点")
    frozen = body.get("_frozen_ranges")
    ranges = {r["candidate_id"]: r for r in frozen} if frozen is not None else None
    candidates = [scoped_candidate(c, metadata,
                    ranges[c["candidate_id"]]["start"] if ranges is not None else body.get("start"),
                    ranges[c["candidate_id"]]["end"] if ranges is not None else body.get("end"))
                  for c in run["candidates"] if (not selected or c["candidate_id"] == selected)
                  and (ranges is None or c["candidate_id"] in ranges)]
    if not candidates:
        raise ValueError("找不到待处理的候选片段")
    signature = read_json(Path(run["path"]) / "retrieval.json", {}).get("source_signature")
    if signature and signature != source_signature(metadata.file_path):
        raise ValueError("源视频已改变，请重新检索后再剪辑")
    return candidates


async def review(key, body, *, persist_in_run=False):
    if body.get("allow_cloud") is not True:
        raise ValueError("Agent 复核会调用已配置的模型 API，需要先明确勾选授权")
    run = RUNS[body["run_id"]]
    source = Path(run["path"])
    meta = VideoReader().read_metadata(MEDIA[run["media_id"]]["path"])
    task = VideoTaskSpec.model_validate(run["task"])
    candidates = operation_candidates(run, body, meta)
    if body.get("candidate_id") and "start" in body:
        save_edit(source, candidates[0])
    folder = source if persist_in_run else begin_operation(source, "review", key, candidates, meta.file_path)
    update_job(key, run_path=str(folder), root_run_path=str(source), run=load_run(source), run_id=run["id"],
               media_id=run["media_id"], request=task.query)
    expected_source = source_signature(meta.file_path)
    try:
        result = await deliver_first_match(task, meta, candidates, folder, use_legacy_action_verifier=False,
                                           allow_boundary_repair=persist_in_run,
                                           on_progress=lambda event: job_event(key, event))
        if source_signature(meta.file_path) != expected_source:
            raise ValueError("源视频在复核过程中发生变化，请重新检索")
        finish_operation(folder)
    except BaseException:
        finish_operation(folder, status="cancelled" if JOBS[key].get("status") == "cancelling" else "failed")
        raise
    state = read_json(folder / "local_search.json", {})
    if result and result.valid and state.get("status") == "satisfied":
        clip = register_media(result.clips[0].file_path)
        delivered = result.clips[0]
        return {"media": clip, "reviewed": True, "outcome": "delivered", "candidate_id": delivered.candidate_id,
                "start": delivered.source_start_seconds, "end": delivered.source_end_seconds,
                "run": load_run(source),
                "message": "Agent 已交付一段通过双重复核的成片", "path": str(folder)}
    status = state.get("status", "no_match")
    message = "复核受阻，等待人工确认" if status == "review_blocked" else "候选未通过全部检查，未交付成片"
    job_event(key, {"stage_id": "done", "kind": "attention" if status != "no_match" else "no_match", "message": message})
    return {"reviewed": False, "outcome": "needs_attention" if status != "no_match" else "no_match",
            "run": load_run(source), "message": message, "path": str(folder)}


def export_trim(key, body):
    checkpoint()
    run = RUNS[body["run_id"]]
    meta = VideoReader().read_metadata(MEDIA[run["media_id"]]["path"])
    if not body.get("candidate_id"):
        raise ValueError("请选择要导出的片段")
    clip = operation_candidates(run, body, meta)[0]
    task = VideoTaskSpec.model_validate(run["task"])
    selection = ClipSelection(choices=[ClipChoice(candidate_id=clip.candidate_id, reason="用户在时间线中手动确认裁剪范围；未进行语义复核")], summary="手动导出")
    checked = ClipSelectionValidator().validate(task, meta, [clip], selection)
    if not checked.valid:
        raise ValueError("当前范围不能导出：" + "；".join(checked.errors))
    source = Path(run["path"])
    save_edit(source, clip)
    folder = begin_operation(source, "export", key, [clip], meta.file_path)
    update_job(key, stage="正在导出当前范围", stage_id="export", run_path=str(folder), root_run_path=str(source),
               run_id=run["id"], media_id=run["media_id"], request=task.query)
    job_event(key, {"stage_id": "export", "message": "按当前入点和出点导出；不进行内容复核"})
    expected_source = source_signature(meta.file_path)
    try:
        report = VideoClipExporter().export(task, meta, checked, folder / "clips")
        checkpoint()
        save_json(folder / "exports.json", report.model_dump(mode="json"))
        if not report.valid or not report.clips:
            raise ValueError("导出校验未通过：" + "；".join(report.errors))
        if any(file_version_status(clip.file_path, clip.file_signature) != "current" for clip in report.clips):
            raise ValueError("导出文件版本已改变，不能标记为本次导出")
        if source_signature(meta.file_path) != expected_source:
            raise ValueError("源视频在导出过程中发生变化，请重新检索")
        finish_operation(folder)
    except BaseException:
        finish_operation(folder, status="cancelled" if JOBS[key].get("status") == "cancelling" else "failed")
        raise
    job_event(key, {"stage_id": "done", "kind": "result", "message": "手动导出完成，未经内容复核"})
    return {"media": register_media(report.clips[0].file_path), "reviewed": False, "candidate_id": clip.candidate_id,
            "start": report.clips[0].source_start_seconds, "end": report.clips[0].source_end_seconds,
            "run": load_run(source),
            "message": "已按当前范围导出，未经内容审核"}


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
                    runs = sorted(RUNS.values(), key=lambda run: run.get("updated_at", 0), reverse=True)
                    media = list(MEDIA.values())
                    active_job = next((JOB_MANAGER.public(j["id"]) for j in JOBS.values() if j["status"] in ACTIVE), None)
                    latest_job = JOB_MANAGER.public(max(JOBS.values(), key=lambda j: j.get("started", 0))["id"]) if JOBS else None
                return self.send_json({"token": TOKEN, "media": media, "runs": runs,
                    "instance_id": INSTANCE_ID,
                    "events": [{"value": k, "label": v} for k, v in EVENT_NAMES.items()],
                    "active_job": active_job, "latest_job": latest_job})
            if len(parts) == 3 and parts[:2] == ["api", "jobs"]:
                with LOCK:
                    return self.send_json({**JOB_MANAGER.public(parts[2]), "instance_id": INSTANCE_ID})
            if len(parts) == 3 and parts[:2] == ["api", "runs"]:
                run = load_run(Path(RUNS[parts[2]]["path"]))
                if run is None:
                    return self.send_json({"error": "原素材不可用，请检查本地文件"}, 404)
                return self.send_json(run)
            if len(parts) == 3 and parts[:2] == ["api", "media"]:
                return self.send_file(Path(registered_media(parts[2])["path"]))
            if len(parts) == 3 and parts[:2] == ["api", "thumb"]:
                media = registered_media(parts[2])
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
                media = registered_media(parts[2])
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
            if not isinstance(body, dict):
                raise ValueError("请求内容必须为对象")
            parts = urlparse(self.path).path.strip("/").split("/")
            if len(parts) == 4 and parts[:2] == ["api", "jobs"]:
                if parts[3] == "cancel":
                    return self.send_json(JOB_MANAGER.cancel(parts[2]))
                if parts[3] == "retry":
                    return self.send_json(retry_job(parts[2], body), 202)
            if self.path == "/api/import/path":
                return self.send_json({"media": register_media(str(body.get("path", "")).strip().strip('"'))})
            if self.path == "/api/retrieve":
                if body.get("media_id") not in MEDIA:
                    raise ValueError("请先导入素材")
                return self.send_json(spawn_job("retrieval", body), 202)
            if self.path == "/api/auto-edit":
                if body.get("media_id") not in MEDIA:
                    raise ValueError("请先导入素材")
                if body.get("allow_cloud") is not True:
                    raise ValueError("请先授权本次需求解析与云端画面复核")
                if not 0 < len(str(body.get("query", "")).strip()) <= 2000:
                    raise ValueError("请填写 1–2000 字的剪辑需求")
                return self.send_json(spawn_job("auto_edit", body), 202)
            if self.path in {"/api/review", "/api/export"}:
                if body.get("run_id") not in RUNS:
                    raise ValueError("请先选择有效的检索结果")
                if self.path == "/api/review" and body.get("allow_cloud") is not True:
                    raise ValueError("请先授权当前范围的云端复核")
                run = RUNS[body["run_id"]]
                meta = VideoReader().read_metadata(MEDIA[run["media_id"]]["path"])
                operation_candidates(run, body, meta)
                if self.path == "/api/export" and not body.get("candidate_id"):
                    raise ValueError("请选择要导出的片段")
                return self.send_json(spawn_job("review" if self.path.endswith("review") else "export", body), 202)
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
    JOB_MANAGER.restore()
    server = ThreadingHTTPServer(("127.0.0.1", 7861), Handler)
    server.daemon_threads = True
    print("cutliner: http://127.0.0.1:7861", flush=True)
    print("Local media stays on this computer. Cloud planning/review requires explicit UI opt-in.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        JOB_MANAGER.close()
