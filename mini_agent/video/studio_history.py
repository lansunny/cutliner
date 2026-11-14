"""Interval-scoped edit/review history for the local editing workbench."""

import json
import hashlib
import math
from pathlib import Path
import time
from uuid import uuid4

from mini_agent.video.schemas import ClipCandidate
from mini_agent.video.media_identity import file_signature, file_version_status


def read_json(path, fallback=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return fallback


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def same_range(first, second):
    return len(first) == len(second) == 2 and all(
        math.isfinite(a) and math.isfinite(b) and abs(a - b) <= 1e-5 for a, b in zip(first, second))


def source_signature(path):
    return file_signature(path)


def verified_attempt(attempt, clip):
    local, final = attempt.get("local_review", {}), attempt.get("output_review", {})
    return (local.get("verdict") == final.get("verdict") == "pass"
            and not local.get("blocked_by_provider") and not final.get("blocked_by_provider")
            and attempt.get("exports", {}).get("valid") is True
            and any(c["candidate_id"] == clip["candidate_id"] and c["file_path"] == clip["file_path"]
                    and same_range([c["source_start_seconds"], c["source_end_seconds"]],
                                   [clip["source_start_seconds"], clip["source_end_seconds"]])
                    for c in attempt.get("exports", {}).get("clips", [])))


def scoped_candidate(raw, metadata, start=None, end=None):
    payload = {k: v for k, v in raw.items() if k in ClipCandidate.model_fields}
    if (start is None) != (end is None):
        raise ValueError("请同时提供入点和出点")
    if start is not None:
        if isinstance(start, bool) or isinstance(end, bool):
            raise ValueError("入点和出点必须是秒数")
        start, end = float(start), float(end)
        if not all(math.isfinite(x) for x in (start, end)) or not 0 <= start < end <= metadata.duration_seconds:
            raise ValueError("剪辑范围必须位于源视频内，且出点晚于入点")
        payload.update(start_seconds=start, end_seconds=end, shot_start_seconds=None, shot_end_seconds=None,
                       review_priority=None, boundary_note="用户当前选择的区间；模型建议边界仅作为历史证据保留")
    return ClipCandidate.model_validate(payload)


def save_edit(folder, candidate):
    edits = read_json(folder / "edits.json", {})
    edits[candidate.candidate_id] = {"start_seconds": candidate.start_seconds,
        "end_seconds": candidate.end_seconds, "updated_at": time.time()}
    save_json(folder / "edits.json", edits)


def begin_operation(folder, kind, key, candidates, video):
    target = folder / f"studio_{kind}_{uuid4().hex[:12]}"
    target.mkdir()
    save_json(target / "operation.json", {"id": target.name, "kind": kind, "job_id": key,
        "status": "running", "started": time.time(),
        "source_signature": source_signature(video),
        "ranges": [{"candidate_id": c.candidate_id, "start": c.start_seconds, "end": c.end_seconds} for c in candidates]})
    return target


def finish_operation(folder, *, status="complete", **details):
    operation = read_json(folder / "operation.json")
    if operation:
        save_json(folder / "operation.json", {**operation, "status": status, "finished": time.time(), **details})


def load_history(folder, raw, metadata):
    """A pass belongs to one source interval, not to a candidate ID forever."""
    edits = read_json(folder / "edits.json", {})
    candidates = []
    for item in raw:
        if not isinstance(item, dict) or not item.get("model_evidence"):
            continue
        edited = edits.get(item["candidate_id"], {})
        effective = scoped_candidate(item, metadata, edited.get("start_seconds"), edited.get("end_seconds"))
        candidates.append({**effective.model_dump(mode="json"), "model_start_seconds": item["start_seconds"],
            "model_end_seconds": item["end_seconds"], "review_verdict": "pending", "review_scope": None})
    sessions = [(folder, {"kind": "review", "started": (folder / "metadata.json").stat().st_mtime})]
    for pattern in ("studio_review_*", "studio_export_*"):
        for child in folder.glob(pattern):
            operation = read_json(child / "operation.json")
            if operation:
                sessions.append((child, operation))
            elif (child / "local_search.json").is_file():
                sessions.append((child, {"kind": "review", "started": child.stat().st_mtime}))
    sessions.sort(key=lambda item: item[1]["started"])
    latest_delivery, latest_state, latest_events = None, "not_reviewed", []
    artifacts = []
    current_source = source_signature(metadata.file_path)
    original_source = read_json(folder / "retrieval.json", {}).get("source_signature")
    updated = sessions[0][1]["started"]
    for directory, operation in sessions:
        state = read_json(directory / "local_search.json", {})
        exported = read_json(directory / "exports.json", {})
        events = read_json(directory / "agent_events.json", [])
        session_complete = operation.get("status", "complete") == "complete"
        source_current = operation.get("source_signature", original_source) == current_source
        latest_events = events
        updated = max(updated, operation.get("finished", operation["started"]))
        if operation["kind"] == "review" and state:
            latest_state = state.get("status", "not_reviewed")
        if operation.get("status") == "failed":
            latest_state = "operation_failed"
        elif operation.get("status") in {"cancelled", "interrupted"}:
            latest_state = operation["status"]
        for attempt in state.get("attempts", []):
            local, final = attempt.get("local_review", {}), attempt.get("output_review", {})
            clips = attempt.get("exports", {}).get("clips", [])
            start = attempt.get("effective_start_seconds", clips[0].get("source_start_seconds") if clips else None)
            end = attempt.get("effective_end_seconds", clips[0].get("source_end_seconds") if clips else None)
            if start is None or end is None:
                continue  # Legacy ID-only verdicts cannot verify an arbitrary edit.
            for candidate in candidates:
                successful_repair = (candidate["candidate_id"] == attempt["candidate_id"]
                    and session_complete and source_current and state.get("status") == "satisfied"
                    and attempt.get("effective_candidate", {}).get("agent_boundary_repair")
                    and edits.get(candidate["candidate_id"], {}).get("updated_at", 0) <= operation["started"]
                    and any(verified_attempt(attempt, c) and file_version_status(c["file_path"], c.get("file_signature")) == "current" for c in clips))
                if successful_repair:
                    repaired = scoped_candidate(attempt["effective_candidate"], metadata)
                    if repaired.candidate_id == candidate["candidate_id"] and same_range(
                        [repaired.start_seconds, repaired.end_seconds], [start, end]):
                        candidate.update(repaired.model_dump(mode="json"))
                if candidate["candidate_id"] != attempt["candidate_id"] or not same_range(
                    [candidate["start_seconds"], candidate["end_seconds"]], [start, end]):
                    continue
                passed = session_complete and source_current and state.get("status") == "satisfied" and any(
                    verified_attempt(attempt, c) and file_version_status(c["file_path"], c.get("file_signature")) == "current" for c in clips)
                verdict = final.get("verdict") or ("source_pass" if local.get("verdict") == "pass" else local.get("verdict", "pending"))
                candidate["review_verdict"] = "pass" if passed else "pending" if not source_current else (
                    "uncertain" if verdict == "pass" else verdict)
                candidate["review_scope"] = {"start": start, "end": end, "operation_id": directory.name}
        if session_complete and exported.get("valid") is True:
            for clip in exported.get("clips", []):
                was_reviewed = operation["kind"] == "review" and state.get("status") == "satisfied" and any(
                    a["candidate_id"] == clip["candidate_id"] and a.get("local_review", {}).get("verdict") == "pass"
                    and verified_attempt(a, clip)
                    and same_range([a.get("effective_start_seconds", clip["source_start_seconds"]),
                                    a.get("effective_end_seconds", clip["source_end_seconds"])],
                                   [clip["source_start_seconds"], clip["source_end_seconds"]])
                    for a in state.get("attempts", []))
                if operation["kind"] == "export" or was_reviewed:
                    file_status = file_version_status(clip["file_path"], clip.get("file_signature"))
                    matched = next((a for a in state.get("attempts", []) if verified_attempt(a, clip)), {})
                    revision = matched.get("effective_candidate", {}).get("agent_boundary_repair")
                    identity = f"{directory.name}|{clip['candidate_id']}|{Path(clip['file_path']).name}|{clip['source_start_seconds']}|{clip['source_end_seconds']}"
                    artifact = {"id": hashlib.sha256(identity.encode()).hexdigest()[:20], "version": len(artifacts) + 1,
                        "kind": "export" if operation["kind"] == "export" else "auto_edit" if directory == folder else "review",
                        "created_at": operation.get("finished", operation["started"]),
                        "available": file_status in {"current", "legacy"}, "file_status": file_status,
                        "source_current": source_current, "was_reviewed": was_reviewed,
                        "events": events,
                        "file_path": clip["file_path"], "reviewed": bool(was_reviewed and source_current and file_status == "current"),
                        "candidate_id": clip["candidate_id"], "start": clip["source_start_seconds"],
                        "end": clip["source_end_seconds"], "operation_id": directory.name,
                        "agent_boundary_repair": revision}
                    artifacts.append(artifact)
                    latest_delivery = artifact if artifact["available"] else None
                    if operation["kind"] == "export":
                        latest_state = "manual_exported"
    return candidates, latest_delivery, latest_state, latest_events, updated, list(reversed(artifacts))
