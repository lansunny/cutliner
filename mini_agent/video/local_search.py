"""Inspect candidates locally and stop after the first verified exported result."""

import asyncio
import json
from pathlib import Path

from mini_agent.config import Settings
from mini_agent.video.action_verifier import LocalActionVerifier
from mini_agent.video.clip_exporter import VideoClipExporter
from mini_agent.video.clip_validator import ClipSelectionValidator
from mini_agent.video.review_agent import VideoReviewAgent
from mini_agent.video.neural_retriever import audio_evidence_for_candidate
from mini_agent.video.schemas import ClipChoice, ClipSelection


async def deliver_first_match(task, metadata, candidates, run_dir: Path, *, local_verifier=None, allow_remote_review=True,
                              use_legacy_action_verifier=True, on_progress=None):
    context = {}
    phase = "review"

    def emit(message=None, *, stage_id=None, kind="action", **details):
        if on_progress is not None:
            on_progress({**context, "stage_id": stage_id or phase, "kind": kind,
                         "message": message, **details})

    def tool_progress(event):
        emit(**event)

    def save(name, value):
        (run_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    ordered = sorted(candidates, key=lambda c: (-c.relevance_score, c.start_seconds))
    save("local_search.json", {"status": "running", "stop_after_first_pass": True, "attempts": []})
    if not ordered:
        save("local_search.json", {"status": "no_match", "stop_after_first_pass": True, "attempts": []})
        print("没有可供局部补看的候选，未生成视频。")
        return None
    # Optional model tool precedes expensive API review; disabled by default.
    if use_legacy_action_verifier:
        local_verifier = local_verifier or LocalActionVerifier.from_env()
    agent = None
    exporter = VideoClipExporter()
    validator = ClipSelectionValidator()
    attempts = []
    deferred = []
    try:
        for index, candidate in enumerate(ordered, 1):
            phase = "review"
            context = {"candidate_id": candidate.candidate_id, "candidate_index": index, "candidate_total": len(ordered)}
            emit(f"检查候选 {index}/{len(ordered)}：{candidate.start_seconds:.2f}–{candidate.end_seconds:.2f}s")
            attempt_dir = run_dir / "local_search" / f"attempt_{index:03d}"
            attempt_dir.mkdir(parents=True, exist_ok=False)
            attempt = {"candidate_id": candidate.candidate_id}
            attempts.append(attempt)
            if local_verifier is not None:
                result = await local_verifier.inspect(task.query, metadata.file_path,
                                                      candidate.start_seconds, candidate.end_seconds)
                attempt["local_action_tool"] = result
                (attempt_dir / "action_tool.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                save("local_search.json", {"status": "running", "stop_after_first_pass": True, "attempts": attempts})
                if result["decision"] == "reject":
                    attempt["controller_action"] = "skip_candidate_before_api_and_export"
                    print(f"[动作核验工具] {candidate.candidate_id} 动作不符，跳过API复核和视频导出。")
                    continue
            if not allow_remote_review:
                attempt["controller_action"] = "defer_to_semantic_review"
                deferred.append(candidate.candidate_id)
                continue
            if agent is None:
                agent = VideoReviewAgent(Settings.from_env(), on_progress=tool_progress)
            print(f"\n[局部补看] {candidate.candidate_id}：{candidate.start_seconds:.2f}～{candidate.end_seconds:.2f}秒")
            local = await agent.review_interval(
                task, metadata.file_path, candidate.candidate_id, attempt_dir / "source_review",
                candidate.start_seconds, candidate.end_seconds,
                initial_av_evidence=audio_evidence_for_candidate(candidate, metadata.file_path),
            )
            attempt["local_review"] = local.model_dump(mode="json")
            save("local_search.json", {"status": "running", "stop_after_first_pass": True, "attempts": attempts})
            if local.blocked_by_provider:
                emit("候选复核被服务商拦截，停止自动交付，等待人工确认", kind="attention")
                save("local_search.json", {"status": "review_blocked", "stop_after_first_pass": True, "attempts": attempts})
                print("局部补看被平台审核拦截，本次停止，待人工确认。")
                return None
            if local.verdict != "pass":
                emit("候选未通过" if local.verdict == "reject" else "候选证据不足", kind="verdict",
                     verdict=local.verdict)
                print(f"[局部补看] {local.verdict}，不生成这段视频，检查下一个候选。")
                continue
            emit("源片段复核通过，准备裁剪", kind="verdict", verdict="pass")
            selection = ClipSelection(
                choices=[ClipChoice(candidate_id=candidate.candidate_id, reason=local.reason)],
                summary="局部补看通过，生成一段并复核，通过即停止",
            )
            checked = validator.validate(task, metadata, [candidate], selection)
            if not checked.valid:
                attempt["constraint_errors"] = checked.errors
                emit("候选不满足裁剪约束，转向下一个候选", kind="verdict")
                continue
            phase = "export"
            emit("按候选边界裁剪视频，保留原始音轨")
            exported = await asyncio.to_thread(exporter.export, task, metadata, checked, attempt_dir / "clips")
            (attempt_dir / "exports.json").write_text(exported.model_dump_json(indent=2), encoding="utf-8")
            attempt["exports"] = exported.model_dump(mode="json")
            if not exported.valid:
                emit("导出文件未通过技术校验，转向下一个候选", kind="verdict")
                continue
            phase = "verify"
            emit("重新读取实际成片，进行交付前复核")
            final = await agent.review_clip(task, exported.clips[0], attempt_dir / "output_review",
                                            source_metadata=metadata, source_review=local)
            attempt["output_review"] = final.model_dump(mode="json")
            if final.blocked_by_provider:
                emit("成片复核被服务商拦截；文件保留，但不标记为交付成功", kind="attention")
                save("local_search.json", {
                    "status": "review_blocked", "stop_after_first_pass": True, "attempts": attempts,
                    "pending_file": str(exported.clips[0].file_path),
                })
                print("视频已导出，但成片复核被平台拦截；本次停止，未标记为通过。")
                print("待人工确认的视频：", exported.clips[0].file_path)
                return None
            if final.verdict != "pass":
                emit("成片未通过最终复核，继续检查下一个候选", kind="verdict", verdict=final.verdict)
                print("[成片复核] 未通过，保留检查记录并继续下一个候选。")
                continue
            save("highlights.json", checked.model_dump(mode="json"))
            save("exports.json", exported.model_dump(mode="json"))
            save("local_search.json", {
                "status": "satisfied", "stop_after_first_pass": True, "attempts": attempts,
                "accepted_file": str(exported.clips[0].file_path),
                "skipped_candidates": [c.candidate_id for c in ordered[index:]],
            })
            print("[任务完成] 已找到并生成符合要求的片段，停止后续候选补看和视频生成。")
            emit("源片段与成片均已通过复核，停止检查其余候选", stage_id="done", kind="delivered")
            print("通过复核的视频：", exported.clips[0].file_path)
            return exported
        if deferred:
            save("local_search.json", {"status": "needs_semantic_review", "stop_after_first_pass": True,
                                       "attempts": attempts, "deferred_candidates": deferred})
            print("[本地核验结束] 部分候选仍需语义复核；未调用远程API，未导出视频。")
            return None
        save("local_search.json", {"status": "no_match", "stop_after_first_pass": True, "attempts": attempts})
        print("候选已检查完，没有片段通过全部检查；未将任何片段标记为交付成功。")
        return None
    finally:
        try:
            if agent is not None:
                await agent.close()
        finally:
            if local_verifier is not None:
                await local_verifier.close()
