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
from mini_agent.video.job_control import checkpoint, run_sync
from mini_agent.video.boundary_repair import CONTEXT_SECONDS, MAX_REPAIRS_PER_TASK, apply_boundary_proposal
from mini_agent.video.studio_history import save_json, source_signature
from mini_agent.video.media_identity import file_version_status


async def deliver_first_match(task, metadata, candidates, run_dir: Path, *, local_verifier=None, allow_remote_review=True,
                              use_legacy_action_verifier=True, on_progress=None, allow_boundary_repair=False):
    context = {}
    phase = "review"
    repair_count = 0
    repair_enabled = bool(allow_boundary_repair and task.audio_condition_type == "none" and not task.audio_event_classes)

    def emit(message=None, *, stage_id=None, kind="action", **details):
        if on_progress is not None:
            on_progress({**context, "stage_id": stage_id or phase, "kind": kind,
                         "message": message, **details})

    def tool_progress(event):
        emit(**event)

    def save(name, value):
        if name == "local_search.json":
            value["boundary_repair_policy"] = {"enabled": repair_enabled, "performed": repair_count,
                "max_repairs_per_task": MAX_REPAIRS_PER_TASK, "max_repairs_per_candidate": 1,
                "context_seconds_each_side": CONTEXT_SECONDS}
        save_json(run_dir / name, value)

    ordered = sorted(candidates, key=lambda c: (c.review_priority is None, c.review_priority or 0,
                                                -c.relevance_score, c.start_seconds))
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
    source_version = source_signature(metadata.file_path)
    try:
        for index, candidate in enumerate(ordered, 1):
            checkpoint()
            if source_signature(metadata.file_path) != source_version:
                raise ValueError("源视频已改变，不能继续复用本轮候选与复核证据")
            phase = "review"
            context = {"candidate_id": candidate.candidate_id, "candidate_index": index, "candidate_total": len(ordered)}
            emit(f"检查候选 {index}/{len(ordered)}：{candidate.start_seconds:.2f}–{candidate.end_seconds:.2f}s")
            attempt_dir = run_dir / "local_search" / f"attempt_{index:03d}"
            attempt_dir.mkdir(parents=True, exist_ok=False)
            attempt = {"candidate_id": candidate.candidate_id,
                       "effective_start_seconds": candidate.start_seconds, "effective_end_seconds": candidate.end_seconds}
            if candidate.boundary_refinement is not None:
                attempt["boundary_refinement"] = candidate.boundary_refinement.model_dump(mode="json")
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
                allow_boundary_repair=repair_enabled and repair_count < MAX_REPAIRS_PER_TASK,
            )
            attempt["local_review"] = local.model_dump(mode="json")
            save("local_search.json", {"status": "running", "stop_after_first_pass": True, "attempts": attempts})
            if local.blocked_by_provider:
                emit("候选复核被服务商拦截，停止自动交付，等待人工确认", kind="attention")
                save("local_search.json", {"status": "review_blocked", "stop_after_first_pass": True, "attempts": attempts})
                print("局部补看被平台审核拦截，本次停止，待人工确认。")
                return None
            if local.boundary_proposal is not None:
                attempt["initial_review"] = local.model_dump(mode="json")
                if not repair_enabled or repair_count >= MAX_REPAIRS_PER_TASK:
                    attempt["repair_status"] = "not_authorized_or_budget_exhausted"
                    emit("本次未允许修改边界或修正预算已用尽，保留为待确认", kind="verdict")
                    continue
                try:
                    repaired = apply_boundary_proposal(candidate, local.boundary_proposal, metadata)
                except ValueError as error:
                    attempt.update(repair_status="invalid_proposal", repair_error=str(error))
                    emit("边界建议未通过观察范围校验，转向下一个候选", kind="verdict")
                    continue
                repair_count += 1
                candidate = repaired
                attempt.update(repair_status="rechecking", effective_candidate=candidate.model_dump(mode="json"),
                    effective_start_seconds=candidate.start_seconds, effective_end_seconds=candidate.end_seconds)
                phase = "boundary_repair"
                emit(f"依据已观察画面修正为 {candidate.start_seconds:.2f}–{candidate.end_seconds:.2f}s，重新复核实际区间",
                     tool="propose_boundary")
                save("local_search.json", {"status": "running", "stop_after_first_pass": True, "attempts": attempts})
                local = await agent.review_interval(task, metadata.file_path, candidate.candidate_id,
                    attempt_dir / "repaired_source_review", candidate.start_seconds, candidate.end_seconds,
                    allow_boundary_repair=False)
                attempt["local_review"] = local.model_dump(mode="json")
                attempt["repair_status"] = "source_pass" if local.verdict == "pass" else "not_passed"
                save("local_search.json", {"status": "running", "stop_after_first_pass": True, "attempts": attempts})
                if local.blocked_by_provider:
                    emit("修正后复核被服务商拦截，停止自动交付", kind="attention")
                    save("local_search.json", {"status": "review_blocked", "stop_after_first_pass": True, "attempts": attempts})
                    return None
                if local.boundary_proposal is not None:
                    attempt["repair_status"] = "second_proposal_refused"
                    continue
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
            if source_signature(metadata.file_path) != source_version:
                raise ValueError("源视频已改变，不能按旧证据导出")
            emit("按候选边界裁剪视频，保留原始音轨")
            exported = await run_sync(exporter.export, task, metadata, checked, attempt_dir / "clips")
            (attempt_dir / "exports.json").write_text(exported.model_dump_json(indent=2), encoding="utf-8")
            attempt["exports"] = exported.model_dump(mode="json")
            if not exported.valid:
                emit("导出文件未通过技术校验，转向下一个候选", kind="verdict")
                continue
            phase = "verify"
            if file_version_status(exported.clips[0].file_path, exported.clips[0].file_signature) != "current":
                save("local_search.json", {"status": "output_changed", "stop_after_first_pass": True, "attempts": attempts})
                emit("成片文件已改变，停止交付，需重新导出", kind="attention")
                return None
            emit("重新读取实际成片，进行交付前复核")
            final = await agent.review_clip(task, exported.clips[0], attempt_dir / "output_review",
                                            source_metadata=metadata, source_review=local)
            attempt["output_review"] = final.model_dump(mode="json")
            if file_version_status(exported.clips[0].file_path, exported.clips[0].file_signature) != "current":
                save("local_search.json", {"status": "output_changed", "stop_after_first_pass": True, "attempts": attempts})
                emit("成片在复核期间发生变化，本次不作为有效交付", kind="attention")
                return None
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
            if source_signature(metadata.file_path) != source_version:
                raise ValueError("源视频已改变，本轮结果不能作为有效交付")
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
    except asyncio.CancelledError:
        save("local_search.json", {"status": "cancelled", "stop_after_first_pass": True, "attempts": attempts})
        raise
    finally:
        try:
            if agent is not None:
                await agent.close()
        finally:
            if local_verifier is not None:
                await local_verifier.close()
