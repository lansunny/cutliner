from mini_agent.video.schemas import (
    ClipCandidate,
    ClipSelection,
    ClipValidationReport,
    SelectedClip,
    VideoMetadata,
    VideoTaskSpec,
)


class ClipSelectionValidator:
    def validate(
        self,
        task: VideoTaskSpec,
        metadata: VideoMetadata,
        candidates: list[ClipCandidate],
        selection: ClipSelection,
    ) -> ClipValidationReport:
        tolerance = 1e-6
        candidate_map = {item.candidate_id: item for item in candidates}
        if len(candidate_map) != len(candidates):
            raise ValueError("候选片段ID重复")

        errors: list[str] = []
        selected: list[SelectedClip] = []
        seen: set[str] = set()
        if task.max_clips is not None and len(selection.choices) > task.max_clips:
            errors.append(f"选择了{len(selection.choices)}段，超过最多{task.max_clips}段的要求")

        for choice in selection.choices:
            if choice.candidate_id in seen:
                errors.append(f"重复选择候选：{choice.candidate_id}")
                continue
            seen.add(choice.candidate_id)
            candidate = candidate_map.get(choice.candidate_id)
            if candidate is None:
                errors.append(f"候选ID不存在：{choice.candidate_id}")
                continue
            if candidate.end_seconds <= candidate.start_seconds:
                errors.append(f"{choice.candidate_id}的结束时间必须大于开始时间")
                continue
            if candidate.end_seconds > metadata.duration_seconds + tolerance:
                errors.append(f"{choice.candidate_id}超出源视频时长")
            if candidate.shot_start_seconds is not None and candidate.start_seconds < candidate.shot_start_seconds - tolerance:
                errors.append(f"{choice.candidate_id}开始时间超出镜头范围")
            if candidate.shot_end_seconds is not None and candidate.end_seconds > candidate.shot_end_seconds + tolerance:
                errors.append(f"{choice.candidate_id}结束时间超出镜头范围")
            selected.append(
                SelectedClip(
                    **candidate.model_dump(),
                    selection_reason=choice.reason,
                )
            )

        selected.sort(key=lambda item: item.start_seconds)
        for previous, current in zip(selected, selected[1:]):
            if current.start_seconds < previous.end_seconds - tolerance:
                errors.append(f"{previous.candidate_id}与{current.candidate_id}时间范围重叠")
        total = sum(item.duration_seconds for item in selected)
        if task.max_total_duration_seconds is not None and total > task.max_total_duration_seconds + tolerance:
            errors.append(f"总时长{total:.3f}秒，超过上限{task.max_total_duration_seconds}秒")

        return ClipValidationReport(
            valid=not errors,
            errors=errors,
            total_duration_seconds=total,
            selected_clips=selected if not errors else [],
        )
