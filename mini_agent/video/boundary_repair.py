"""Validate an Agent's observed-frame proposal; never certify its semantics."""

from mini_agent.video.schemas import AgentBoundaryRepair, ClipCandidate


CONTEXT_SECONDS = 6.
MAX_REPAIRS_PER_TASK = 2


def repair_context(candidate, metadata):
    return (max(0., candidate.start_seconds - CONTEXT_SECONDS),
            min(metadata.duration_seconds, candidate.end_seconds + CONTEXT_SECONDS))


def apply_boundary_proposal(candidate, proposal, metadata):
    if isinstance(proposal, AgentBoundaryRepair):
        proposal = proposal.model_dump()
    proposal = AgentBoundaryRepair.model_validate(proposal)
    lower, upper = repair_context(candidate, metadata)
    if (abs(proposal.original_start_seconds - candidate.start_seconds) > 1e-5
            or abs(proposal.original_end_seconds - candidate.end_seconds) > 1e-5):
        raise ValueError("边界建议不属于当前候选版本")
    if abs(lower - proposal.context_start_seconds) > 1e-5 or abs(upper - proposal.context_end_seconds) > 1e-5:
        raise ValueError("边界建议超出本轮允许的上下文")
    frames = {frame.frame_index: frame.source_timestamp_seconds for frame in proposal.observed_frames}
    if (len(frames) != len(proposal.observed_frames) or any(index >= metadata.frame_count for index in frames)
            or any(not lower <= t < upper for t in frames.values())):
        raise ValueError("边界依据包含重复或越界的画面")
    if proposal.start_frame_index not in frames or proposal.end_frame_index not in frames:
        raise ValueError("边界没有实际观察帧作为依据")
    start, end = proposal.proposed_start_seconds, proposal.proposed_end_seconds
    if (abs(start - frames[proposal.start_frame_index]) > 1e-5
            or abs(end - min(upper, frames[proposal.end_frame_index] + 1 / metadata.fps)) > 1e-5):
        raise ValueError("边界时间与实际观察帧不一致")
    if not lower <= start < end <= upper or end - start < 1 / metadata.fps:
        raise ValueError("修正区间无效")
    if max(start, candidate.start_seconds) >= min(end, candidate.end_seconds):
        raise ValueError("局部修正不能跳到与当前候选完全不重叠的事件")
    if abs(start - candidate.start_seconds) + abs(end - candidate.end_seconds) < 1 / metadata.fps:
        raise ValueError("边界没有发生有效变化")
    # Preserve the neural evidence as it was; this is an Agent proposal, not a new
    # trained-network prediction or a semantic pass inherited from the old range.
    return ClipCandidate.model_validate({**candidate.model_dump(), "start_seconds": start, "end_seconds": end,
        "shot_start_seconds": None, "shot_end_seconds": None, "agent_boundary_repair": proposal,
        "boundary_note": "Agent依据实际观察帧提出的局部修正；原模型证据保留，新区间必须重新复核"})
