import math

from mini_agent.video.schemas import (
    ClipCandidate,
    VideoFrameEvidence,
    VideoMetadata,
    VideoTaskSpec,
)


class ClipCandidateBuilder:
    def build(
        self,
        task: VideoTaskSpec,
        metadata: VideoMetadata,
        evidence: list[VideoFrameEvidence],
    ) -> list[ClipCandidate]:
        duration = metadata.duration_seconds
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("源视频时长必须大于0")

        ordered = sorted(evidence, key=lambda item: item.frame.timestamp_seconds)
        frame_ids = [item.frame.frame_index for item in ordered]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("视觉证据中存在重复帧编号")
        for item in ordered:
            frame = item.frame
            if frame.frame_index != item.observation.frame_index:
                raise ValueError("视觉观察与源帧编号不一致")
            if not math.isfinite(frame.timestamp_seconds) or not 0 <= frame.timestamp_seconds < duration:
                raise ValueError("视觉证据的时间戳不在源视频范围内")
            if frame.frame_index >= metadata.frame_count:
                raise ValueError("视觉证据的帧编号超出源视频")
            if frame.shot_id is not None:
                if frame.shot_start_seconds is None or frame.shot_end_seconds is None:
                    raise ValueError("代表帧缺少镜头边界")
                if not 0 <= frame.shot_start_seconds < frame.shot_end_seconds <= duration + 1e-6:
                    raise ValueError("镜头边界不在源视频范围内")
                if not frame.shot_start_seconds <= frame.timestamp_seconds < frame.shot_end_seconds:
                    raise ValueError("代表帧时间戳不在其镜头范围内")

        # Nearby matches in the same shot form one event, not duplicate clips.
        groups: list[list[VideoFrameEvidence]] = []
        for item in ordered:
            if not item.observation.matches_query:
                continue
            if (groups and item.frame.shot_id == groups[-1][-1].frame.shot_id
                    and item.frame.timestamp_seconds - groups[-1][-1].frame.timestamp_seconds <= 3.0):
                groups[-1].append(item)
            else:
                groups.append([item])

        candidates: list[ClipCandidate] = []
        seen_windows: set[tuple[str | None, float, float]] = set()
        for group in groups:
            anchor = max(group, key=lambda item: item.observation.relevance_score)
            lower = anchor.frame.shot_start_seconds if anchor.frame.shot_start_seconds is not None else 0.0
            upper = anchor.frame.shot_end_seconds if anchor.frame.shot_end_seconds is not None else duration
            first = group[0].frame.timestamp_seconds
            last = group[-1].frame.timestamp_seconds
            before, after = 1.5, 2.5
            if task.clip_duration_seconds is not None:
                context = max(0.0, task.clip_duration_seconds - (last - first))
                before, after = context * 0.4, max(1.0 / metadata.fps, context * 0.6)
            # Preserve short complete shots; otherwise keep context around the evidence.
            short_shot_limit = task.clip_duration_seconds or 8.0
            if anchor.frame.shot_id is not None and upper - lower <= short_shot_limit:
                start, end = lower, upper
            else:
                start, end = max(lower, first - before), min(upper, last + after)
                if start - lower <= 0.75:
                    start = lower
                if upper - end <= 0.75:
                    end = upper
            if task.max_total_duration_seconds is not None and end - start > task.max_total_duration_seconds:
                length = min(task.max_total_duration_seconds, upper - lower)
                start = min(max(lower, anchor.frame.timestamp_seconds - length / 2), upper - length)
                end = start + length
            window_key = (anchor.frame.shot_id, round(start, 6), round(end, 6))
            if window_key in seen_windows:
                continue
            seen_windows.add(window_key)
            support = [
                item for item in ordered
                if start - 1e-6 <= item.frame.timestamp_seconds < end
                and (anchor.frame.shot_id is None or item.frame.shot_id == anchor.frame.shot_id)
            ]
            if not any(item.frame.frame_index == anchor.frame.frame_index for item in support):
                raise ValueError("候选窗口未包含对应的匹配帧")
            sharpness = [
                item.frame.sharpness_score for item in support
                if item.frame.sharpness_score is not None
            ]
            candidates.append(
                ClipCandidate(
                    candidate_id=f"clip_{len(candidates) + 1:04d}",
                    start_seconds=start,
                    end_seconds=end,
                    shot_id=anchor.frame.shot_id,
                    shot_start_seconds=anchor.frame.shot_start_seconds,
                    shot_end_seconds=anchor.frame.shot_end_seconds,
                    anchor_frame_index=anchor.frame.frame_index,
                    evidence_frame_indices=[item.frame.frame_index for item in support],
                    observations=[item.observation for item in support],
                    relevance_score=sum(item.observation.relevance_score for item in support) / len(support),
                    sharpness_score=sum(sharpness) / len(sharpness) if sharpness else None,
                    boundary_note=(
                        "按匹配帧簇及前后上下文自动裁剪，短镜头保留；整段语义由输出复核Agent检查"
                        if anchor.frame.shot_id is not None
                        else "时间范围由匹配采样帧附近的固定窗口估计，连续内容尚未验证"
                    ),
                )
            )
        return candidates
