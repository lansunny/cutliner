import json
from collections.abc import Callable

from mini_agent.video.candidate_builder import ClipCandidateBuilder
from mini_agent.video.clip_validator import ClipSelectionValidator
from mini_agent.video.schemas import (
    ClipValidationReport,
    VideoFrameEvidence,
    VideoMetadata,
    VideoTaskSpec,
)
from mini_agent.video.task_planner import VideoTaskPlanner


class VideoClipSelector:
    def __init__(self, planner: VideoTaskPlanner):
        self.planner = planner
        self.builder = ClipCandidateBuilder()
        self.validator = ClipSelectionValidator()

    async def select(
        self,
        task: VideoTaskSpec,
        metadata: VideoMetadata,
        evidence: list[VideoFrameEvidence],
        save_result: Callable[[str, object], None],
    ) -> ClipValidationReport:
        candidates = self.builder.build(task, metadata, evidence)
        save_result("candidates.json", [item.model_dump(mode="json") for item in candidates])
        print(f"\n生成{len(candidates)}个候选时间窗口，正在选择片段……")

        feedback = None
        for attempt in range(1, 3):
            selection = await self.planner.select_clips(task, candidates, feedback)
            report = self.validator.validate(task, metadata, candidates, selection)
            save_result(
                f"selection_attempt_{attempt:02d}.json",
                {
                    "selection": selection.model_dump(mode="json"),
                    "validation": report.model_dump(mode="json"),
                },
            )
            if report.valid:
                break
            print("片段校验未通过：", "；".join(report.errors))
            if attempt == 1:
                print("将具体错误反馈给DeepSeek，修正一次……")
                feedback = json.dumps(
                    {
                        "previous_selection": selection.model_dump(mode="json"),
                        "errors": report.errors,
                    },
                    ensure_ascii=False,
                )

        save_result("highlights.json", report.model_dump(mode="json"))
        return report
