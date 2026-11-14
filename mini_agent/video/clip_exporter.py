import math
import subprocess
from pathlib import Path

import imageio_ffmpeg

from mini_agent.video.clip_validator import ClipSelectionValidator
from mini_agent.video.job_control import checkpoint, run_process
from mini_agent.video.media_identity import file_signature
from mini_agent.video.reader import VideoReader
from mini_agent.video.schemas import (
    ClipCandidate,
    ClipChoice,
    ClipSelection,
    ClipValidationReport,
    ExportedVideoClip,
    VideoExportReport,
    VideoMetadata,
    VideoTaskSpec,
)


class VideoClipExporter:
    def __init__(self):
        self.reader = VideoReader()
        self.validator = ClipSelectionValidator()

    @staticmethod
    def _container_duration(path: Path) -> float:
        reader = imageio_ffmpeg.read_frames(str(path))
        try:
            duration = float(next(reader)["duration"])
        finally:
            reader.close()
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"无法确认导出文件的播放时长: {path}")
        return duration

    def export(
        self,
        task: VideoTaskSpec,
        metadata: VideoMetadata,
        selection_report: ClipValidationReport,
        output_dir: Path,
    ) -> VideoExportReport:
        if not selection_report.valid or selection_report.errors:
            raise ValueError("片段选择未通过校验，不能执行导出")
        if not selection_report.selected_clips:
            return VideoExportReport(
                valid=True,
                total_video_duration_seconds=0.0,
                total_container_duration_seconds=0.0,
            )

        source = self.reader.read_metadata(metadata.file_path)
        if (
            source.frame_count != metadata.frame_count
            or source.width != metadata.width
            or source.height != metadata.height
            or abs(source.fps - metadata.fps) > 1e-3
        ):
            raise ValueError("源视频元数据已变化，请重新运行视频检索")
        candidates = [
            ClipCandidate.model_validate(clip.model_dump(exclude={"selection_reason"}))
            for clip in selection_report.selected_clips
        ]
        decision = ClipSelection(
            choices=[
                ClipChoice(candidate_id=clip.candidate_id, reason=clip.selection_reason)
                for clip in selection_report.selected_clips
            ],
            summary="导出前复核选中片段的硬约束",
        )
        checked = self.validator.validate(task, source, candidates, decision)
        if not checked.valid:
            raise ValueError("导出前校验未通过：" + "；".join(checked.errors))

        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=False)
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        exported: list[ExportedVideoClip] = []
        errors: list[str] = []

        for index, clip in enumerate(checked.selected_clips, 1):
            checkpoint()
            output_path = output_dir / f"clip_{index:02d}.mp4"
            print(f"正在导出第{index}段：{clip.start_seconds:.2f}～{clip.end_seconds:.2f}秒……")
            command = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
                "-ss", f"{clip.start_seconds:.6f}",
                "-i", str(source.file_path),
                "-t", f"{clip.duration_seconds:.6f}",
                "-map", "0:v:0", "-map", "0:a?",
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-r", f"{source.fps:.6f}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-threads", "2",
                "-c:a", "aac", "-movflags", "+faststart",
                str(output_path),
            ]
            process = run_process(
                command,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=180,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if process.returncode != 0:
                raise RuntimeError("FFmpeg 导出失败：" + process.stderr[-2000:])

            checkpoint()
            actual = self.reader.read_metadata(output_path)
            container_duration = self._container_duration(output_path)
            tolerance = max(0.01, 1.0 / actual.fps) + 0.005
            if actual.frame_count <= 0 or actual.duration_seconds <= 0:
                raise ValueError("导出文件没有有效视频帧")
            if abs(actual.duration_seconds - clip.duration_seconds) > tolerance:
                errors.append(f"第{index}段画面时长与请求时长相差超过一帧容差")
            if abs(container_duration - clip.duration_seconds) > tolerance:
                errors.append(f"第{index}段文件播放时长与请求时长相差超过一帧容差")

            exported.append(
                ExportedVideoClip(
                    candidate_id=clip.candidate_id,
                    file_path=output_path,
                    file_signature=file_signature(output_path),
                    source_start_seconds=clip.start_seconds,
                    source_end_seconds=clip.end_seconds,
                    expected_duration_seconds=clip.duration_seconds,
                    video_duration_seconds=actual.duration_seconds,
                    container_duration_seconds=container_duration,
                    duration_tolerance_seconds=tolerance,
                    fps=actual.fps,
                    frame_count=actual.frame_count,
                )
            )

        video_total = sum(clip.video_duration_seconds for clip in exported)
        container_total = sum(clip.container_duration_seconds for clip in exported)
        total_tolerance = sum(clip.duration_tolerance_seconds for clip in exported)
        if task.max_total_duration_seconds is not None and max(video_total, container_total) > task.max_total_duration_seconds + total_tolerance:
            errors.append("导出后的实际总时长超过任务上限及帧级容差")
        return VideoExportReport(
            valid=not errors,
            errors=errors,
            clips=exported,
            total_video_duration_seconds=video_total,
            total_container_duration_seconds=container_total,
        )
