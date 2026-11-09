import math
from bisect import bisect_left, bisect_right
from pathlib import Path
from time import perf_counter

from mini_agent.video.frame_extractor import FrameExtractor
from mini_agent.video.quality_analyzer import FrameQualityAnalyzer
from mini_agent.video.schemas import (
    FrameSamplePoint,
    ShotSamplingConfig,
    ShotSamplingResult,
    ShotSamplingStats,
    ShotSegment,
    VideoMetadata,
)
from mini_agent.video.shot_detector import ScanPoint, ShotDetector, histogram_distance


class ShotFrameSampler:
    def __init__(self, config: ShotSamplingConfig | None = None):
        self.config = config or ShotSamplingConfig()
        self.detector = ShotDetector(self.config)
        self.extractor = FrameExtractor()
        self.quality = FrameQualityAnalyzer(max_side=self.config.quality_max_side)

    def iter_prepare(self, metadata: VideoMetadata, output_dir: Path):
        """Yield new frame batches immediately, then one finalized sampling result."""
        started = perf_counter()
        rate, width, height = self.detector.scan_geometry(metadata)
        if self.config.static_coverage_seconds < 1 / rate:
            raise ValueError("时间覆盖间隔不能小于缩略图扫描间隔")
        frames = []
        shots = []
        seen = set()
        last_kept = None
        last_timestamp = None
        current_id = None
        current_frames = []
        scan_count = 0
        windows = self.detector.iter_windows(metadata)
        try:
            for window in windows:
                shot = window.shot
                scan_count = window.scan_frame_count
                if current_id != shot.shot_id:
                    current_id = shot.shot_id
                    current_frames = []
                    last_kept = None
                    last_timestamp = None
                margin = min(0.2, (shot.end_seconds - window.start_seconds) / 5)
                pool = [p for p in window.points
                        if window.start_seconds + margin <= p.timestamp_seconds < shot.end_seconds - margin
                        and not p.transient]
                local_shot = shot.model_copy(update={"start_seconds": window.start_seconds})
                candidates = self._select_within_shot(local_shot, pool)
                points = {p.timestamp_seconds: p for p in pool}
                samples = []
                for sample in candidates:
                    point = points.get(sample.timestamp_seconds)
                    coverage_due = (last_timestamp is None or
                                    sample.timestamp_seconds - last_timestamp >= self.config.static_coverage_seconds)
                    if coverage_due or (point is not None and (last_kept is None or self._novel(point, last_kept))):
                        samples.append(sample)
                        last_kept = point
                        last_timestamp = sample.timestamp_seconds
                # Cover a long unchanged tail even when the window's best frame is too early.
                if (last_timestamp is not None and pool and
                        shot.end_seconds - last_timestamp > self.config.static_coverage_seconds):
                    target = last_timestamp + self.config.static_coverage_seconds
                    point = min(pool, key=lambda p: abs(p.timestamp_seconds - target))
                    if point.timestamp_seconds > last_timestamp:
                        samples.append(FrameSamplePoint(
                            shot_id=shot.shot_id, timestamp_seconds=point.timestamp_seconds,
                            reason="长镜头提前提交：时间覆盖保底",
                        ))
                        last_kept, last_timestamp = point, point.timestamp_seconds
                batch = self.extractor.extract_samples(
                    metadata, [shot], samples, output_dir,
                    max_side=self.config.representative_max_side,
                ) if samples else []
                batch = self.quality.analyze_many([f for f in batch if f.frame_index not in seen])
                for frame in batch:
                    seen.add(frame.frame_index)
                    current_frames.append(frame.frame_index)
                    if not window.final:
                        frame.shot_end_seconds = None
                frames.extend(batch)
                if window.final:
                    shot.representative_frame_indices = list(current_frames)
                    if not current_frames:
                        raise ValueError(f"{shot.shot_id}没有成功读取到内部代表帧")
                    shots.append(shot)
                if batch:
                    status = "镜头结束提交" if window.final else "长镜头提前提交"
                    print(f"[{status}] {shot.shot_id} 已扫描到{shot.end_seconds:.2f}秒，新增{len(batch)}张代表帧", flush=True)
                    yield batch
        finally:
            windows.close()
        shot_map = {shot.shot_id: shot for shot in shots}
        frames = sorted([
            frame.model_copy(update={"shot_end_seconds": shot_map[frame.shot_id].end_seconds})
            for frame in frames
        ], key=lambda frame: frame.timestamp_seconds)
        baseline = math.ceil(metadata.frame_count / max(1, round(metadata.fps * 2)))
        yield ShotSamplingResult(
            frames=frames, shots=shots, config=self.config,
            stats=ShotSamplingStats(
                original_frame_count=metadata.frame_count, scan_frame_count=scan_count,
                shot_count=len(shots), representative_frame_count=len(frames),
                uniform_two_second_frame_count=baseline,
                reduction_vs_original=1 - len(frames) / metadata.frame_count,
                reduction_vs_uniform=1 - len(frames) / baseline,
                effective_scan_fps=rate, scan_width=width, scan_height=height,
                elapsed_seconds=perf_counter() - started,
            ),
        )

    def prepare(self, metadata: VideoMetadata, output_dir: Path) -> ShotSamplingResult:
        started = perf_counter()
        if self.config.static_coverage_seconds < 1.0 / min(self.config.scan_fps, metadata.fps):
            raise ValueError("时间覆盖间隔不能小于缩略图扫描间隔")
        print("正在低分辨率扫描并定位镜头切换……", flush=True)
        scan = self.detector.scan(metadata)
        print(f"发现{len(scan.shots)}个镜头，正在挑选镜头内部代表帧……", flush=True)
        timestamps = [point.timestamp_seconds for point in scan.points]
        samples: list[FrameSamplePoint] = []
        for shot in scan.shots:
            margin = min(0.2, shot.duration_seconds / 5)
            left = bisect_left(timestamps, shot.start_seconds + margin)
            right = bisect_left(timestamps, shot.end_seconds - margin)
            pool = [point for point in scan.points[left:right] if not point.transient]
            samples.extend(self._select_within_shot(shot, pool))

        frames = self.extractor.extract_samples(
            metadata, scan.shots, samples, output_dir,
            max_side=self.config.representative_max_side,
        )
        frames = self.quality.analyze_many(frames)
        groups = {shot.shot_id: [] for shot in scan.shots}
        for frame in frames:
            groups[frame.shot_id].append(frame.frame_index)
        for shot in scan.shots:
            shot.representative_frame_indices = groups[shot.shot_id]
            if not shot.representative_frame_indices:
                raise ValueError(f"{shot.shot_id}没有成功读取到内部代表帧")

        baseline_step = max(1, round(metadata.fps * 2.0))
        baseline_count = math.ceil(metadata.frame_count / baseline_step)
        return ShotSamplingResult(
            frames=frames,
            shots=scan.shots,
            config=self.config,
            stats=ShotSamplingStats(
                original_frame_count=metadata.frame_count,
                scan_frame_count=len(scan.points),
                shot_count=len(scan.shots),
                representative_frame_count=len(frames),
                uniform_two_second_frame_count=baseline_count,
                reduction_vs_original=1.0 - len(frames) / metadata.frame_count,
                reduction_vs_uniform=1.0 - len(frames) / baseline_count,
                effective_scan_fps=scan.scan_fps,
                scan_width=scan.width,
                scan_height=scan.height,
                elapsed_seconds=perf_counter() - started,
            ),
        )

    def _novel(self, first: ScanPoint, second: ScanPoint, strong: bool = False) -> bool:
        distance = histogram_distance(first.histogram, second.histogram)
        bits = (first.visual_hash ^ second.visual_hash).bit_count()
        threshold = self.config.novelty_histogram_distance * (1.5 if strong else 1.0)
        return distance >= threshold or (
            bits >= self.config.novelty_hash_bits * (2 if strong else 1)
            and distance >= threshold / 3
        )

    @staticmethod
    def _quality(point: ScanPoint, center: float, span: float) -> float:
        brightness = 1.0 - abs(point.brightness - 127.5) / 127.5
        center_score = 1.0 - min(1.0, abs(point.timestamp_seconds - center) / max(span / 2, 0.001))
        return (
            math.log1p(point.sharpness)
            + 0.2 * brightness
            + 0.15 * center_score
            - point.rgb_change
            - 0.8 * point.histogram_distance
        )

    def _select_within_shot(self, shot: ShotSegment, pool: list[ScanPoint]) -> list[FrameSamplePoint]:
        if not pool:
            return [FrameSamplePoint(
                shot_id=shot.shot_id,
                timestamp_seconds=(shot.start_seconds + shot.end_seconds) / 2,
                reason="短镜头缺少内部扫描点，读取原视频中点",
            )]
        times = [point.timestamp_seconds for point in pool]
        chosen: dict[float, tuple[ScanPoint, str]] = {}
        last_kept = None
        window_start = shot.start_seconds
        while window_start < shot.end_seconds:
            window_end = min(shot.end_seconds, window_start + self.config.representative_window_seconds)
            window = pool[bisect_left(times, window_start):bisect_left(times, window_end)]
            if window:
                center = (window_start + window_end) / 2
                primary = max(window, key=lambda point: self._quality(point, center, window_end - window_start))
                representatives = [primary]
                alternatives = [
                    point for point in window
                    if abs(point.timestamp_seconds - primary.timestamp_seconds) >= 2.0
                    and self._novel(point, primary, strong=True)
                ]
                if alternatives:
                    representatives.append(max(alternatives, key=lambda point: self._quality(point, center, window_end - window_start)))
                for point in sorted(representatives, key=lambda item: item.timestamp_seconds):
                    if last_kept is None or self._novel(point, last_kept):
                        reason = "镜头内部清晰稳定代表帧" if last_kept is None else "镜头内部出现新的视觉内容"
                        chosen[point.timestamp_seconds] = (point, reason)
                        last_kept = point
            window_start = window_end

        anchors = [shot.start_seconds] + sorted(chosen) + [shot.end_seconds]
        for left, right in zip(anchors, anchors[1:]):
            current = left
            while right - current > self.config.static_coverage_seconds:
                target = current + self.config.static_coverage_seconds
                index = bisect_right(times, target) - 1
                if index < 0 or times[index] <= current:
                    index = bisect_right(times, current)
                if index >= len(pool) or times[index] >= right:
                    break
                point = pool[index]
                chosen.setdefault(point.timestamp_seconds, (point, "长镜头时间覆盖保底，避免只保留开头"))
                current = point.timestamp_seconds

        return [
            FrameSamplePoint(shot_id=shot.shot_id, timestamp_seconds=timestamp, reason=reason)
            for timestamp, (_, reason) in sorted(chosen.items())
        ]
