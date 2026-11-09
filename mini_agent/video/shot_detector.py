import math
from dataclasses import dataclass

import cv2
import imageio_ffmpeg
import numpy as np

from mini_agent.video.schemas import ShotSamplingConfig, ShotSegment, VideoMetadata


@dataclass(slots=True)
class ScanPoint:
    timestamp_seconds: float
    histogram: np.ndarray
    histogram_distance: float
    rgb_change: float
    sharpness: float
    brightness: float
    visual_hash: int
    transient: bool = False


@dataclass(slots=True)
class ShotScanResult:
    shots: list[ShotSegment]
    points: list[ScanPoint]
    scan_fps: float
    width: int
    height: int


@dataclass(slots=True)
class ShotWindow:
    shot: ShotSegment
    start_seconds: float
    points: list[ScanPoint]
    final: bool
    scan_frame_count: int


def hsv_histogram(rgb_image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    histogram = cv2.calcHist(
        [hsv], [0, 1, 2], None, [16, 4, 4], [0, 180, 0, 256, 0, 256]
    )
    return cv2.normalize(histogram, histogram, alpha=1.0, norm_type=cv2.NORM_L1).reshape(-1)


def histogram_distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(cv2.compareHist(first, second, cv2.HISTCMP_BHATTACHARYYA))


def difference_hash(gray_image: np.ndarray) -> int:
    small = cv2.resize(gray_image, (9, 8), interpolation=cv2.INTER_AREA)
    value = 0
    for bit in (small[:, 1:] > small[:, :-1]).flat:
        value = (value << 1) | int(bit)
    return value


class ShotDetector:
    def __init__(self, config: ShotSamplingConfig | None = None):
        self.config = config or ShotSamplingConfig()

    def scan_geometry(self, metadata: VideoMetadata) -> tuple[float, int, int]:
        if metadata.frame_count <= 0 or metadata.duration_seconds <= 0:
            raise ValueError("视频没有有效的时长或帧数")
        rate = min(max(self.config.scan_fps, 1.0 / metadata.duration_seconds), metadata.fps)
        scale = min(1.0, self.config.thumbnail_max_side / max(metadata.width, metadata.height))
        width = max(2, round(metadata.width * scale / 2) * 2)
        height = max(2, round(metadata.height * scale / 2) * 2)
        return rate, width, height

    def iter_points(self, metadata: VideoMetadata):
        rate, width, height = self.scan_geometry(metadata)
        reader = imageio_ffmpeg.read_frames(
            str(metadata.file_path),
            output_params=[
                "-vf", f"fps={rate:.8f}:start_time=0,scale={width}:{height}",
                "-vsync", "0",
            ],
        )
        previous_image = None
        previous_histogram = None
        next_progress = 60.0
        try:
            stream_metadata = next(reader)
            if tuple(stream_metadata["size"]) != (width, height):
                raise ValueError("缩略图扫描尺寸与配置不一致")
            for index, raw_frame in enumerate(reader):
                timestamp = index / rate
                if timestamp >= metadata.duration_seconds:
                    break
                image = np.frombuffer(raw_frame, dtype=np.uint8).reshape(height, width, 3)
                histogram = hsv_histogram(image)
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                distance = histogram_distance(previous_histogram, histogram) if previous_histogram is not None else 0.0
                rgb_change = float(cv2.absdiff(image, previous_image).mean()) / 255.0 if previous_image is not None else 0.0
                point = ScanPoint(
                        timestamp_seconds=timestamp,
                        histogram=histogram,
                        histogram_distance=distance,
                        rgb_change=rgb_change,
                        sharpness=float(cv2.Laplacian(gray, cv2.CV_32F).var()),
                        brightness=float(gray.mean()),
                        visual_hash=difference_hash(gray),
                )
                previous_image = image
                previous_histogram = histogram
                yield point
                if timestamp >= next_progress:
                    print(f"缩略图扫描已到视频第{timestamp:.0f}秒……", flush=True)
                    next_progress += 60.0
        finally:
            reader.close()

    def scan(self, metadata: VideoMetadata) -> ShotScanResult:
        rate, width, height = self.scan_geometry(metadata)
        points = list(self.iter_points(metadata))
        if not points:
            raise ValueError("无法读取视频缩略图")

        peak_indices = self._find_peaks(points, rate, metadata.duration_seconds)
        cuts = self._refine_cuts(metadata, points, peak_indices, rate, width, height)
        boundaries = [(0.0, 0, None, "video_start")] + cuts
        shots: list[ShotSegment] = []
        for index, (start, frame_index, distance, method) in enumerate(boundaries):
            if index + 1 < len(boundaries):
                end, end_frame = boundaries[index + 1][:2]
            else:
                end, end_frame = metadata.duration_seconds, metadata.frame_count
            shots.append(
                ShotSegment(
                    shot_id=f"shot_{index + 1:04d}",
                    start_seconds=start,
                    end_seconds=end,
                    start_frame_index=frame_index,
                    end_frame_index_exclusive=end_frame,
                    boundary_histogram_distance=distance,
                    boundary_method=method,
                    boundary_scan_resolution_seconds=1.0 / rate,
                )
            )
        return ShotScanResult(shots=shots, points=points, scan_fps=rate, width=width, height=height)

    def iter_windows(self, metadata: VideoMetadata):
        """Emit stable windows while decoding, retaining only a bounded thumbnail buffer."""
        rate, width, height = self.scan_geometry(metadata)
        radius_seconds = max(2, math.ceil(self.config.adaptive_window_seconds * rate)) / rate
        # Look ahead for local peaks, flash suppression and adjacent-peak competition.
        lookahead = radius_seconds + self.config.min_shot_seconds + 2 / rate
        buffer: list[ScanPoint] = []
        shot_number = 1
        shot_start, shot_start_frame = 0.0, 0
        boundary_distance, boundary_method = None, "video_start"
        window_start = 0.0
        checked_until = -1.0
        scan_count = 0

        def make_window(end: float, end_frame: int, final: bool) -> ShotWindow:
            shot = ShotSegment(
                shot_id=f"shot_{shot_number:04d}", start_seconds=shot_start,
                end_seconds=end, start_frame_index=shot_start_frame,
                end_frame_index_exclusive=end_frame,
                boundary_histogram_distance=boundary_distance,
                boundary_method=boundary_method,
                boundary_scan_resolution_seconds=1 / rate,
            )
            return ShotWindow(
                shot=shot, start_seconds=window_start,
                points=[p for p in buffer if window_start <= p.timestamp_seconds < end],
                final=final, scan_frame_count=scan_count,
            )

        source = self.iter_points(metadata)
        try:
            # None flushes the remaining lookahead and the final shot at EOF.
            from itertools import chain
            for point in chain(source, [None]):
                if point is not None:
                    buffer.append(point)
                    scan_count += 1
                    mature_until = point.timestamp_seconds - lookahead
                else:
                    mature_until = metadata.duration_seconds
                if mature_until <= checked_until:
                    continue
                indices = self._find_peaks(buffer, rate, metadata.duration_seconds)
                indices = [i for i in indices if checked_until < buffer[i].timestamp_seconds <= mature_until]
                cuts = self._refine_cuts(metadata, buffer, indices, rate, width, height)
                for end, end_frame, distance, method in cuts:
                    if end - shot_start < self.config.min_shot_seconds or end <= window_start:
                        continue
                    while window_start + self.config.representative_window_seconds < end:
                        window_end = window_start + self.config.representative_window_seconds
                        yield make_window(window_end, min(metadata.frame_count, math.ceil(window_end * metadata.fps)), False)
                        window_start = window_end
                    yield make_window(end, end_frame, True)
                    shot_number += 1
                    shot_start, shot_start_frame = end, end_frame
                    boundary_distance, boundary_method = distance, method
                    window_start = end
                checked_until = mature_until
                # A future refined cut can move left by at most 1.5 scan intervals.
                safe_until = mature_until - 1.5 / rate
                while window_start + self.config.representative_window_seconds <= safe_until:
                    window_end = window_start + self.config.representative_window_seconds
                    yield make_window(window_end, min(metadata.frame_count, math.ceil(window_end * metadata.fps)), False)
                    window_start = window_end
                if point is None:
                    if not scan_count:
                        raise ValueError("无法读取视频缩略图")
                    yield make_window(metadata.duration_seconds, metadata.frame_count, True)
                else:
                    retain_from = min(window_start, checked_until - radius_seconds - self.config.min_shot_seconds - 2 / rate)
                    buffer = [p for p in buffer if p.timestamp_seconds >= retain_from]
        finally:
            source.close()

    def _find_peaks(self, points: list[ScanPoint], rate: float, duration: float) -> list[int]:
        distances = np.array([point.histogram_distance for point in points])
        radius = max(2, math.ceil(self.config.adaptive_window_seconds * rate))
        flash_indices: set[int] = set()
        for index in range(1, len(points) - 1):
            if (
                self.config.suppress_exposure_flashes
                and (points[index].brightness > 240 or points[index].brightness < 8)
                and points[index].sharpness < 5.0
                and abs(points[index].brightness - points[index - 1].brightness) > 40
                and distances[index] >= self.config.histogram_threshold
                and distances[index + 1] >= self.config.histogram_threshold
                and histogram_distance(points[index - 1].histogram, points[index + 1].histogram) < 0.08
            ):
                flash_indices.update((index, index + 1))
        for index in flash_indices:
            points[index].transient = True

        peaks = []
        for index in range(1, len(points)):
            if index in flash_indices:
                continue
            value = distances[index]
            if value < self.config.histogram_threshold or points[index].rgb_change < self.config.minimum_rgb_change:
                continue
            neighborhood = distances[max(1, index - radius): min(len(points), index + radius + 1)]
            median = float(np.median(neighborhood))
            mad = float(np.median(np.abs(neighborhood - median)))
            threshold = min(0.95, max(self.config.histogram_threshold, median + max(0.08, self.config.mad_multiplier * mad)))
            if value < threshold or value < distances[index - 1]:
                continue
            if index + 1 < len(points) and value < distances[index + 1]:
                continue
            timestamp = points[index].timestamp_seconds
            if self.config.min_shot_seconds <= timestamp <= duration - self.config.min_shot_seconds:
                peaks.append(index)

        accepted: list[int] = []
        for index in peaks:
            if not accepted or points[index].timestamp_seconds - points[accepted[-1]].timestamp_seconds >= self.config.min_shot_seconds:
                accepted.append(index)
            elif distances[index] > distances[accepted[-1]]:
                accepted[-1] = index
        return accepted

    def _refine_cuts(
        self, metadata: VideoMetadata, points: list[ScanPoint], indices: list[int],
        rate: float, width: int, height: int,
    ) -> list[tuple[float, int, float, str]]:
        if not indices:
            return []
        capture = cv2.VideoCapture(str(metadata.file_path))
        if not capture.isOpened():
            raise ValueError("无法打开原视频细化镜头边界")
        cuts = []
        try:
            for index in indices:
                peak = points[index]
                left_time = max(0.0, peak.timestamp_seconds - 1.5 / rate)
                right_time = min(metadata.duration_seconds, peak.timestamp_seconds + 1.0 / rate)
                capture.set(cv2.CAP_PROP_POS_MSEC, left_time * 1000)
                previous = None
                best = None
                best_distance = 0.0
                while True:
                    success, image = capture.read()
                    if not success:
                        break
                    frame_index = max(0, round(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
                    timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    if not math.isfinite(timestamp) or (timestamp <= 0 and frame_index > 0):
                        timestamp = frame_index / metadata.fps
                    if timestamp > right_time or frame_index >= metadata.frame_count:
                        break
                    thumbnail = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                    histogram = hsv_histogram(cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB))
                    if previous is not None:
                        distance = histogram_distance(previous, histogram)
                        if distance > best_distance:
                            best_distance = distance
                            best = (timestamp, frame_index)
                    previous = histogram
                if best is not None and best_distance >= 0.2:
                    cuts.append((best[0], best[1], best_distance, "hsv_peak_refined"))
                else:
                    estimate = min(metadata.frame_count - 1, round(peak.timestamp_seconds * metadata.fps))
                    cuts.append((estimate / metadata.fps, estimate, peak.histogram_distance, "hsv_peak_estimate"))
        finally:
            capture.release()

        filtered = []
        last_time = 0.0
        for cut in sorted(cuts):
            if cut[0] - last_time >= self.config.min_shot_seconds and metadata.duration_seconds - cut[0] >= self.config.min_shot_seconds:
                filtered.append(cut)
                last_time = cut[0]
        return filtered
