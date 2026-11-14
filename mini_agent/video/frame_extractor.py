from pathlib import Path
import math
import cv2

from mini_agent.video.reader import VideoReader
from mini_agent.video.job_control import checkpoint
from mini_agent.video.schemas import FrameSamplePoint, ShotSegment, VideoFrame, VideoMetadata

class FrameExtractor:
    def __init__(self):
        self.reader=VideoReader()

    def extract(self,video_path:str|Path,output_dir:str|Path,interval_seconds:float=2.0,max_frames:int|None=None)->list[VideoFrame]:
        if not math.isfinite(interval_seconds) or interval_seconds<=0:
            raise ValueError(
                "抽帧间隔必须大于0"
            )
        if max_frames is not None and max_frames < 1:
            raise ValueError("采样帧数上限必须大于0")
        metadata=self.reader.read_metadata(video_path)
        frames_dir=Path(output_dir).resolve()
        frames_dir.mkdir(
            parents=True,
            exist_ok=True
        )
        capture=cv2.VideoCapture(
            str(metadata.file_path)
        )
        extracted_frames:list[VideoFrame]=[]
        step_frames=max(
            1,
            round(
                metadata.fps*interval_seconds
            )
        )
        if max_frames is not None:
            step_frames = max(
                step_frames,
                math.ceil(metadata.frame_count / max_frames),
            )
        try:
            if not capture.isOpened():
                raise ValueError(
                    "无法打开视频文件"
                )
            for frame_index in range(0,metadata.frame_count,step_frames):
                checkpoint()
                capture.set(cv2.CAP_PROP_POS_FRAMES,frame_index)
                success,image=capture.read()
                if not success:
                    continue
                timestamp_seconds=(
                    frame_index/metadata.fps
                )
                image_path=(
                    frames_dir / f"frame_{frame_index:08d}.jpg"
                )
                saved, encoded_image=cv2.imencode(
                    ".jpg",
                    image,
                    [
                        cv2.IMWRITE_JPEG_QUALITY,
                        90
                    ]
                )
                if not saved:
                    raise RuntimeError(
                        f"保存视频帧失败:{image_path}"
                    )
                image_path.write_bytes(encoded_image.tobytes())
                extracted_frames.append(
                    VideoFrame(
                        frame_index=frame_index,
                        timestamp_seconds=timestamp_seconds,
                        image_path=image_path
                    )
                )
        finally:
            capture.release()
        return extracted_frames

    def extract_samples(
        self,
        metadata: VideoMetadata,
        shots: list[ShotSegment],
        samples: list[FrameSamplePoint],
        output_dir: Path,
        max_side: int = 960,
    ) -> list[VideoFrame]:
        if max_side < 1:
            raise ValueError("代表帧最大尺寸必须大于0")
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        shot_map = {shot.shot_id: shot for shot in shots}
        capture = cv2.VideoCapture(str(metadata.file_path))
        frames: list[VideoFrame] = []
        seen: set[int] = set()
        try:
            if not capture.isOpened():
                raise ValueError("无法读取原视频代表帧")
            for sample in sorted(samples, key=lambda item: item.timestamp_seconds):
                checkpoint()
                shot = shot_map[sample.shot_id]
                capture.set(cv2.CAP_PROP_POS_MSEC, sample.timestamp_seconds * 1000)
                success, image = capture.read()
                if not success:
                    fallback_index = min(
                        shot.end_frame_index_exclusive - 1,
                        max(shot.start_frame_index, round(sample.timestamp_seconds * metadata.fps)),
                    )
                    # Some decoders cannot seek back after reaching EOF on a very short video.
                    capture.release()
                    capture = cv2.VideoCapture(str(metadata.file_path))
                    capture.set(cv2.CAP_PROP_POS_FRAMES, fallback_index)
                    success, image = capture.read()
                    if not success:
                        raise ValueError(f"无法读取{sample.timestamp_seconds:.3f}秒的代表帧")
                frame_index = max(0, round(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
                timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if not math.isfinite(timestamp) or (timestamp <= 0 and frame_index > 0):
                    timestamp = frame_index / metadata.fps
                if not shot.start_seconds <= timestamp < shot.end_seconds:
                    frame_index = min(
                        shot.end_frame_index_exclusive - 1,
                        max(shot.start_frame_index, round(sample.timestamp_seconds * metadata.fps)),
                    )
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    success, image = capture.read()
                    if not success:
                        raise ValueError("无法读取镜头边界内的代表帧")
                    timestamp = frame_index / metadata.fps
                if frame_index in seen:
                    continue
                seen.add(frame_index)
                height, width = image.shape[:2]
                if max(height, width) > max_side:
                    scale = max_side / max(height, width)
                    image = cv2.resize(
                        image,
                        (max(1, round(width * scale)), max(1, round(height * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if not success:
                    raise RuntimeError("代表帧编码失败")
                image_path = output_dir / f"frame_{frame_index:08d}.jpg"
                image_path.write_bytes(encoded.tobytes())
                frames.append(VideoFrame(
                    frame_index=frame_index,
                    timestamp_seconds=timestamp,
                    image_path=image_path,
                    shot_id=shot.shot_id,
                    shot_start_seconds=shot.start_seconds,
                    shot_end_seconds=shot.end_seconds,
                    sampling_reason=sample.reason,
                ))
        finally:
            capture.release()
        return frames


















































































