import cv2
import numpy as np

from mini_agent.video.schemas import VideoFrame


class FrameQualityAnalyzer:
    def __init__(self, max_side: int = 320):
        if max_side < 1:
            raise ValueError("质量分析图像尺寸必须大于0")
        self.max_side = max_side

    def analyze(self, frame: VideoFrame) -> VideoFrame:
        image_bytes = frame.image_path.read_bytes()
        image = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            raise ValueError(f"无法读取视频帧: {frame.image_path}")

        height, width = image.shape[:2]
        if max(height, width) > self.max_side:
            scale = self.max_side / max(height, width)
            image = cv2.resize(
                image,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharpness_score = float(
            cv2.Laplacian(gray_image, cv2.CV_64F).var()
        )
        brightness_score = float(gray_image.mean())

        return frame.model_copy(
            update={
                "sharpness_score": round(sharpness_score, 2),
                "brightness_score": round(brightness_score, 2),
            }
        )

    def analyze_many(
        self,
        frames: list[VideoFrame],
    ) -> list[VideoFrame]:
        return [self.analyze(frame) for frame in frames]

