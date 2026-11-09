from pathlib import Path
import cv2
from mini_agent.video.schemas import VideoMetadata

class VideoReader:
    def read_metadata(self,video_path:str|Path)->VideoMetadata:
        path=Path(video_path).resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"视频文件不存在{path}"
            )
        if not path.is_file():
            raise ValueError(
                f"视频路径不是文件:{path}"
            )
        capture=cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError(
                    f"无法打开视频文件{path}"
                )
            fps=float(
                capture.get(cv2.CAP_PROP_FPS)
            )
            frame_count=int(
                capture.get(cv2.CAP_PROP_FRAME_COUNT)
            )
            width=int(
                capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            )
            height=int(
                capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )
            if fps<=0:
                raise ValueError(
                    "视频FPS无效"
                )
            duration_seconds=(
                frame_count/fps
            )
            return VideoMetadata(
                file_path=path,
                duration_seconds=duration_seconds,
                fps=fps,
                frame_count=frame_count,
                width=width,
                height=height
            )
        finally:
            capture.release()






















































