import asyncio
import base64
import json

import cv2
import numpy as np
from openai import AsyncOpenAI

from mini_agent.config import Settings
from mini_agent.video.schemas import (
    VideoFrame,
    VideoFrameEvidence,
    VideoObservationBatch,
    VideoTaskSpec,
)


class VideoVisionAnalyzer:
    def __init__(self, settings: Settings):
        if not settings.qwen_api_key:
            raise ValueError("视频视觉分析需要配置 DASHSCOPE_API_KEY")
        self.settings = settings
        self.client = AsyncOpenAI(
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url,
            timeout=60.0,
            max_retries=2,
        )

    @staticmethod
    def _image_url(frame: VideoFrame) -> str:
        image = cv2.imdecode(
            np.frombuffer(frame.image_path.read_bytes(), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            raise ValueError(f"无法读取候选帧: {frame.image_path}")
        height, width = image.shape[:2]
        if max(height, width) > 960:
            scale = 960 / max(height, width)
            image = cv2.resize(
                image,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        success, encoded = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85]
        )
        if not success:
            raise RuntimeError("候选帧编码失败")
        encoded_text = base64.b64encode(encoded.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded_text}"

    async def analyze_batch(
        self,
        task: VideoTaskSpec,
        frames: list[VideoFrame],
    ) -> list[VideoFrameEvidence]:
        if not frames or len(frames) > 4:
            raise ValueError("每批视觉观察需要1到4张候选帧")
        frame_ids = [frame.frame_index for frame in frames]
        if len(set(frame_ids)) != len(frame_ids):
            raise ValueError("候选帧编号重复")

        schema_text = json.dumps(
            VideoObservationBatch.model_json_schema(),
            ensure_ascii=False,
        )
        system_prompt = (
            "你负责观察视频的采样画面，判断每张图片是否与检索目标匹配。\n"
            "只返回符合以下 JSON Schema 的 JSON，不返回 Markdown 代码块。\n"
            "对每张图片恰好返回一个 observation，使用提供的 frame_index。\n"
            "description 与 reason 均使用简短中文，分别不超过100字。\n"
            "描述可见内容，保留用户的组合条件和排除条件。\n"
            "画面不匹配或证据不足时，matches_query 为 false，并说明原因。\n"
            "这些是稀疏采样帧，不保证动作连续；不要补写未看到的事件。\n"
            "relevance_score 是任务相关度，不是对事实正确性的置信概率。\n"
            "清晰度和亮度只是参考，不能单凭画面较暗否定日落等场景。\n"
            "不要生成时间段，不要给未提供的帧编号。\n\n"
            f"JSON Schema：{schema_text}"
        )
        audio_note = ""
        if task.audio_condition_type == "event_activity":
            system_prompt += (
                "\n本次任务包含声音条件，但你只接收静态图片。matches_query和relevance_score仅判断可见必要条件，"
                "例如主体、物体、可见动作和视觉排除条件；不要因为图片无法听见声音就判为不匹配。"
                "也不得推断已经有声音、说了某句话或整条检索要求成立。有声事件由后续本地声画模型另行核验。"
            )
            audio_note = "\n后续另行检查的有声事件类别："+json.dumps(task.audio_event_classes, ensure_ascii=False)
        content = [
            {
                "type": "text",
                "text": f"完整检索目标（声音不由图片证明）：{task.query}{audio_note}\n优先清晰画面：{task.prefer_clear_frames}",
            }
        ]
        for frame in frames:
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"下面图片的 frame_index={frame.frame_index}；"
                        f"源时间戳={frame.timestamp_seconds:.3f}秒；"
                        f"镜头编号={frame.shot_id}；"
                        f"镜头起点={frame.shot_start_seconds}秒；"
                        f"镜头终点={frame.shot_end_seconds if frame.shot_end_seconds is not None else '尚未确认，当前为提前提交帧'}；"
                        f"清晰度={frame.sharpness_score}；"
                        f"平均亮度={frame.brightness_score}。"
                    ),
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": await asyncio.to_thread(self._image_url, frame),
                    },
                }
            )

        response = await self.client.chat.completions.create(
            model=self.settings.qwen_vl_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            temperature=0.0,
            max_tokens=2000,
        )
        if not response.choices or not response.choices[0].message.content:
            raise RuntimeError("Qwen3-VL 返回了空的视频观察")
        result = VideoObservationBatch.model_validate_json(
            response.choices[0].message.content
        )
        result_ids = [item.frame_index for item in result.observations]
        if len(result_ids) != len(frame_ids) or set(result_ids) != set(frame_ids):
            raise ValueError("视觉结果中的帧编号与输入不一致")

        observations = {item.frame_index: item for item in result.observations}
        return [
            VideoFrameEvidence(
                frame=frame,
                observation=observations[frame.frame_index],
            )
            for frame in frames
        ]

    async def close(self) -> None:
        await self.client.close()
