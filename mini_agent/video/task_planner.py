import json

from openai import AsyncOpenAI

from mini_agent.config import Settings
from mini_agent.video.audio_visual_events import EVENT_NAMES
from mini_agent.video.schemas import (
    ClipCandidate,
    ClipSelection,
    VideoMetadata,
    VideoTaskSpec,
)


class VideoTaskPlanner:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AsyncOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            timeout=60.0,
            max_retries=2,
        )

    async def parse_task(
        self,
        request: str,
        metadata: VideoMetadata,
    ) -> VideoTaskSpec:
        request = request.strip()
        if not request:
            raise ValueError("视频检索需求不能为空")

        schema_text = json.dumps(
            VideoTaskSpec.model_json_schema(),
            ensure_ascii=False,
            indent=2,
        )
        system_prompt = (
            "你负责将用户的视频检索需求转换成结构化任务。\n"
            "只返回一个符合下面 JSON Schema 的 JSON 对象，返回Schema中全部字段。\n"
            "规则：\n"
            "1. query 保留用户要找的画面、声音、具体对白、组合条件和排除条件，不能只保留画面。\n"
            "2. 用户明确指定的时长换算为秒。单段时长作为裁剪偏好，不要求精确凑时长。\n"
            "3. 未指定片段数量时max_clips为null，由内容自动选择数量。明确指定时才填写数量上限。\n"
            "4. max_total_duration_seconds仅在用户明确限制总时长时填写，否则为null；不要由单段时长与数量推导。\n"
            "5. 默认优先清晰画面；用户明确表示不考虑清晰度时设为false。\n"
            "6. clip_duration_seconds是用户的单段时长偏好，未指定为null；短镜头可以短于偏好，连续匹配内容可更长。\n"
            "7. 你没有看到视频画面，不得声称已找到内容、生成时间段或选择片段。\n"
            "8. 不返回工具调用、文件路径、解释或 Markdown 代码块。\n\n"
            "9. 只要求可见画面时audio_condition_type=none，audio_event_classes=[]。用户明确要求声音且发声源可见、并能准确映射下列正向事件时，"
            "audio_condition_type=event_activity，audio_event_classes填写对应标准英文类别（最多4类）。不能增加用户没有要求的听觉约束。\n"
            "10. 用户要求说出某句话、台词含义或对话内容时为spoken_content，不能用男性/女性讲话代替具体台词条件；"
            "要求画外声、没有某声音、不支持的声音事件时为unsupported。以上两类audio_event_classes=[]，完整条件仍保留在query。\n"
            "11. visual_query_en填写准确英文视觉查询，供本地CLIP/QD-DETR使用。纯视觉需求保留可见动作、关系、外观及否定条件，"
            "不要将骑木马变成马嘶鸣或增加声音要求。声音任务只翻译可见子条件；完整声音与对白条件不得从query删除。\n"
            "12. event_activity还要求服装、特定人物、人物物体关系或额外动作时requires_extra_visual_grounding=true；"
            "只要求支持的发声事件及发声源可见时为false。声音类别明确为或关系时audio_match_mode=any，否则all；时序关系原样保留在query。\n"
            "受支持声画事件（英文类别:中文含义）："+json.dumps(EVENT_NAMES, ensure_ascii=False)+"\n\n"
            f"JSON Schema：\n{schema_text}"
        )
        user_prompt = (
            f"用户需求：\n{request}\n\n"
            f"源视频时长：{metadata.duration_seconds:.3f}秒\n"
            f"源视频分辨率：{metadata.width}×{metadata.height}\n"
            "请仅解析任务约束。输出时长上限可以大于源视频时长，"
            "实际可用内容由后续执行程序判断。"
        )

        response = await self.client.chat.completions.create(
            model=self.settings.deepseek_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=1000,
        )
        if not response.choices:
            raise RuntimeError("DeepSeek 没有返回视频任务")
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("DeepSeek 返回了空的视频任务")

        return VideoTaskSpec.model_validate_json(content)

    async def select_clips(
        self,
        task: VideoTaskSpec,
        candidates: list[ClipCandidate],
        feedback: str | None = None,
    ) -> ClipSelection:
        if not candidates:
            return ClipSelection(
                choices=[],
                summary="当前视觉证据未产生满足目标与单段时长要求的候选片段。",
            )
        schema_text = json.dumps(ClipSelection.model_json_schema(), ensure_ascii=False)
        system_prompt = (
            "你负责根据任务与已有视觉证据选择候选视频片段。\n"
            "仅返回符合 JSON Schema 的 JSON 对象。\n"
            "只能选择输入中的candidate_id，不能创造或修改时间段。\n"
            "优先满足query中的语义条件，参考候选中的全部observations。\n"
            "prefer_clear_frames为true时，在语义合适的候选中优先清晰画面。\n"
            "片段数量与总时长仅在非null时构成上限；单段时长是偏好，不要因此排除短片段。选择的时间段不得重叠。\n"
            "不要重复选择同一个ID，也不要为了凑数量选择不合适的片段。\n"
            "数量未指定时，由相关内容和重复程度自动决定：保留不同的有价值片段，去掉重复或不匹配片段，不默认只选3段。\n"
            "reason说明具体画面证据，不编造未观察到的连续动作。\n"
            "边界是程序生成的粗略窗口，不能宣称整段内容或精确边界已验证。\n"
            "所有理由使用简短中文；不返回 Markdown 代码块。\n\n"
            f"JSON Schema：{schema_text}"
        )
        payload = {
            "task": task.model_dump(mode="json"),
            "candidates": [item.model_dump(mode="json") for item in candidates],
        }
        user_prompt = json.dumps(payload, ensure_ascii=False)
        if feedback:
            user_prompt += "\n\n上次选择未通过程序校验，请根据以下反馈重新选择：\n" + feedback

        response = await self.client.chat.completions.create(
            model=self.settings.deepseek_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=2000,
        )
        if not response.choices or not response.choices[0].message.content:
            raise RuntimeError("DeepSeek 返回了空的片段选择")
        return ClipSelection.model_validate_json(response.choices[0].message.content)

    async def close(self) -> None:
        await self.client.close()
