from openai import AsyncOpenAI
from mini_agent.config import Settings
from mini_agent.schemas import VisualObservation

VISION_SYSTEM_PROMPT="""
你是一个视觉内容理解模型。
你的职责是观察用户提供的网页截图或图片，提取画面中的重要信息。
你只负责视觉观察，不负责决定浏览器动作。
你必须只返回下面格式的JSON:
{
    "scene_summary":"画面整体内容",
    "visible_text":[
        "画面中的重要的文字"
    ],
    "key_elements":[
        "画面中的重要要素"
    ],
    "spatial_relations":[
        "重要元素的位置关系"
    ],
    "confidence":0.9
}
规则:
-只能描述图片中真正可见的内容
-不允许编造不存在的元素
-confidence必须在0到1之间
-所有字段都必须返回
-只能返回JSON,不能使用Markdown代码块。
""".strip()

class QwenVisionClient:
    def __init__(self,settings:Settings):
        if not settings.qwen_api_key:
            raise ValueError(
                "没有配置DASHSCOPE_API_KEY"
            )
        self.settings=settings
        self.client=AsyncOpenAI(
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url,
            timeout=60.0,
            max_retries=2,
        )

    async def analyze(self,task:str,screenshot_base64:str)->VisualObservation:
        if not screenshot_base64:
            raise ValueError("截图内容为空")
        image_url=(
            "data:image/png;base64,"
            f"{screenshot_base64}"
        )
        user_prompt=(
            f"用户当前任务:\n{task}\n\n"
            "请观察这张截图，并返回结构化的视觉观察结果。"
        )
        response=await self.client.chat.completions.create(
            model=self.settings.qwen_vl_model,
            messages=[
                {
                    "role":"system",
                    "content":VISION_SYSTEM_PROMPT
                },
                {
                    "role":"user",
                    "content":[
                        {
                            "type":"image_url",
                            "image_url":{
                                "url":image_url
                            },
                        },
                        {
                            "type":"text",
                            "text":user_prompt
                        },
                    ],
                },
            ],
            temperature=0.0,
            max_tokens=1000
        )
        content=response.choices[0].message.content
        if not content:
            raise RuntimeError(
                "Qwen-VL返回了空内容"
            )
        return VisualObservation.model_validate_json(
            content
        )
        





























































