from openai import AsyncOpenAI
from pydantic import ValidationError

from mini_agent.config import Settings
from mini_agent.schemas import (
    ActionResult,
    AgentDecision,
    BrowserAction,
    BrowserState,
    VisualObservation
)

SYSTEM_PROMPT="""
    你是一个浏览器自动化Agent
    你的任务是根据用户目标和当前网页状态，
    选择下一步浏览器动作。
    你每次只能返回一个动作。
    可用动作：
    1.打开网页
    {"action":"goto","url":"https://example.com"}
    2.点击元素
    {"action":"click","index":0}
    3.输入文字
    {"action":"type","index":0,"text":"需要输入的文字"}
    4. 滚动页面
    {"action": "scroll", "direction": "down", "amount": 600}
    5. 完成任务
    {
      "action": "done",
      "success": true,
      "answer": "最终答案"
    }
    你必须返回以下 JSON 结构：
    {
      "observation": "对当前页面的简短总结",
      "next_goal": "这一轮准备完成的目标",
      "action": {
        "action": "动作名称"
      }
    }
    规则：
    - 只能返回 JSON，不能使用 Markdown 代码块。
    - 每次只能返回一个动作。
    - 点击或输入只能使用页面中实际存在的元素编号。
    - 不允许编造元素编号。
    - 任务完成后必须返回 done。
    - 当前页面不能完成任务时，选择 goto、click、type 或 scroll。
""".strip()

class DeepSeekClient:
    def __init__(self,settings:Settings):
        self.settings=settings
        self.client=AsyncOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            timeout=60.0,
            max_retries=2
        )
    async def decide(self,task:str,state:BrowserState,previous_result:ActionResult|None=None,visual_observation:VisualObservation|None=None)->AgentDecision:
        if previous_result is None:
            previous_result_text="这是任务第一步"
        else:
            previous_result_text=(
                f"执行成功：{previous_result.success}\n"
                f"执行结果：{previous_result.message}"
            )
        if visual_observation is None:
            visual_observation_text=(
                "本轮没有视觉观察结果,只能参考网页DOM信息"
            )
        else:
            visual_observation_text=(
                visual_observation.to_prompt_text()
            )

        user_prompt=(
            f"用户任务:\n{task}\n\n"
            f"上一步执行结果：\n"
            f"{previous_result_text}\n\n"
            f"当前网页DOM状态:\n"
            f"{state.to_prompt_text()}\n\n"
            f"当前页面视觉观察：\n"
            f"{visual_observation_text}"
        )
        response=await self.client.chat.completions.create(
            model=self.settings.deepseek_model,
            messages=[
                {
                    "role":"system",
                    "content":SYSTEM_PROMPT,
                },
                {
                    "role":"user",
                    "content":user_prompt
                }
            ],
            response_format={
                "type":"json_object"
            },
            temperature=0.0,
            max_tokens=1000
        )
        content=response.choices[0].message.content
        if not content:
            raise RuntimeError(
                "Deepseek返回了空内容"
            )
        try:
            return AgentDecision.model_validate_json(content)
        except ValidationError as error:
            raise ValueError(
                "Deepseek返回的JSON"
                "不返回AgentDecision结构"
            ) from error
































































