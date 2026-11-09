import asyncio
from mini_agent.browser import BrowserSession
from mini_agent.config import Settings
from mini_agent.controller import BrowserController
from mini_agent.deepseek_client import DeepSeekClient
from mini_agent.schemas import (
    ActionResult,
    AgentStepRecord
)
from mini_agent.state import AgentState
from mini_agent.run_store import RunStore
from mini_agent.qwen_vision_client import QwenVisionClient


class BrowserAgent:
    def __init__(self,settings:Settings):
        self.setting=settings
        self.browser=BrowserSession(
            headless=settings.browser_headless
        )
        self.controller=BrowserController(self.browser)
        self.deepseek=DeepSeekClient(settings)
        self.vision=(
            QwenVisionClient(settings)
            if settings.qwen_api_key
            else None
        )
        self.state:AgentState|None=None

    @property
    def history(self)->list[AgentStepRecord]:
        if self.state is None:
            return []
        return self.state.history

    async def run(self,task:str)->ActionResult:
        task=task.strip()
        if not task:
            raise ValueError("任务不能为空")
        self.state=AgentState(task=task)
        self.state.mark_running()
        run_store=RunStore(task)
        previous_result:ActionResult|None=None
        try:
            await self.browser.start()
            for step_number in range(1,self.setting.max_steps+1):
                print(f"第{step_number}步")
                browser_state=await self.browser.get_state()
                run_store.save_screenshot(
                    step_number,
                    browser_state.screenshot_base64
                )
                visual_observation=None
                if(self.vision is not None and browser_state.screenshot_base64):
                    print("正在进行视觉观察...")
                    visual_observation=await self.vision.analyze(
                        task=task,
                        screenshot_base64=(browser_state.screenshot_base64)
                    )
                    print("视觉场景概述:",visual_observation.scene_summary)
                    print("视觉置信度:",visual_observation.confidence)
                run_store.save_visual_observation(
                    step_number,
                    visual_observation
                )
                print("当前网址:",browser_state.url)
                print("网页标题:",browser_state.title)
                print("可操作元素数量:",len(browser_state.elements))
                decision=await self.deepseek.decide(
                    task=task,
                    state=browser_state,
                    previous_result=previous_result,
                    visual_observation=visual_observation
                )
                print("页面观察:",decision.observation)
                print("下一目标:",decision.next_goal)
                print("模型动作:",decision.action.model_dump())
                result=await self.controller.execute(decision.action)
                print("执行成功:",result.success)
                print("执行结果:",result.message)
                record=AgentStepRecord(
                    step_number=step_number,
                    decision=decision,
                    result=result,
                )
                self.state.add_record(record)
                if result.is_done:
                    run_store.save_state(self.state)
                    return result
                stop_reason=self.state.check_stop_policy()
                if stop_reason:
                    self.state.mark_failed(stop_reason)
                    run_store.save_state(self.state)
                    return ActionResult(
                        success=False,
                        message=stop_reason,
                        is_done=True,
                        answer=stop_reason
                    )
                run_store.save_state(self.state)
                previous_result=result
                await asyncio.sleep(0.5)
            reason=f"达到最大步数{self.setting.max_steps}"
            self.state.mark_failed(reason)
            run_store.save_state(self.state)
            return ActionResult(
                success=False,
                message=(reason),
                is_done=True,
                answer=reason
            )
        finally:
            await self.browser.close()








































