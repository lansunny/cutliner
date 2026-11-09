from urllib.parse import urlparse

from mini_agent.browser import BrowserSession
from mini_agent.schemas import (
    ActionResult,
    BrowserAction,
    ClickAction,
    DoneAction,
    GoToAction,
    ScrollAction,
    TypeTextAction,
)

class BrowserController:
    def __init__(self,browser:BrowserSession):
        self.browser=browser

    async def execute(self,action:BrowserAction,)->ActionResult:
        try :
            if isinstance(action,GoToAction):
                return await self._goto(action)
            if isinstance(action,ClickAction):
                return await self._click(action)
            if isinstance(action,TypeTextAction):
                return await self._type_text(action)
            if isinstance(action,ScrollAction):
                return await self._scroll(action)
            if isinstance(action,DoneAction):
                return self._done(action)

            return ActionResult(
                success=False,
                message=f"不支持的动作类型：{type(action).__name__}",
            )
        except Exception as error:
            return ActionResult(
                success=False,
                message=(
                    f"{type(error).__name__}:{error}"
                ),
            )

    async def _goto(self,action:GoToAction)->ActionResult:
        parsed_url=urlparse(action.url)
        if parsed_url.scheme not in {"http","https"}:
            return ActionResult(
                success=False,
                message="只允许访问http或https地址"
            )
        await self.browser.goto(action.url)
        return ActionResult(
            success=True,
            message=f"已打开网页:{action.url}"
        )
    async def _click(self,action:ClickAction)->ActionResult:
        selector=(
            f'[data-agent-index="{action.index}"]'
        )
        locator=self.browser.page.locator(selector)
        if await locator.count()==0:
            return ActionResult(
                success=False,
                message=f"找不到编号为{action.index}的元素",
            )
        await locator.first.click(timeout=5000)
        return ActionResult(
            success=True,
            message=f"已点击编号为{action.index}的元素"
        )
    async def _type_text(self,action:TypeTextAction)->ActionResult:
        selector=(
            f'[data-agent-index="{action.index}"]'
        )
        locator=self.browser.page.locator(selector)
        if await locator.count()==0:
            return ActionResult(
                success=False,
                message=f"找不到编号为{action.index}的元素"
            )
        await locator.first.fill(
            action.text,
            timeout=5000
        )
        return ActionResult(
            success=True,
            message=(
                f"已向编号为{action.index}"
                f"的元素输入文本"
            )
        )
    async def _scroll(self,action:ScrollAction)->ActionResult:
        delta=action.amount
        if action.direction=="up":
            delta=-delta
        await self.browser.page.mouse.wheel(
            0,
            delta
        )
        return ActionResult(
            success=True,
            message=(
                f"已向{action.direction}滚动"
                f"{action.amount}像素"
            )
        )

    def _done(self,action:DoneAction)->ActionResult:
        return ActionResult(
            success=action.success,
            message="任务执行结束",
            is_done=True,
            answer=action.answer
        )
































































































