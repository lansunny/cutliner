
from typing import Annotated,Literal
from pydantic import BaseModel,ConfigDict,Field

class StrictModel(BaseModel):
    model_config=ConfigDict(extra="forbid")

class GoToAction(StrictModel):
    action:Literal["goto"]
    url:str=Field(min_length=1)

class ClickAction(StrictModel):
    action:Literal["click"]
    index:int=Field(ge=0)

class TypeTextAction(StrictModel):
    action:Literal["type"]
    index:int=Field(ge=0)
    text:str=Field(min_length=1)

class ScrollAction(StrictModel):
    action:Literal["scroll"]
    direction:Literal["up","down"]
    amount:int=Field(default=600,ge=100,le=2000)

class DoneAction(StrictModel):
    action:Literal["done"]
    success:bool
    answer:str=Field(min_length=1)

BrowserAction=Annotated[
    GoToAction
    | ClickAction
    | TypeTextAction
    | ScrollAction
    | DoneAction,
    Field(discriminator="action")
]

class AgentDecision(StrictModel):
    observation:str=Field(
        min_length=1,
        max_length=500,
        description="对当前网页状态的简短总结",
    )
    next_goal:str=Field(
        min_length=1,
        max_length=300,
        description="这一轮准备完成的目标",
    )
    action:BrowserAction

class PageElement(StrictModel):
    index:int =Field(ge=0)
    tag:str
    text:str=""
    element_type:str|None=None
    placeholder:str|None=None
    aria_label:str|None=None

    def to_prompt_line(self) -> str:
        attributes = []

        if self.text:
            attributes.append(f'text="{self.text}"')

        if self.element_type:
            attributes.append(
                f'type="{self.element_type}"'
            )

        if self.placeholder:
            attributes.append(
                f'placeholder="{self.placeholder}"'
            )

        if self.aria_label:
            attributes.append(
                f'aria-label="{self.aria_label}"'
            )

        attribute_text = " ".join(attributes)

        if not attribute_text:
            attribute_text = "无可读属性"

        return (
            f"[{self.index}] "
            f"<{self.tag}> "
            f"{attribute_text}"
        )

class BrowserState(StrictModel):
    url:str
    title:str
    visible_text:str
    elements:list[PageElement]
    screenshot_base64:str|None=None

    def to_prompt_text(self) -> str:
        element_lines = [
            element.to_prompt_line()
            for element in self.elements
        ]

        elements_text = "\n".join(element_lines)
        if not elements_text:
            elements_text = "当前没有可操作元素"
        return (
            f"当前网址：{self.url}\n"
            f"网页标题：{self.title}\n\n"
            f"当前页面文字：\n"
            f"{self.visible_text}\n\n"
            f"可以操作的元素：\n"
            f"{elements_text}"
        )

class ActionResult(StrictModel):
    success:bool
    message:str
    is_done:bool=False
    answer:str|None=None


class AgentStepRecord(StrictModel):
    step_number:int=Field(ge=1)
    decision:AgentDecision
    result:ActionResult

class VisualObservation(StrictModel):
    scene_summary:str=Field(
        min_length=1,
        max_length=1000
    )
    visible_text:list[str]=Field(
        default_factory=list
    )
    key_elements:list[str]=Field(
        default_factory=list
    )
    spatial_relations:list[str]=Field(
        default_factory=list
    )
    confidence:float=Field(
        ge=0.0,
        le=1.0
    )
    def to_prompt_text(self)->str:
        visible_text="\n".join(
            f"- {text}"
            for text in self.visible_text
        )
        key_elements="\n".join(
            f"- {element}"
            for element in self.key_elements
        )
        spatial_relations="\n".join(
            f"- {relation}"
            for relation in self.spatial_relations
        )
        return(
            f"画面概述:{self.scene_summary}\n\n"
            f"可见文字:\n{visible_text or '无'}\n\n"
            f"关键元素:\n{key_elements or '无'}\n\n"
            f"位置关系:\n{spatial_relations or '无'}\n\n"
            f"视觉置信度:{self.confidence}"
        )























































