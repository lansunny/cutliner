import os
from dataclasses import dataclass,field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT=Path(__file__).resolve().parent.parent
ENV_FILE=PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE)

@dataclass(frozen=True,slots=True)
class Settings:
    deepseek_api_key:str=field(repr=False)
    deepseek_base_url:str
    deepseek_model:str
    qwen_api_key:str=field(repr=False)
    qwen_base_url:str
    qwen_vl_model:str
    max_steps:int
    browser_headless:bool

    @classmethod
    def from_env(cls)->"Settings":
        api_key=os.getenv(
            "DEEPSEEK_API_KEY",
            "",
        ).strip()
        if not api_key:
            raise ValueError(
                "没有找到DEEPSEEK_API_KEY"
                "请检查项目根目录的.env"
            )
        max_steps_text=os.getenv(
            "MAX_STEPS",
            "10"
        )
        try:
            max_steps=int(max_steps_text)
        except ValueError as error:
            raise ValueError(
                "MAX_STEPS必须是整数"
            ) from error
        if max_steps<1:
            raise ValueError(
                "MAXSTEPS必须大于0"
            )
        headless_text=os.getenv(
            "BROWSER_HEADLESS",
            "false"
        ).lower()
        browser_headless=headless_text in {
            "1",
            "true",
            "yes",
            "on",
        }
        return cls(
            deepseek_api_key=api_key,
            deepseek_base_url=os.getenv(
                "DEEPSEEK_BASE_URL",
                "https://api.deepseek.com",
            ),
            deepseek_model=os.getenv(
                "DEEPSEEK_MODEL",
                "deepseek-chat"
            ),
            qwen_api_key=os.getenv(
                "DASHSCOPE_API_KEY",
                ""
            ).strip(),
            qwen_base_url=os.getenv(
                "QWEN_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1"
            ).strip(),
            qwen_vl_model=os.getenv(
                "QWEN_VL_MODEL",
                "qwen3-vl-flash"
            ).strip(),
            max_steps=max_steps,
            browser_headless=browser_headless,

        )







































































