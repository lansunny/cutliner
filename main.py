import asyncio

from mini_agent.agent import BrowserAgent
from mini_agent.config import Settings


async def main():
    settings = Settings.from_env()
    agent = BrowserAgent(settings)

    task = input("请输入浏览器任务：").strip()

    result = await agent.run(task)

    print("\n========== 任务结束 ==========")
    print("任务成功：", result.success)
    print("最终答案：", result.answer)
    print("总步骤数：", len(agent.history))


if __name__ == "__main__":
    asyncio.run(main())