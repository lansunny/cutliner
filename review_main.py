"""Review previously exported clips without rescanning or exporting the source video."""

import asyncio
import json
from pathlib import Path

from mini_agent.video.review_agent import review_exports
from mini_agent.video.schemas import VideoExportReport, VideoTaskSpec


async def main():
    source = Path(input("请输入已有视频运行目录（包含task.json和exports.json）：").strip().strip('"')).resolve()
    task = VideoTaskSpec.model_validate(json.loads((source / "task.json").read_text(encoding="utf-8"))["task"])
    exports = VideoExportReport.model_validate_json((source / "exports.json").read_text(encoding="utf-8"))
    print("本次只复核已导出片段，将调用Qwen视觉模型和DeepSeek复核决策。")
    await review_exports(task, exports, source)


if __name__ == "__main__":
    asyncio.run(main())
