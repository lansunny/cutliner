"""Deep-learning retrieval first; only candidate intervals are sent to the review Agent."""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from mini_agent.config import PROJECT_ROOT, Settings
from mini_agent.video.audio_visual_events import canonical_events
from mini_agent.video.local_search import deliver_first_match
from mini_agent.video.neural_retriever import NeuralVideoRetriever, route_task
from mini_agent.video.reader import VideoReader
from mini_agent.video.schemas import VideoTaskSpec
from mini_agent.video.task_planner import VideoTaskPlanner


def save_json(run_dir: Path, filename: str, data: object):
    (run_dir/filename).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--request")
    parser.add_argument("--task-json", type=Path, help="Explicit structured task; avoids the text planning API")
    parser.add_argument("--visual-query-en", help="Explicit English visual query; no translation API")
    parser.add_argument("--events", nargs="+", help="Explicit positive sounding-source event classes or exact Chinese aliases")
    parser.add_argument("--extra-visual", action="store_true", help="Use both QD-DETR and AV event models")
    parser.add_argument("--audio-match-mode", choices=["all", "any"], default="all")
    parser.add_argument("--retrieve-only", action="store_true", help="Generate candidates without paid VLM/Agent review")
    parser.add_argument("--top-k", type=int, default=12, help="Candidate review budget, not the requested number of output clips")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-boundary-refine", action="store_true", help="显式禁用纯视觉候选的局部边界网络")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    source = args.video or Path(input("请输入视频路径，或已有运行目录：").strip().strip('"'))
    source = source.resolve()
    previous, previous_request = None, None
    if source.is_dir():
        previous = source
        stored = json.loads((source/"metadata.json").read_text(encoding="utf-8"))
        source = Path(stored["file_path"]).resolve()
        if (previous/"task.json").is_file():
            item = json.loads((previous/"task.json").read_text(encoding="utf-8"))
            previous_request = item.get("request") or item.get("task", {}).get("query")
        print("复用源视频与原需求，按新模型流程重新检索；不复用旧VLM视觉粗筛。")
    metadata = await asyncio.to_thread(VideoReader().read_metadata, source)
    request = args.request or previous_request
    task = None
    if args.task_json:
        item = json.loads(args.task_json.read_text(encoding="utf-8"))
        task = VideoTaskSpec.model_validate(item.get("task", item))
        request = request or item.get("request") or task.query
    else:
        request = request or input("请输入视频检索需求：").strip()
        if not request:
            raise ValueError("视频检索需求不能为空")
        if args.events:
            task = VideoTaskSpec(query=request, audio_condition_type="event_activity",
                audio_event_classes=canonical_events(args.events), visual_query_en=args.visual_query_en,
                requires_extra_visual_grounding=args.extra_visual, audio_match_mode=args.audio_match_mode)
        elif args.visual_query_en:
            task = VideoTaskSpec(query=request, visual_query_en=args.visual_query_en)
    run_dir = PROJECT_ROOT/"runs"/f"neural_video_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    save_json(run_dir, "metadata.json", metadata.model_dump(mode="json"))
    if previous:
        save_json(run_dir, "source_run.json", {"source_run": str(previous), "old_visual_observations_reused": False})
    if task is None:
        print("正在解析文字需求与模型路由；此步不观察视频画面。")
        planner = VideoTaskPlanner(Settings.from_env())
        try:
            task = await planner.parse_task(request, metadata)
        finally:
            await planner.close()
    save_json(run_dir, "task.json", {"request": request, "task": task.model_dump(mode="json")})
    try:
        route, _classes = route_task(task)
    except ValueError as error:
        save_json(run_dir, "retrieval.json", {"status": "unsupported_task", "reason": str(error), "vlm_prescreen_calls": 0})
        print("当前模型无法执行该需求：", error)
        print("结果目录：", run_dir)
        return
    print("模型路由：", {"visual": "QD-DETR视觉定位", "audio_visual": "本地声画事件定位", "hybrid": "QD-DETR＋声画事件定位"}[route])
    print("检索阶段不调用Qwen视觉粗筛；只在候选产生后由Agent审核。")
    save_json(run_dir, "retrieval_state.json", {"status": "running", "route": route})
    try:
        candidates, report = await NeuralVideoRetriever(top_k=args.top_k, device=args.device,
            refine_visual_boundaries=not args.no_boundary_refine).retrieve(task, metadata)
    except Exception as error:
        save_json(run_dir, "retrieval_state.json", {"status": "failed", "route": route, "reason": str(error)})
        raise
    save_json(run_dir, "candidates.json", [c.model_dump(mode="json") for c in candidates])
    save_json(run_dir, "retrieval.json", report)
    save_json(run_dir, "retrieval_state.json", {"status": "complete", "candidate_count": len(candidates), "route": route})
    print(f"模型检索完成：{len(candidates)}个候选。分数不是完整需求成立的概率。")
    print("结果目录：", run_dir)
    if args.retrieve_only:
        print("仅检索模式结束：未调用候选视觉复核、Agent决策或视频导出。")
        return
    if not candidates:
        print("模型未提出候选，不等于证明视频里没有目标；本次不调用复核API。")
        return
    settings = Settings.from_env()
    if not settings.qwen_api_key:
        raise ValueError("模型候选已保存；候选复核需要DASHSCOPE_API_KEY，或使用--retrieve-only仅运行本地检索")
    await deliver_first_match(task, metadata, candidates, run_dir, use_legacy_action_verifier=False,
                              allow_boundary_repair=True)


if __name__ == "__main__":
    asyncio.run(main())
