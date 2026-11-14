"""Task-evidence routing and local neural retrieval, without VLM prescreening."""

import asyncio
import json
import subprocess
from pathlib import Path

from mini_agent.video.audio_visual_events import DEFAULT_PYTHON, PROJECT, LocalAudioVisualEventTool, canonical_events
from mini_agent.video.boundary_refiner import LocalBoundaryRefiner
from mini_agent.video.job_control import terminate_process
from mini_agent.video.schemas import ClipCandidate, ModelIntervalEvidence, VideoMetadata, VideoTaskSpec


def route_task(task: VideoTaskSpec):
    if task.audio_condition_type in ("spoken_content", "unsupported"):
        raise ValueError("具体对白、画外声和否定声音条件尚无对应本地模型；不能改成语音活动或纯视觉检索")
    if task.audio_condition_type == "none":
        if task.audio_event_classes:
            raise ValueError("纯视觉任务不能附加声音类别约束")
        route, classes = "visual", []
    else:
        classes = canonical_events(task.audio_event_classes)
        if not classes:
            raise ValueError("声画检索需要明确的受支持事件类别")
        route = "hybrid" if task.requires_extra_visual_grounding else "audio_visual"
    if route in ("visual", "hybrid") and not task.visual_query_en:
        raise ValueError("QD-DETR需要准确英文视觉查询，请通过任务解析或visual_query_en提供，不能静默使用中文替代")
    return route, classes


def audio_evidence_for_candidate(candidate: ClipCandidate, file_path: Path):
    events = []
    for item in candidate.model_evidence:
        if item.modality != "audio_visual":
            continue
        lo, hi = max(candidate.start_seconds, item.start_seconds), min(candidate.end_seconds, item.end_seconds)
        if lo < hi:
            events.append({"event_class": item.label, "source_start_seconds": item.start_seconds,
                           "source_end_seconds": item.end_seconds, "target_overlap_start_seconds": lo,
                           "target_overlap_end_seconds": hi, "mean_score": item.score})
    if not events:
        return []
    return [{"status": "ok", "source_file": str(file_path), "origin": "neural_retrieval",
             "events": events, "certifies_full_query": False, "absence_not_proven": True,
             "timestamp_resolution_seconds": 1.}]


class NeuralVideoRetriever:
    def __init__(self, python: Path = DEFAULT_PYTHON, top_k: int = 12, padding_seconds: float = 1., device="auto",
                 refine_visual_boundaries: bool = True):
        self.python, self.top_k, self.padding_seconds, self.device = Path(python), top_k, padding_seconds, device
        self.refine_visual_boundaries = refine_visual_boundaries

    async def retrieve(self, task: VideoTaskSpec, metadata: VideoMetadata, *, on_progress=None):
        route, classes = route_task(task)
        initial_stat = metadata.file_path.stat()
        signature = {"path": str(metadata.file_path), "size": initial_stat.st_size, "mtime_ns": initial_stat.st_mtime_ns}
        av_tool = LocalAudioVisualEventTool.from_env() if classes else None
        if classes and av_tool is None:
            raise ValueError("声画模型未配置，不能自动降级成纯视觉或全视频VLM扫描")
        if not self.python.is_file():
            raise FileNotFoundError(f"本地模型Python环境不存在：{self.python}")
        request = {"route": route, "video": str(metadata.file_path), "start_seconds": 0.,
                   "duration_seconds": metadata.duration_seconds, "visual_query_en": task.visual_query_en,
                   "audio_event_classes": classes, "audio_match_mode": task.audio_match_mode,
                   "av_run": str(av_tool.run_dir) if av_tool else None, "top_k": self.top_k,
                   "padding_seconds": self.padding_seconds, "device": self.device}
        command = PROJECT/"experiments/neural_retrieval/retrieve_worker.py"
        process = await asyncio.create_subprocess_exec(str(self.python), "-X", "utf8", str(command),
            cwd=str(command.parent), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

        async def progress():
            recent = []
            while line := await process.stderr.readline():
                text = line.decode("utf-8", errors="replace").rstrip()
                print(f"[本地模型] {text}", flush=True)
                recent.append(text)
                recent = recent[-12:]
            return recent

        monitor = asyncio.create_task(progress())
        try:
            process.stdin.write((json.dumps(request, ensure_ascii=False)+"\n").encode("utf-8"))
            await process.stdin.drain()
            process.stdin.close()
            stdout = await asyncio.wait_for(process.stdout.read(), timeout=1800)
            await process.wait()
            recent = await monitor
            if process.returncode:
                raise RuntimeError("本地神经检索失败："+"\n".join(recent))
            result = json.loads(stdout.decode("utf-8"))
            for index, item in enumerate(result["predictions"], 1):
                item["candidate_id"] = f"neural_{index:04d}"
            if route == "visual" and self.refine_visual_boundaries and result["predictions"]:
                refiner = LocalBoundaryRefiner.from_env()
                if refiner is not None:
                    if on_progress is not None:
                        on_progress({"stage_id": "refinement", "kind": "action", "message": "读取候选附近画面，修正边界并重排"})
                    result["coarse_predictions"] = result["predictions"]
                    result["predictions"], result["details"]["boundary_refinement"] = await refiner.refine(
                        task, metadata, result["predictions"], device=self.device)
                    if on_progress is not None:
                        count = len(result["details"]["boundary_refinement"]["refinements"])
                        on_progress({"stage_id": "refinement", "kind": "result", "message": f"已完成 {count} 个候选的边界精修与重排"})
                else:
                    result["details"]["boundary_refinement"] = {"status": "disabled_explicitly"}
            else:
                result["details"]["boundary_refinement"] = {"status": "skipped", "reason":
                    "audio_constraints_preserved" if route != "visual" else "disabled_or_no_candidates"}
            candidates = []
            source = {"visual": "qd_detr", "audio_visual": "av_event", "hybrid": "hybrid_neural"}[route]
            for item in result["predictions"]:
                lo, hi = max(0., item["start_seconds"]), min(metadata.duration_seconds, item["end_seconds"])
                if lo < hi:
                    candidates.append(ClipCandidate(candidate_id=item["candidate_id"], start_seconds=lo, end_seconds=hi,
                        evidence_source=source, model_evidence=[ModelIntervalEvidence.model_validate(e) for e in item["model_evidence"]],
                        relevance_score=item["score"], retrieval_notes=item.get("retrieval_notes", []),
                        review_priority=item.get("review_priority"), boundary_refinement=item.get("boundary_refinement"),
                        boundary_note=item.get("boundary_note", "深度学习模型预测区间，非逐帧语义保证；需Agent审核完整条件")))
            final_stat = metadata.file_path.stat()
            if (final_stat.st_size, final_stat.st_mtime_ns) != (initial_stat.st_size, initial_stat.st_mtime_ns):
                raise ValueError("源视频在粗检索与边界精修期间被修改，不能复用不同版本的证据")
            result["source_signature"] = signature
            return candidates, result
        finally:
            if process.returncode is None:
                await terminate_process(process, stderr_owned=True)
            if not monitor.done():
                await monitor
