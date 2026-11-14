"""Thin application adapter for the separately installed local boundary network."""

import asyncio
import json
import math
import os
from pathlib import Path
import subprocess

from mini_agent.video.audio_visual_events import DEFAULT_PYTHON, PROJECT
from mini_agent.video.job_control import terminate_process


EXPERIMENT = PROJECT / "experiments/boundary_refiner_local"


def merge_refinements(predictions, result, duration):
    """Keep coarse evidence and scores; make neural review priority explicit."""
    if result.get("status") != "ok" or result.get("certifies_full_query") is not False:
        raise ValueError("无效的边界网络输出")
    updates = result["refinements"]
    indices = [row["coarse_index"] for row in updates]
    if len(indices) != len(set(indices)) or any(not 1 <= i <= min(3, len(predictions)) for i in indices):
        raise ValueError("边界网络返回了重复或未知候选")
    ordered = []
    for update in sorted(updates, key=lambda row: (-row["selection_logit"], row["coarse_index"])):
        index = update["coarse_index"]
        original = predictions[index - 1]
        start, end = update["start_seconds"], update["end_seconds"]
        evidence = update["boundary_refinement"]
        lower, upper = evidence["observation_start_seconds"], evidence["observation_end_seconds"]
        numbers = (start, end, lower, upper, update["selection_logit"], update["predicted_iou"])
        if not all(math.isfinite(x) for x in numbers) or not 0 <= lower <= start < end <= upper <= duration:
            raise ValueError("边界网络输出超出实际观察范围")
        if (evidence["coarse_start_seconds"], evidence["coarse_end_seconds"]) != (original["start_seconds"], original["end_seconds"]):
            raise ValueError("边界修正与原始候选不一致")
        if (evidence["proposed_start_seconds"], evidence["proposed_end_seconds"], evidence["coarse_rank"]) != (start, end, index):
            raise ValueError("边界预测与留存证据不一致")
        if not 0 <= update["predicted_iou"] <= 1:
            raise ValueError("非法的模型质量分数")
        ordered.append({**original, "start_seconds": start, "end_seconds": end,
            "boundary_refinement": evidence,
            "model_evidence": [*original["model_evidence"], {
                "modality": "visual", "model_name": result["model_name"], "label": result["query"],
                "start_seconds": start, "end_seconds": end, "score": update["predicted_iou"],
                "score_is_calibrated_probability": False}],
            "boundary_note": "查询条件局部网络提出的边界，保留粗候选证据；需Agent重新审核实际区间"})
    # Candidates outside the trained Top-3 budget remain available as coarse
    # fallbacks, after the refined group. Scores from different models are not mixed.
    ordered.extend(dict(item) for i, item in enumerate(predictions, 1) if i not in indices)
    for priority, item in enumerate(ordered, 1):
        item["review_priority"] = priority
    return ordered


class LocalBoundaryRefiner:
    def __init__(self, manifest: Path, python: Path = DEFAULT_PYTHON):
        self.manifest, self.python = Path(manifest), Path(python)

    @classmethod
    def from_env(cls):
        configured = os.getenv("LOCAL_BOUNDARY_MODEL", "").strip()
        if configured.lower() in ("off", "0"):
            return None
        return cls(Path(configured) if configured else EXPERIMENT / "approved_boundary_model.json",
                   Path(os.getenv("LOCAL_BOUNDARY_PYTHON", str(DEFAULT_PYTHON))))

    async def refine(self, task, metadata, predictions, *, device="auto"):
        if task.audio_condition_type != "none" or task.audio_event_classes:
            raise ValueError("视觉边界网络不能静默修改声音约束区间")
        if not self.manifest.is_file() or not self.python.is_file():
            raise FileNotFoundError("本地边界网络配置或Python环境不存在；可设置LOCAL_BOUNDARY_MODEL=off显式禁用")
        request = {"manifest": str(self.manifest.resolve()), "video": str(metadata.file_path),
                   "duration_seconds": metadata.duration_seconds, "query": task.visual_query_en,
                   "candidates": [{"start_seconds": c["start_seconds"], "end_seconds": c["end_seconds"],
                                   "score": c["score"]} for c in predictions[:3]], "device": device}
        artifact_root = os.getenv("LOCAL_BOUNDARY_ARTIFACT_ROOT")
        if artifact_root:
            request["artifact_root"] = artifact_root
        process = await asyncio.create_subprocess_exec(str(self.python), "-X", "utf8",
            str(EXPERIMENT / "predict_boundary.py"), cwd=str(EXPERIMENT), stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

        async def progress():
            recent = []
            while line := await process.stderr.readline():
                message = line.decode("utf-8", errors="replace").rstrip()
                print(f"[边界网络] {message}", flush=True)
                recent.append(message)
                recent = recent[-12:]
            return recent

        monitor = asyncio.create_task(progress())
        try:
            process.stdin.write((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))
            await process.stdin.drain()
            process.stdin.close()
            stdout = await asyncio.wait_for(process.stdout.read(), timeout=1800)
            await process.wait()
            recent = await monitor
            if process.returncode:
                raise RuntimeError("本地边界网络执行失败：" + "\n".join(recent))
            result = json.loads(stdout.decode("utf-8"))
            return merge_refinements(predictions, result, metadata.duration_seconds), result
        finally:
            if process.returncode is None:
                await terminate_process(process, stderr_owned=True)
            if not monitor.done():
                await monitor
