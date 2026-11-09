"""Offline paired evaluation. Human labels are never exposed to the retrieval models."""

import asyncio
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mini_agent.config import Settings
from mini_agent.video.candidate_builder import ClipCandidateBuilder
from mini_agent.video.clip_exporter import VideoClipExporter
from mini_agent.video.clip_validator import ClipSelectionValidator
from mini_agent.video.frame_extractor import FrameExtractor
from mini_agent.video.quality_analyzer import FrameQualityAnalyzer
from mini_agent.video.reader import VideoReader
from mini_agent.video.review_agent import VideoReviewAgent
from mini_agent.video.schemas import (
    ClipChoice, ClipSelection, ShotSamplingResult, VideoFrame,
    VideoMetadata, VideoTaskSpec,
)
from mini_agent.video.shot_sampler import ShotFrameSampler
from mini_agent.video.vision_analyzer import VideoVisionAnalyzer


METHODS = ("uniform_2s", "scene_aware", "scene_react")


class EventInterval(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: float = Field(ge=0, allow_inf_nan=False)
    end: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def ordered(self):
        if self.end <= self.start:
            raise ValueError("事件结束时间必须晚于开始时间")
        return self


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    split: Literal["dev", "eval"]
    video_path: str | None = None
    query: str = Field(min_length=1)
    category: Literal["scene", "action", "brief", "negative"]
    annotation_status: Literal["pending", "labeled"] = "pending"
    target_exists: bool | None = None
    events: list[EventInterval] = Field(default_factory=list)
    annotator: str = ""
    notes: str = ""

    @model_validator(mode="after")
    def labels(self):
        if self.annotation_status == "labeled":
            missing = []
            if not self.video_path or not self.video_path.strip():
                missing.append("video_path（视频路径）")
            if not self.annotator.strip():
                missing.append("annotator（标注人）")
            if self.target_exists is None:
                missing.append("target_exists（有目标填true，无目标填false，不能留null）")
            if missing:
                raise ValueError(f"{self.case_id}已设为labeled，但缺少：{'、'.join(missing)}")
            if self.target_exists != bool(self.events):
                raise ValueError("有目标须标注全部事件区间；无目标须events为空")
            if (self.category == "negative") != (not self.target_exists):
                raise ValueError("negative类别须与无目标标签一致")
        ordered = sorted(self.events, key=lambda e: e.start)
        if any(a.end > b.start for a, b in zip(ordered, ordered[1:])):
            raise ValueError("事件标注区间不应重叠")
        return self


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_cases(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = [EvalCase.model_validate(item) for item in raw["cases"]]
    if len({c.case_id for c in cases}) != len(cases):
        raise ValueError("case_id重复")
    # A source video cannot be used for both tuning and held-out evaluation.
    splits = {}
    for case in cases:
        if case.video_path:
            source = str(Path(case.video_path).resolve()).casefold()
            if source in splits and splits[source] != case.split:
                raise ValueError("同一视频不能同时属于dev与eval")
            splits[source] = case.split
    return cases


def validate_sources(cases):
    metadata = {}
    for case in cases:
        if not case.video_path:
            continue
        source = str(Path(case.video_path).resolve())
        if source not in metadata:
            metadata[source] = VideoReader().read_metadata(source)
        if any(e.end > metadata[source].duration_seconds for e in case.events):
            raise ValueError(f"{case.case_id}标注超出视频时长；必须使用当前素材内相对秒数")
    return metadata


def fingerprint():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def prepare(dataset, output, split):
    all_cases = load_cases(dataset)
    cases = [c for c in all_cases if c.split == split and c.annotation_status == "labeled"]
    if not cases:
        raise ValueError(f"没有已人工标注的{split}任务；请先填写标注，不能把模型答案当作标签")
    metadata = validate_sources(cases)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save(output / "cases.json", {"cases": [c.model_dump(mode="json") for c in cases]})
    manifest = {
        "code_sha256": fingerprint(), "split": split, "videos": {},
        "protocol": "paired_offline_v1", "final_review_for_all_methods": True,
        "timing_note": "offline sequential evaluation; shared stage time attributed to each method; not production streaming latency",
    }
    for index, (source, meta) in enumerate(metadata.items(), 1):
        directory = output / f"video_{index:03d}"
        directory.mkdir()
        started = time.perf_counter()
        sampler = ShotFrameSampler()
        result = None
        for item in sampler.iter_prepare(meta, directory / "scene_frames"):
            if isinstance(item, ShotSamplingResult):
                result = item
        if result is None:
            raise RuntimeError("镜头采样未完成")
        scene_seconds = time.perf_counter() - started
        started = time.perf_counter()
        uniform = FrameExtractor().extract(source, directory / "uniform_frames", interval_seconds=2, max_frames=None)
        uniform = FrameQualityAnalyzer().analyze_many(uniform)
        shot_index = 0
        for frame in uniform:
            while shot_index + 1 < len(result.shots) and frame.timestamp_seconds >= result.shots[shot_index].end_seconds:
                shot_index += 1
            shot = result.shots[shot_index]
            frame.shot_id = shot.shot_id
            frame.shot_start_seconds = shot.start_seconds
            frame.shot_end_seconds = shot.end_seconds
        uniform_seconds = time.perf_counter() - started
        save(directory / "metadata.json", meta.model_dump(mode="json"))
        save(directory / "scene.json", [f.model_dump(mode="json") for f in result.frames])
        save(directory / "uniform.json", [f.model_dump(mode="json") for f in uniform])
        save(directory / "sampling.json", result.model_dump(mode="json"))
        stat = Path(source).stat()
        manifest["videos"][source] = {
            "directory": directory.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "scene_preparation_seconds": scene_seconds, "uniform_extraction_seconds": uniform_seconds,
            "preparation_timing_note": "both methods use common shot boundaries; common offline preparation time is charged equally",
        }
    save(output / "manifest.json", manifest)
    coverage(output)
    print("本地素材准备完成，没有调用模型：", output)


class ImageBudgetExceeded(RuntimeError):
    pass


class ApiMeter:
    def __init__(self, limit):
        self.limit = limit
        self.events = []

    def attach(self, owner, stage):
        client = owner.client.with_options(max_retries=0)

        async def create(**kwargs):
            images = sum(1 for msg in kwargs.get("messages", [])
                         if isinstance(msg.get("content"), list)
                         for part in msg["content"] if part.get("type") == "image_url")
            if sum(e["images"] for e in self.events) + images > self.limit:
                raise ImageBudgetExceeded("达到本次评测图片输入预算，停止新增请求")
            event = {"stage": stage, "model": kwargs.get("model"), "images": images, "status": "requested"}
            self.events.append(event)
            started = time.perf_counter()
            try:
                response = await client.chat.completions.create(**kwargs)
                event["status"] = "returned"
                event["usage"] = response.usage.model_dump() if response.usage else None
                return response
            except Exception as error:
                event["status"] = "error"
                event["error_type"] = type(error).__name__
                raise
            finally:
                event["seconds"] = time.perf_counter() - started
        owner.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=client.close)


def is_blocked(error):
    message = str(error).lower()
    return any(word in message for word in ("data_inspection_failed", "datainspectionfailed", "access_denied")) or getattr(error, "status_code", None) in (401, 403)


async def evaluate_method(task, metadata, evidence, method, directory, agent):
    candidates = ClipCandidateBuilder().build(task, metadata, evidence)
    candidates.sort(key=lambda c: (-c.relevance_score, c.start_seconds))
    save(directory / "candidates.json", [c.model_dump(mode="json") for c in candidates])
    attempts = []
    result = {"status": "no_match", "prediction": None, "attempts": attempts,
              "candidate_intervals": [[c.start_seconds, c.end_seconds] for c in candidates]}
    for index, candidate in enumerate(candidates, 1):
        folder = directory / f"attempt_{index:03d}"
        folder.mkdir()
        entry = {"candidate_id": candidate.candidate_id}
        attempts.append(entry)
        if method == "scene_react":
            local = await agent.review_interval(task, metadata.file_path, candidate.candidate_id,
                                                folder / "local", candidate.start_seconds, candidate.end_seconds)
            entry["local"] = local.model_dump(mode="json")
            if local.blocked_by_provider:
                result["status"] = "blocked"
                return result
            if local.verdict != "pass":
                continue
        choice = ClipSelection(choices=[ClipChoice(candidate_id=candidate.candidate_id, reason="统一候选排序")], summary="评测")
        checked = ClipSelectionValidator().validate(task, metadata, [candidate], choice)
        if not checked.valid:
            entry["constraint_errors"] = checked.errors
            continue
        exported = await asyncio.to_thread(VideoClipExporter().export, task, metadata, checked, folder / "clips")
        entry["exports"] = exported.model_dump(mode="json")
        save(folder / "exports.json", entry["exports"])
        if not exported.valid:
            result["status"] = "error"
            return result
        # Identical final semantic gate for all three methods.
        final = await agent.review_clip(task, exported.clips[0], folder / "final")
        entry["final"] = final.model_dump(mode="json")
        if final.blocked_by_provider:
            result["status"] = "blocked"
            return result
        if final.verdict == "pass":
            result.update(status="delivered", prediction={
                "start": candidate.start_seconds, "end": candidate.end_seconds,
                "file_path": str(exported.clips[0].file_path),
            })
            return result
    return result


async def run(prepared, max_images):
    prepared = Path(prepared)
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    if manifest["code_sha256"] != fingerprint():
        raise ValueError("代码已变化，请重新prepare，避免混用不同版本")
    cases = load_cases(prepared / "cases.json")
    for source, data in manifest["videos"].items():
        stat = Path(source).stat()
        if stat.st_size != data["size"] or stat.st_mtime_ns != data["mtime_ns"]:
            raise ValueError("视频已变化，请重新prepare")
    output = prepared / "evaluation"
    output.mkdir(exist_ok=False)
    settings = Settings.from_env()
    meter = ApiMeter(max_images)
    rows, judgments = [], []
    config = {"vlm": settings.qwen_vl_model, "planner": settings.deepseek_model,
              "max_images": max_images, "sdk_retries": 0, "methods": METHODS,
              "shared_scene_observations": True}
    save(output / "config.json", config)
    try:
        for case in cases:
            source = str(Path(case.video_path).resolve())
            info = manifest["videos"][source]
            video_dir = prepared / info["directory"]
            metadata = VideoMetadata.model_validate_json((video_dir / "metadata.json").read_text(encoding="utf-8"))
            task = VideoTaskSpec(query=case.query)
            caches = {}
            for method in METHODS:
                directory = output / f"{case.case_id}__{method}"
                directory.mkdir()
                kind = "uniform" if method == "uniform_2s" else "scene"
                frames = [VideoFrame.model_validate(f) for f in json.loads((video_dir / f"{kind}.json").read_text(encoding="utf-8"))]
                row = {"case_id": case.case_id, "method": method, "status": "error",
                       "prediction": None, "sample_times": [f.timestamp_seconds for f in frames]}
                rows.append(row)
                start_index = len(meter.events)
                started = time.perf_counter()
                try:
                    if kind not in caches:
                        observer = VideoVisionAnalyzer(settings)
                        meter.attach(observer, f"{case.case_id}:{kind}:coarse")
                        evidence = []
                        coarse_started = time.perf_counter()
                        try:
                            for offset in range(0, len(frames), 4):
                                evidence.extend(await observer.analyze_batch(task, frames[offset:offset + 4]))
                        finally:
                            await observer.close()
                        caches[kind] = (evidence, time.perf_counter() - coarse_started,
                                        list(meter.events[start_index:]))
                    evidence, coarse_seconds, coarse_events = caches[kind]
                    save(directory / "observations.json", [e.model_dump(mode="json") for e in evidence])
                    after_coarse = len(meter.events)
                    agent = VideoReviewAgent(settings)
                    meter.attach(agent.vision, f"{case.case_id}:{method}:review_vlm")
                    meter.attach(agent, f"{case.case_id}:{method}:review_decision")
                    review_started = time.perf_counter()
                    try:
                        row.update(await evaluate_method(task, metadata, evidence, method, directory, agent))
                    finally:
                        await agent.close()
                    row["attributed_api_events"] = coarse_events + meter.events[after_coarse:]
                    # Both get the same offline frame/shot preparation cost for a controlled comparison.
                    row["shared_offline_preparation_seconds"] = info["scene_preparation_seconds"] + info["uniform_extraction_seconds"]
                    row["coarse_observation_seconds"] = coarse_seconds
                    row["post_observation_seconds"] = time.perf_counter() - review_started
                    row["attributed_pipeline_seconds"] = (row["shared_offline_preparation_seconds"]
                                                           + coarse_seconds + row["post_observation_seconds"])
                    row["vlm_image_inputs"] = sum(e["images"] for e in row["attributed_api_events"])
                except Exception as error:
                    row["status"] = "budget_exceeded" if isinstance(error, ImageBudgetExceeded) else "blocked" if is_blocked(error) else "error"
                    row["error_type"] = type(error).__name__
                    if row["status"] in ("blocked", "budget_exceeded"):
                        print("评测停止：", row["status"], "。已保留部分结果，尚未完成的任务不会计为成功。")
                        return
                    raise
                finally:
                    row["incremental_wall_seconds"] = time.perf_counter() - started
                    row["incremental_image_inputs"] = sum(e["images"] for e in meter.events[start_index:])
                    save(directory / "result.json", row)
                judgments.append({"case_id": case.case_id, "method": method,
                                  "file_path": (row.get("prediction") or {}).get("file_path"),
                                  "correct": None, "reviewer": "", "notes": "人工观看后填写true/false，不能复制Agent判定"})
                if row["status"] == "blocked":
                    print("供应商审核拦截，已保存部分结果，停止本次评测。")
                    return
    finally:
        save(output / "results.json", rows)
        save(output / "api_calls.json", meter.events)
        save(output / "judgments.json", judgments)
        print("评测记录：", output)


def summarize(prepared):
    prepared = Path(prepared)
    cases = {c.case_id: c for c in load_cases(prepared / "cases.json")}
    output = prepared / "evaluation"
    rows = json.loads((output / "results.json").read_text(encoding="utf-8"))
    raw = json.loads((output / "judgments.json").read_text(encoding="utf-8"))
    if len({(j["case_id"], j["method"]) for j in raw}) != len(raw):
        raise ValueError("人工判定记录重复")
    judgments = {(j["case_id"], j["method"]): j for j in raw}
    summary = {}
    for method in METHODS:
        subset = [r for r in rows if r["method"] == method]
        finished = [r for r in subset if r["status"] in ("delivered", "no_match")]
        positives = [r for r in finished if cases[r["case_id"]].target_exists]
        negatives = [r for r in finished if not cases[r["case_id"]].target_exists]
        success, pending = 0, 0
        localization = []
        for row in positives:
            if row["status"] != "delivered":
                continue
            judgment = judgments.get((row["case_id"], method), {})
            correct = judgment.get("correct")
            if not isinstance(correct, bool) or not judgment.get("reviewer", "").strip():
                pending += 1
                continue
            success += int(correct)
            pred = row["prediction"]
            events = cases[row["case_id"]].events
            def iou(event):
                intersection = max(0, min(pred["end"], event.end) - max(pred["start"], event.start))
                return intersection / (pred["end"] - pred["start"] + event.end - event.start - intersection)
            best = max(events, key=iou)
            localization.append({"case_id": row["case_id"], "human_correct": correct, "temporal_iou": iou(best),
                                 "boundary_mae_seconds": (abs(pred["start"] - best.start) + abs(pred["end"] - best.end)) / 2})
        hits = total = 0
        for row in subset:
            for event in cases[row["case_id"]].events:
                total += 1
                hits += any(event.start <= t < event.end for t in row["sample_times"])
        images = [r["vlm_image_inputs"] for r in finished if "vlm_image_inputs" in r]
        durations = [r["attributed_pipeline_seconds"] for r in finished if "attributed_pipeline_seconds" in r]
        summary[method] = {
            "planned_tasks": len(cases), "finished_tasks": len(finished),
            "blocked_or_error_tasks": len(subset) - len(finished), "not_started_tasks": len(cases) - len(subset),
            "positive_tasks_finished": len(positives), "pending_human_judgments": pending,
            "first_delivery_success_rate": success / len(positives) if positives and not pending else None,
            "negative_tasks_finished": len(negatives),
            "negative_false_delivery_rate": sum(r["status"] == "delivered" for r in negatives) / len(negatives) if negatives else None,
            "event_sample_hit_rate": hits / total if total else None,
            "event_sample_hits": hits, "annotated_events_checked": total,
            "mean_vlm_image_inputs": sum(images) / len(images) if images else None,
            "mean_offline_attributed_seconds": sum(durations) / len(durations) if durations else None,
            "localization_diagnostics": localization,
        }
    # Only compare identical tasks completed by all methods.
    finished_ids = [{r["case_id"] for r in rows if r["method"] == method and r["status"] in ("delivered", "no_match")}
                    for method in METHODS]
    paired_ids = set.intersection(*finished_ids)
    paired = {"completed_case_ids": sorted(paired_ids), "count": len(paired_ids), "methods": {}}
    for method in METHODS:
        paired_rows = [r for r in rows if r["method"] == method and r["case_id"] in paired_ids]
        positive_rows = [r for r in paired_rows if cases[r["case_id"]].target_exists]
        success, pending = 0, 0
        for row in positive_rows:
            if row["status"] == "delivered":
                judgment = judgments.get((row["case_id"], method), {})
                if not isinstance(judgment.get("correct"), bool) or not judgment.get("reviewer", "").strip():
                    pending += 1
                else:
                    success += int(judgment["correct"])
        images = [r["vlm_image_inputs"] for r in paired_rows]
        paired["methods"][method] = {
            "first_delivery_success_rate": success / len(positive_rows) if positive_rows and not pending else None,
            "positive_tasks": len(positive_rows), "pending_human_judgments": pending,
            "mean_vlm_image_inputs": sum(images) / len(images) if images else None,
        }
    summary["paired_comparison"] = paired
    summary["notes"] = ["首个交付成功率依赖人工判定；受阻任务单列，不当作成功或悄悄删除。",
                        "帧命中率不是动作识别召回率；时间IoU仅辅助，不代替语义人工判定。",
                        "B/C共享粗观察，但成本比较各自计入一次共享粗观察；实际付费调用见api_calls.json。",
                        "存在未完成任务时，不可直接比较不同任务子集的平均值；需完整配对结果。",
                        "attributed_pipeline_seconds为离线分阶段时间合计，不是线上并发首结果延迟。"]
    save(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def coverage(prepared):
    """Compute complete annotated-event sampling coverage without using any model."""
    prepared = Path(prepared)
    cases = load_cases(prepared / "cases.json")
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for case in cases:
        source = str(Path(case.video_path).resolve())
        folder = prepared / manifest["videos"][source]["directory"]
        for kind in ("uniform", "scene"):
            frames = json.loads((folder / f"{kind}.json").read_text(encoding="utf-8"))
            times = sorted(f["timestamp_seconds"] for f in frames)
            duration = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))["duration_seconds"]
            boundaries = [0.0] + times + [duration]
            rows.append({"case_id": case.case_id, "sampling": kind, "frame_count": len(times),
                         "max_gap_seconds": max(b - a for a, b in zip(boundaries, boundaries[1:])),
                         "events": [{"start": event.start, "end": event.end,
                                     "sample_count": sum(event.start <= t < event.end for t in times)}
                                    for event in case.events]})
    save(prepared / "coverage.json", rows)
    print("本地采样覆盖记录：", prepared / "coverage.json")
    return rows
