"""Local neural sound-and-visible-source evidence; never certifies a whole user query."""

import asyncio
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
EXPERIMENT = PROJECT/"experiments/av_grounding_local"
DEFAULT_PYTHON = PROJECT/"experiments/video_grounding_local/.venv/Scripts/python.exe"
EVENT_NAMES = {
    "Accordion": "手风琴演奏", "Acoustic guitar": "原声吉他演奏", "Baby cry, infant cry": "婴儿哭声",
    "Banjo": "班卓琴演奏", "Bark": "狗叫", "Bus": "公交车行驶", "Cat": "猫叫",
    "Chainsaw": "电锯作业", "Church bell": "教堂钟声", "Clock": "时钟走动",
    "Female speech, woman speaking": "女性讲话", "Fixed-wing aircraft, airplane": "飞机飞行",
    "Flute": "长笛演奏", "Frying (food)": "煎炸食物", "Goat": "山羊叫", "Helicopter": "直升机飞行",
    "Horse": "马嘶鸣", "Male speech, man speaking": "男性讲话", "Mandolin": "曼陀林演奏",
    "Motorcycle": "摩托车行驶", "Race car, auto racing": "赛车行驶", "Rodents, rats, mice": "啮齿动物叫声",
    "Shofar": "羊角号演奏", "Toilet flush": "马桶冲水", "Train horn": "火车鸣笛",
    "Truck": "卡车行驶", "Ukulele": "尤克里里演奏", "Violin, fiddle": "小提琴演奏",
}
ALIASES = {**{name: name for name in EVENT_NAMES}, **{zh: name for name, zh in EVENT_NAMES.items()}}


def canonical_events(values):
    # These are explicit tool arguments, not free-text query parsing. Do not
    # strip negation, person identities, qualifiers, or dialogue-content clauses.
    unsupported = [value for value in values if value not in ALIASES]
    if unsupported:
        raise ValueError(f"Unsupported audio-visual event classes: {unsupported}")
    return list(dict.fromkeys(ALIASES[value] for value in values))


class LocalAudioVisualEventTool:
    def __init__(self, run_dir: Path, python: Path = DEFAULT_PYTHON):
        self.run_dir, self.python = Path(run_dir).resolve(), Path(python).resolve()

    @classmethod
    def from_env(cls):
        configured = os.getenv("LOCAL_AV_EVENT_RUN")
        if configured is not None:
            folder = configured.strip()
            if not folder or folder.lower() in ("off", "0"):
                return None
        else:
            release = EXPERIMENT/"approved_event_model.json"
            if not release.is_file():
                return None
            manifest = json.loads(release.read_text(encoding="utf-8"))
            folder = str(EXPERIMENT/manifest["run_dir"])
            result = Path(folder)/"heldout_results.json"
            if not result.is_file() or hashlib.sha256(result.read_bytes()).hexdigest() != manifest["heldout_results_sha256"]:
                return None
        return cls(Path(folder), Path(os.getenv("LOCAL_AV_EVENT_PYTHON", str(DEFAULT_PYTHON))))

    @property
    def catalog(self):
        return [{"event_class": key, "label_zh": value} for key, value in EVENT_NAMES.items()]

    async def inspect(self, video: Path, start: float, end: float, event_classes: list[str]):
        base = {"certifies_full_query": False, "auxiliary_evidence_only": True,
                "scope": "28类有声且可见的事件活动，不识别具体台词、身份、否定条件或任意动作过程"}
        if not all(math.isfinite(x) for x in (start, end)) or not 0 <= start < end or end-start > 10.000001:
            return {**base, "status": "unsupported", "reason": "requires_source_interval_of_at_most_10_seconds"}
        try:
            classes = canonical_events(event_classes)
        except ValueError:
            return {**base, "status": "unsupported", "reason": "outside_explicit_event_taxonomy"}
        if not 1 <= len(classes) <= 4:
            return {**base, "status": "unsupported", "reason": "request_one_to_four_event_classes"}
        if not self.python.is_file() or not (self.run_dir/"heldout_results.json").is_file():
            return {**base, "status": "unavailable", "reason": "validated_local_model_not_configured"}
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                str(self.python), "-X", "utf8", str(EXPERIMENT/"predict_events.py"), "--run", str(self.run_dir),
                "--video", str(Path(video).resolve()), "--start", str(start), "--end", str(end),
                "--classes-json", json.dumps(classes, ensure_ascii=False), cwd=str(EXPERIMENT),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            stdout, _stderr = await asyncio.wait_for(process.communicate(), timeout=180)
            if process.returncode:
                return {**base, "status": "unavailable", "reason": "local_event_model_failed", "exit_code": process.returncode}
            result = json.loads(stdout.decode("utf-8"))
            if result.get("certifies_full_query") is not False:
                raise ValueError("A neural subcondition tool cannot certify an entire query")
            return {**base, **result}
        except (OSError, ValueError, asyncio.TimeoutError):
            return {**base, "status": "unavailable", "reason": "local_event_evidence_unavailable"}
        finally:
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()
