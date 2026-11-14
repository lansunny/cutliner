"""Real frames/FFmpeg with MOCK semantic decisions; no paid API or accuracy claim."""

import argparse
import asyncio
import json
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch
from uuid import uuid4

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

import studio_server as studio
import mini_agent.video.local_search as search
from mini_agent.video.boundary_repair import apply_boundary_proposal
from mini_agent.video.frame_extractor import FrameExtractor
from mini_agent.video.review_agent import ClipReview, ReviewAction, VideoReviewAgent
from mini_agent.video.schemas import ClipCandidate, VideoFrameEvidence, VideoFrameObservation, VideoTaskSpec
from mini_agent.video.studio_history import read_json, save_json, source_signature
from studio_jobs import StudioJobs


class MockVision:
    async def analyze_batch(self, task, frames):
        return [VideoFrameEvidence(frame=frame, observation=VideoFrameObservation(
            frame_index=frame.frame_index, description="MOCK_ONLY: real decoded frame, synthetic semantic label",
            matches_query=True, relevance_score=.9, reason="MOCK_ONLY, not an accuracy evaluation")) for frame in frames]


async def observed_proposal(task, metadata, candidate, folder, *, misuse_context=False):
    # Avoid constructing clients, reading API keys or invoking any remote service.
    reviewer = VideoReviewAgent.__new__(VideoReviewAgent)
    reviewer.vision, reviewer.extractor, reviewer.audio_event_tool = MockVision(), FrameExtractor(), None
    reviewer.on_progress, reviewer.max_rounds, reviewer.max_frames = None, 3, 12
    decisions = []

    async def decide(task, duration, evidence, remaining, can_inspect, feedback, *args, boundary_context=None):
        if not decisions:
            result = ReviewAction(action="inspect_context", reason="MOCK_ONLY inspect observed source context",
                                  timestamps_seconds=[-1.5, duration + 1.5])
        else:
            context = boundary_context["context_observations"]
            assert len(context) == 2 and boundary_context["enabled"]
            ids = [row["frame_index"] for row in context]
            result = (ReviewAction(action="finish", reason="MOCK_ONLY invalid pass using outside evidence",
                                  verdict="pass", evidence_frame_indices=ids) if misuse_context else
                      ReviewAction(action="propose_boundary", reason="MOCK_ONLY request expansion based on actual observed frames",
                                   boundary_start_frame_index=ids[0], boundary_end_frame_index=ids[-1]))
        decisions.append(result)
        return result

    reviewer.decide = decide
    result = await reviewer.review_interval(task, metadata.file_path, candidate.candidate_id, folder,
        candidate.start_seconds, candidate.end_seconds, allow_boundary_repair=True)
    if misuse_context:
        assert result.verdict == "uncertain" and result.boundary_proposal is None
    else:
        assert result.verdict == "uncertain" and result.boundary_proposal is not None and len(decisions) == 2
        assert result.sampled_frames == 6 and len(list((folder / "frames").glob("*.jpg"))) == 6
    return result


def main(args):
    folder = PROJECT / "runs" / f"repair_check_{uuid4().hex[:12]}"
    folder.mkdir()
    originals = {name: (args.run / name).read_bytes() for name in
                 ("metadata.json", "task.json", "candidates.json", "retrieval.json", "retrieval_state.json")}
    metadata = studio.VideoReader().read_metadata(read_json(args.run / "metadata.json")["file_path"])
    source_task = read_json(args.run / "task.json")
    task = VideoTaskSpec.model_validate(source_task.get("task", source_task))
    candidate = ClipCandidate.model_validate(json.loads(originals["candidates.json"])[0])
    initial = asyncio.run(observed_proposal(task, metadata, candidate, folder / "real_frame_observation"))
    asyncio.run(observed_proposal(task, metadata, candidate, folder / "context_not_delivery_evidence", misuse_context=True))
    proposal = initial.boundary_proposal
    effective = apply_boundary_proposal(candidate, proposal, metadata)
    assert effective.start_seconds < candidate.start_seconds and effective.end_seconds > candidate.end_seconds
    assert effective.model_evidence == candidate.model_evidence and effective.boundary_refinement == candidate.boundary_refinement
    for invalid in (proposal.model_copy(update={"start_frame_index": 99999999}),
                    proposal.model_copy(update={"proposed_start_seconds": proposal.proposed_start_seconds + .5}),
                    proposal.model_copy(update={"context_end_seconds": metadata.duration_seconds + 1})):
        try:
            apply_boundary_proposal(candidate, invalid, metadata)
        except ValueError:
            pass
        else:
            raise AssertionError("An ungrounded/out-of-bounds proposal was accepted")

    studio.JOB_MANAGER = StudioJobs(folder / "jobs", studio.JOBS, studio.LOCK, studio.BUSY)
    calls = []

    def fixture(name):
        directory = folder / name
        directory.mkdir()
        for filename, content in originals.items():
            save_json(directory / filename, json.loads(content))
        labeled = read_json(directory / "task.json")
        labeled["request"] = labeled["task"]["query"] = "离线边界修正验证（MOCK，不代表语义识别结果）"
        save_json(directory / "task.json", labeled)
        if name in {"rejected", "repeated_proposal"}:
            save_json(directory / "candidates.json", [candidate.model_dump(mode="json")])
        return studio.load_run(directory)

    def key():
        value = uuid4().hex
        studio.JOBS[value] = {"id": value, "events": [], "status": "running", "started": time.time()}
        return value

    class MockAgent:
        strict_verdict = "pass"
        second_proposal = False
        closed = 0
        current_proposal = proposal

        def __init__(self, *args, **kwargs):
            pass

        async def review_interval(self, task, video, candidate_id, directory, start, end, **kwargs):
            calls.append({"kind": "source", "range": [start, end], "allowed": kwargs.get("allow_boundary_repair")})
            strict = directory.name == "repaired_source_review"
            return ClipReview(candidate_id=candidate_id, file_path=video,
                verdict=self.strict_verdict if strict else "uncertain", sampled_frames=6,
                reason="MOCK_ONLY: controller wiring, not semantic evidence",
                boundary_proposal=self.current_proposal if not strict or self.second_proposal else None)

        async def review_clip(self, task, clip, directory, **kwargs):
            calls.append({"kind": "output", "range": [clip.source_start_seconds, clip.source_end_seconds]})
            assert clip.file_path.is_file()
            assert kwargs["source_review"].verdict == "pass" and kwargs["source_review"].boundary_proposal is None
            return ClipReview(candidate_id=clip.candidate_id, file_path=clip.file_path, verdict="pass",
                              reason="MOCK_ONLY: actual FFmpeg file, synthetic output verdict", sampled_frames=4)

        async def close(self):
            type(self).closed += 1

    with patch.object(search, "VideoReviewAgent", MockAgent), patch.object(search.Settings, "from_env", return_value=None):
        run = fixture("accepted")
        result = asyncio.run(studio.review(key(), {"run_id": run["id"], "allow_cloud": True}, persist_in_run=True))
        accepted = result["run"]
        new_range = [effective.start_seconds, effective.end_seconds]
        assert result["reviewed"] and len(calls) == 3
        assert calls[0]["allowed"] is True and calls[1]["allowed"] is False
        assert calls[1]["range"] == calls[2]["range"] == new_range
        refreshed = studio.load_run(Path(accepted["path"]))
        current = refreshed["candidates"][0]
        assert [current["start_seconds"], current["end_seconds"]] == new_range
        assert current["agent_boundary_repair"] and current["review_verdict"] == "pass"
        assert current["model_start_seconds"] == candidate.start_seconds
        assert [refreshed["delivery"]["start"], refreshed["delivery"]["end"]] == new_range
        assert json.loads((Path(run["path"]) / "candidates.json").read_text(encoding="utf-8")) == json.loads(originals["candidates.json"])
        preview_run = folder / "preview_run"
        shutil.copytree(Path(accepted["path"]), preview_run)
        save_json(folder / "preview/studio_data/jobs/repair_preview.json", {
            "id": "repair_preview", "kind": "auto_edit", "status": "complete", "stage": "MOCK流程验证完成",
            "started": time.time(), "events": read_json(preview_run / "agent_events.json", []),
            "source_signature": source_signature(metadata.file_path), "root_run_path": str(preview_run),
            "media_id": accepted["media_id"], "run_id": "preview_run", "result": result})

        inside = [frame for frame in proposal.observed_frames
                  if candidate.start_seconds <= frame.source_timestamp_seconds < candidate.end_seconds]
        shorter = proposal.model_copy(update={"start_frame_index": inside[0].frame_index,
            "end_frame_index": inside[-1].frame_index, "proposed_start_seconds": inside[0].source_timestamp_seconds,
            "proposed_end_seconds": inside[-1].source_timestamp_seconds + 1 / metadata.fps,
            "reason": "MOCK_ONLY shorten using observed interior frames"})
        calls.clear()
        MockAgent.current_proposal = shorter
        trimmed_run = fixture("trimmed")
        trimmed = asyncio.run(studio.review(key(), {"run_id": trimmed_run["id"], "allow_cloud": True}, persist_in_run=True))
        assert trimmed["reviewed"] and len(calls) == 3 and calls[1]["range"] == calls[2]["range"]
        assert trimmed["end"] - trimmed["start"] < candidate.duration_seconds
        assert read_json(Path(trimmed_run["path"]) / "local_search.json")["boundary_repair_policy"]["performed"] == 1
        MockAgent.current_proposal = proposal

        # Manual review cannot expand the user-approved range or inherit the old pass.
        calls.clear()
        manual_start = effective.start_seconds + .1
        manual = asyncio.run(studio.review(key(), {"run_id": run["id"], "candidate_id": candidate.candidate_id,
            "start": manual_start, "end": effective.end_seconds, "allow_cloud": True}))
        assert not manual["reviewed"] and len(calls) == 1 and calls[0]["allowed"] is False
        assert manual["run"]["candidates"][0]["review_verdict"] != "pass"
        assert manual["run"]["delivery"]["start"] == effective.start_seconds

        for name, strict_verdict, repeated in (("rejected", "reject", False), ("repeated_proposal", "uncertain", True)):
            calls.clear()
            MockAgent.strict_verdict, MockAgent.second_proposal = strict_verdict, repeated
            run = fixture(name)
            result = asyncio.run(studio.review(key(), {"run_id": run["id"], "allow_cloud": True}, persist_in_run=True))
            assert not result["reviewed"] and len(calls) == 2 and not result["run"]["delivery"]
            assert not list(Path(run["path"]).rglob("*.mp4"))
        MockAgent.strict_verdict, MockAgent.second_proposal = "reject", False

        # Three identical intervals with separate IDs exercise the task-wide budget.
        calls.clear()
        budget = folder / "budget"
        budget.mkdir()
        duplicated = [candidate.model_copy(update={"candidate_id": f"budget_{i}"}) for i in range(3)]
        asyncio.run(search.deliver_first_match(task, metadata, duplicated, budget,
            allow_boundary_repair=True, use_legacy_action_verifier=False))
        assert len(calls) == 5 and sum(call.get("allowed") is True for call in calls) == 2
        assert sum(attempt.get("repair_status") == "not_authorized_or_budget_exhausted"
                   for attempt in read_json(budget / "local_search.json")["attempts"]) == 1

        calls.clear()
        sound = task.model_copy(update={"audio_condition_type": "event_activity", "audio_event_classes": ["Bark"]})
        sound_dir = folder / "audio_unchanged"
        sound_dir.mkdir()
        asyncio.run(search.deliver_first_match(sound, metadata, [candidate], sound_dir,
            allow_boundary_repair=True, use_legacy_action_verifier=False))
        assert len(calls) == 1 and calls[0]["allowed"] is False

        async def cancel_recheck():
            entered = asyncio.Event()

            class SlowAgent(MockAgent):
                async def review_interval(self, *args, **kwargs):
                    if args[3].name == "repaired_source_review":
                        entered.set()
                        await asyncio.sleep(60)
                    return await super().review_interval(*args, **kwargs)

            cancelled = fixture("cancelled")
            job_id = key()
            with patch.object(search, "VideoReviewAgent", SlowAgent):
                work = asyncio.create_task(studio.review(job_id, {"run_id": cancelled["id"], "allow_cloud": True}, persist_in_run=True))
                await asyncio.wait_for(entered.wait(), timeout=10)
                studio.JOBS[job_id]["status"] = "cancelling"
                work.cancel()
                try:
                    await work
                except asyncio.CancelledError:
                    pass
                else:
                    raise AssertionError("Cancelled recheck completed as success")
            assert read_json(Path(cancelled["path"]) / "local_search.json")["status"] == "cancelled"
            assert not studio.load_run(Path(cancelled["path"]))["delivery"]
            assert not list(Path(cancelled["path"]).rglob("*.mp4")) and SlowAgent.closed > 0

        asyncio.run(cancel_recheck())

    assert all((args.run / name).read_bytes() == content for name, content in originals.items())
    summary = {"status": "passed", "real_source_frames_used": initial.sampled_frames,
        "outside_context_cannot_certify_current_clip": True, "unguarded_or_fabricated_boundaries_rejected": True,
        "repaired_interval_freshly_reviewed_validated_and_exported": True,
        "repair_and_actual_output_ranges_identical": True, "repair_survives_history_reload": True,
        "original_neural_evidence_unchanged": True, "manual_and_audio_ranges_not_repaired": True,
        "failed_recheck_or_second_proposal_never_exported": True, "task_repair_budget_enforced": True,
        "cancellation_during_recheck_never_exports": True,
        "both_expansion_and_trimming_reach_real_export": True,
        "semantic_decisions": "MOCK ONLY, no accuracy claim", "paid_api_calls": 0, "folder": str(folder)}
    save_json(folder / "repair_check.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    main(parser.parse_args())
