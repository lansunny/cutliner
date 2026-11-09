import asyncio
from pathlib import Path

from mini_agent.video.clip_exporter import VideoClipExporter
from mini_agent.video.shot_sampler import ShotFrameSampler
from mini_agent.video.schemas import (
    ClipValidationReport,
    VideoFrame,
    VideoFrameEvidence,
    VideoMetadata,
    VideoTaskSpec,
    VideoExportReport,
    ShotSamplingResult,
)
from mini_agent.video.vision_analyzer import VideoVisionAnalyzer


class VideoToolExecutor:
    def __init__(self, vision: VideoVisionAnalyzer | None = None):
        self.sampler = ShotFrameSampler()
        self.vision = vision
        self.exporter = VideoClipExporter()

    def _prepare_frames(
        self,
        metadata: VideoMetadata,
        output_dir: Path,
    ) -> ShotSamplingResult:
        return self.sampler.prepare(metadata, output_dir)

    async def prepare_frames(
        self,
        metadata: VideoMetadata,
        output_dir: Path,
    ) -> ShotSamplingResult:
        return await asyncio.to_thread(
            self._prepare_frames,
            metadata,
            output_dir,
        )

    async def inspect_frames(
        self,
        task: VideoTaskSpec,
        frames: list[VideoFrame],
    ) -> list[VideoFrameEvidence]:
        if self.vision is None:
            raise ValueError("视觉观察工具尚未初始化")
        return await self.vision.analyze_batch(task, frames)

    async def stream_inspect(
        self, task: VideoTaskSpec, metadata: VideoMetadata, output_dir: Path,
        on_frames=None, on_evidence=None, concurrency: int = 2, queue_size: int = 4,
    ) -> tuple[ShotSamplingResult, list[VideoFrameEvidence]]:
        """Decode in a worker thread while bounded asynchronous workers call the VLM."""
        if self.vision is None:
            raise ValueError("视觉观察工具尚未初始化")
        if concurrency < 1 or queue_size < 1:
            raise ValueError("并发数和队列长度必须大于0")
        queue = asyncio.Queue(maxsize=queue_size)
        iterator = self.sampler.iter_prepare(metadata, output_dir)
        sampling = None
        evidence = []
        frames = []

        async def produce():
            nonlocal sampling
            try:
                while True:
                    # Shield next() so cancellation never closes a running generator.
                    step = asyncio.create_task(asyncio.to_thread(next, iterator, None))
                    try:
                        item = await asyncio.shield(step)
                    except asyncio.CancelledError:
                        await step
                        raise
                    if item is None:
                        break
                    if isinstance(item, ShotSamplingResult):
                        sampling = item
                        continue
                    frames.extend(item)
                    if on_frames is not None:
                        on_frames(list(frames))
                    for offset in range(0, len(item), 4):
                        await queue.put(item[offset:offset + 4])
                for _ in range(concurrency):
                    await queue.put(None)
            finally:
                await asyncio.to_thread(iterator.close)

        async def consume(worker_id):
            while True:
                batch = await queue.get()
                if batch is None:
                    return
                print(f"[视觉识别{worker_id}] 开始识别{len(batch)}张，首帧{batch[0].timestamp_seconds:.2f}秒", flush=True)
                result = await self.inspect_frames(task, batch)
                evidence.extend(result)
                if on_evidence is not None:
                    on_evidence(sorted(evidence, key=lambda item: item.frame.timestamp_seconds), result)

        async with asyncio.TaskGroup() as group:
            group.create_task(produce())
            for worker_id in range(1, concurrency + 1):
                group.create_task(consume(worker_id))
        if sampling is None:
            raise RuntimeError("流式采样未返回最终镜头结构")
        frame_map = {frame.frame_index: frame for frame in sampling.frames}
        if len(evidence) != len(frame_map) or {item.frame.frame_index for item in evidence} != set(frame_map):
            raise ValueError("流式观察结果不完整或包含重复帧")
        # Early observations gain final boundaries before candidate generation.
        evidence = sorted([
            item.model_copy(update={"frame": frame_map[item.frame.frame_index]})
            for item in evidence
        ], key=lambda item: item.frame.timestamp_seconds)
        return sampling, evidence

    async def export_clips(
        self,
        task: VideoTaskSpec,
        metadata: VideoMetadata,
        report: ClipValidationReport,
        output_dir: Path,
    ) -> VideoExportReport:
        return await asyncio.to_thread(
            self.exporter.export,
            task,
            metadata,
            report,
            output_dir,
        )
