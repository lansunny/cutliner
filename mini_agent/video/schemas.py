from pathlib import Path
from typing import Literal
from pydantic import BaseModel,ConfigDict,Field,model_validator

class VideoModel(BaseModel):
    model_config=ConfigDict(
        extra="forbid"
    )

class VideoTaskSpec(VideoModel):
    query: str = Field(
        min_length=1,
        max_length=2000,
        description="需要检索的画面、声音和对白内容，保留组合条件和排除条件",
    )
    audio_condition_type: Literal["none", "event_activity", "spoken_content", "unsupported"] = Field(
        default="none",
        description="纯视觉为none；明确要求有声且可见的受支持事件为event_activity；具体台词或对白语义为spoken_content；画外声、否定声音条件或不支持的事件为unsupported",
    )
    audio_event_classes: list[str] = Field(
        default_factory=list, max_length=4,
        description="event_activity时填写正向声画事件的标准英文类别；其余情况为空。不把具体台词、人物身份或否定条件降级成语音活动",
    )
    visual_query_en: str | None = Field(default=None, min_length=1, max_length=2000,
        description="供英文CLIP/QD-DETR使用的准确英文视觉查询，保留可见属性、关系、动作和排除条件，不编造声音；完整需求仍在query")
    requires_extra_visual_grounding: bool = Field(default=False,
        description="声音任务还包含发声源类别以外的服装、人物身份、空间关系或额外动作时为true；纯视觉任务始终使用视觉定位器")
    audio_match_mode: Literal["all", "any"] = Field(default="all",
        description="多个声音类别全部需要为all，明确或关系为any；具体顺序仍由Agent结合完整query审核")
    max_total_duration_seconds: float | None = Field(
        default=None,
        gt=0.0,
        allow_inf_nan=False,
        description="仅用户明确指定时设置总时长上限，未指定为null",
    )
    max_clips: int | None = Field(
        default=None,
        ge=1,
        description="仅用户明确指定时设置片段数量上限，未指定为null，由内容决定数量",
    )
    clip_duration_seconds: float | None = Field(
        default=None,
        gt=0.0,
        allow_inf_nan=False,
        description="用户期望的单段时长偏好，非精确长度约束；未指定为null，按内容自动裁剪",
    )
    prefer_clear_frames: bool = Field(
        default=True,
        description="在满足检索目标的候选中，是否优先选择清晰画面",
    )

class VideoMetadata(VideoModel):
    file_path:Path
    duration_seconds:float=Field(ge=0.0)
    fps:float=Field(gt=0.0)
    frame_count:int=Field(ge=0)
    width:int=Field(gt=0)
    height:int=Field(gt=0)

class VideoFrame(VideoModel):
    frame_index:int=Field(ge=0)
    timestamp_seconds:float=Field(ge=0.0)
    image_path:Path
    shot_id: str | None = None
    shot_start_seconds: float | None = Field(default=None, ge=0.0)
    shot_end_seconds: float | None = Field(default=None, gt=0.0)
    sampling_reason: str | None = None
    sharpness_score:float|None=Field(
        default=None,
        ge=0.0
    )
    brightness_score:float|None=Field(
        default=None,
        ge=0.0,
        le=255.0
    )

class VideoFrameObservation(VideoModel):
    frame_index: int = Field(ge=0)
    description: str = Field(min_length=1, max_length=600)
    matches_query: bool = Field(description="是否符合可见必要条件；静态图片不能单独证明声音或对白已满足")
    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        description="任务相关度评分，用于候选排序，不代表校准后的正确率",
    )
    reason: str = Field(min_length=1, max_length=400)


class VideoObservationBatch(VideoModel):
    observations: list[VideoFrameObservation] = Field(min_length=1)


class VideoFrameEvidence(VideoModel):
    frame: VideoFrame
    observation: VideoFrameObservation

class ModelIntervalEvidence(VideoModel):
    modality: Literal["visual", "audio_visual"]
    model_name: str
    label: str
    start_seconds: float = Field(ge=0, allow_inf_nan=False)
    end_seconds: float = Field(gt=0, allow_inf_nan=False)
    score: float = Field(ge=0, le=1, allow_inf_nan=False)
    score_is_calibrated_probability: bool = False
    window_index: int | None = None


class BoundaryRefinement(VideoModel):
    model_name: str
    feature_profile: str
    frozen_protocol_sha256: str
    deployment_seed: int
    coarse_start_seconds: float = Field(ge=0, allow_inf_nan=False)
    coarse_end_seconds: float = Field(gt=0, allow_inf_nan=False)
    observation_start_seconds: float = Field(ge=0, allow_inf_nan=False)
    observation_end_seconds: float = Field(gt=0, allow_inf_nan=False)
    proposed_start_seconds: float = Field(ge=0, allow_inf_nan=False)
    proposed_end_seconds: float = Field(gt=0, allow_inf_nan=False)
    coarse_rank: int = Field(ge=1, le=3)
    selection_logit: float = Field(allow_inf_nan=False)
    observed_frames: int = Field(ge=2)


class BoundaryObservation(VideoModel):
    frame_index: int = Field(ge=0)
    source_timestamp_seconds: float = Field(ge=0, allow_inf_nan=False)


class AgentBoundaryRepair(VideoModel):
    original_start_seconds: float = Field(ge=0, allow_inf_nan=False)
    original_end_seconds: float = Field(gt=0, allow_inf_nan=False)
    proposed_start_seconds: float = Field(ge=0, allow_inf_nan=False)
    proposed_end_seconds: float = Field(gt=0, allow_inf_nan=False)
    context_start_seconds: float = Field(ge=0, allow_inf_nan=False)
    context_end_seconds: float = Field(gt=0, allow_inf_nan=False)
    start_frame_index: int = Field(ge=0)
    end_frame_index: int = Field(ge=0)
    observed_frames: list[BoundaryObservation] = Field(min_length=2, max_length=12)
    reason: str = Field(min_length=1, max_length=500)
    scope: Literal["agent_observed_frame_proposal_v1"] = "agent_observed_frame_proposal_v1"


class ClipCandidate(VideoModel):
    candidate_id: str = Field(min_length=1)
    start_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    end_seconds: float = Field(gt=0.0, allow_inf_nan=False)
    shot_id: str | None = None
    shot_start_seconds: float | None = Field(default=None, ge=0.0)
    shot_end_seconds: float | None = Field(default=None, gt=0.0)
    anchor_frame_index: int | None = Field(default=None, ge=0)
    evidence_frame_indices: list[int] = Field(default_factory=list)
    observations: list[VideoFrameObservation] = Field(default_factory=list)
    evidence_source: Literal["visual_observations", "qd_detr", "av_event", "hybrid_neural"] = "visual_observations"
    model_evidence: list[ModelIntervalEvidence] = Field(default_factory=list)
    retrieval_notes: list[str] = Field(default_factory=list)
    review_priority: int | None = Field(default=None, ge=1)
    boundary_refinement: BoundaryRefinement | None = None
    agent_boundary_repair: AgentBoundaryRepair | None = None
    relevance_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    sharpness_score: float | None = Field(default=None, ge=0.0)
    boundary_note: str = "时间范围由匹配采样帧附近的固定窗口估计，连续内容尚未验证"

    @model_validator(mode="after")
    def require_real_evidence(self):
        if self.evidence_source == "visual_observations":
            if self.anchor_frame_index is None or not self.evidence_frame_indices or not self.observations:
                raise ValueError("视觉观察候选需要实际观察的帧和记录")
        elif not self.model_evidence:
            raise ValueError("神经模型候选需要真实模型区间证据，不能伪造视觉观察")
        return self

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


class ClipChoice(VideoModel):
    candidate_id: str = Field(min_length=1)
    reason: str = Field(min_length=1, max_length=400)


class ClipSelection(VideoModel):
    choices: list[ClipChoice] = Field(default_factory=list)
    summary: str = Field(min_length=1, max_length=600)


class SelectedClip(ClipCandidate):
    selection_reason: str = Field(min_length=1, max_length=400)


class ClipValidationReport(VideoModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    total_duration_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    selected_clips: list[SelectedClip] = Field(default_factory=list)

class FileSignature(VideoModel):
    path: str
    size: int = Field(ge=0)
    mtime_ns: int = Field(ge=0)


class ExportedVideoClip(VideoModel):
    candidate_id: str = Field(min_length=1)
    file_path: Path
    file_signature: FileSignature | None = None
    source_start_seconds: float = Field(ge=0.0)
    source_end_seconds: float = Field(gt=0.0)
    expected_duration_seconds: float = Field(gt=0.0)
    video_duration_seconds: float = Field(gt=0.0)
    container_duration_seconds: float = Field(gt=0.0)
    duration_tolerance_seconds: float = Field(gt=0.0)
    fps: float = Field(gt=0.0)
    frame_count: int = Field(gt=0)


class VideoExportReport(VideoModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    clips: list[ExportedVideoClip] = Field(default_factory=list)
    total_video_duration_seconds: float = Field(ge=0.0)
    total_container_duration_seconds: float = Field(ge=0.0)

class ShotSamplingConfig(VideoModel):
    scan_fps: float = Field(default=4.0, gt=0.0, allow_inf_nan=False)
    thumbnail_max_side: int = Field(default=160, ge=32)
    quality_max_side: int = Field(default=320, ge=32)
    representative_max_side: int = Field(default=960, ge=64)
    histogram_threshold: float = Field(default=0.35, gt=0.0, le=1.0)
    minimum_rgb_change: float = Field(default=0.035, ge=0.0, le=1.0)
    suppress_exposure_flashes: bool = True
    adaptive_window_seconds: float = Field(default=3.0, gt=0.0)
    mad_multiplier: float = Field(default=3.0, ge=0.0)
    min_shot_seconds: float = Field(default=0.5, gt=0.0)
    representative_window_seconds: float = Field(default=10.0, gt=0.0)
    static_coverage_seconds: float = Field(default=30.0, gt=0.0)
    novelty_histogram_distance: float = Field(default=0.15, ge=0.0, le=1.0)
    novelty_hash_bits: int = Field(default=8, ge=1, le=64)


class ShotSegment(VideoModel):
    shot_id: str = Field(min_length=1)
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(gt=0.0)
    start_frame_index: int = Field(ge=0)
    end_frame_index_exclusive: int = Field(gt=0)
    boundary_histogram_distance: float | None = Field(default=None, ge=0.0, le=1.0)
    boundary_method: str
    boundary_scan_resolution_seconds: float = Field(gt=0.0)
    representative_frame_indices: list[int] = Field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


class FrameSamplePoint(VideoModel):
    shot_id: str
    timestamp_seconds: float = Field(ge=0.0)
    reason: str


class ShotSamplingStats(VideoModel):
    original_frame_count: int = Field(gt=0)
    scan_frame_count: int = Field(gt=0)
    shot_count: int = Field(gt=0)
    representative_frame_count: int = Field(gt=0)
    uniform_two_second_frame_count: int = Field(gt=0)
    reduction_vs_original: float = Field(ge=0.0, le=1.0)
    reduction_vs_uniform: float = Field(le=1.0)
    effective_scan_fps: float = Field(gt=0.0)
    scan_width: int = Field(gt=0)
    scan_height: int = Field(gt=0)
    elapsed_seconds: float = Field(ge=0.0)


class ShotSamplingResult(VideoModel):
    frames: list[VideoFrame]
    shots: list[ShotSegment]
    config: ShotSamplingConfig
    stats: ShotSamplingStats



















































