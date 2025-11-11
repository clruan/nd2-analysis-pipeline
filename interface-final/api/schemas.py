"""Pydantic schemas used by the Interface-Final API."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Literal

from pydantic import BaseModel, Field


class RatioDefinition(BaseModel):
    id: Optional[str] = None
    label: Optional[str] = None
    numerator_channel: int = Field(..., ge=1, le=3)
    denominator_channel: int = Field(..., ge=1, le=3)


class ConfigScanRequest(BaseModel):
    input_dir: str = Field(..., description="Directory containing ND2 data to scan")
    recursive: bool = Field(True, description="Whether to search recursively for ND2 files")


class ReplicaInfo(BaseModel):
    filename: str
    absolute_path: str


class SubjectInfo(BaseModel):
    subject_id: str
    replicates: List[ReplicaInfo]


class GroupInfo(BaseModel):
    group_name: str
    subjects: List[SubjectInfo]


class ConfigScanResponse(BaseModel):
    study_name: str
    input_dir: str
    nd2_files: List[str]
    groups: List[GroupInfo]


class ConfigCreateRequest(BaseModel):
    input_dir: str
    study_name: str
    groups: Dict[str, List[str]]
    pixel_size_um: Optional[float] = None
    thresholds: Optional[Dict[str, Dict[str, float]]] = None
    output_path: Optional[str] = None
    ratios: Optional[List[RatioDefinition]] = None


class ConfigCreateResponse(BaseModel):
    config_path: str
    study_name: str
    groups: Dict[str, List[str]]
    ratios: Optional[List[RatioDefinition]] = None


class ConfigReadResponse(BaseModel):
    config_path: str
    study_name: str
    groups: Dict[str, List[str]]
    pixel_size_um: Optional[float] = None
    thresholds: Optional[Dict[str, Dict[str, float]]] = None
    ratios: Optional[List[RatioDefinition]] = None


class ThresholdRunRequest(BaseModel):
    input_dir: str
    config_path: str
    output_path: Optional[str] = None
    is_3d: bool = True
    marker: Optional[str] = None
    n_jobs: int = 1
    max_threshold: int = 4095
    reuse_existing: bool = True


class RunStatus(BaseModel):
    job_id: str
    state: str
    message: Optional[str] = None
    output_path: Optional[str] = None
    config_path: Optional[str] = None
    input_dir: Optional[str] = None
    study_name: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    latest_source_mtime: Optional[datetime] = None
    source_hash: Optional[str] = None


class LoadStudyRequest(BaseModel):
    file_path: str
    input_dir_override: Optional[str] = Field(
        None,
        description="Optional path to the ND2 root directory if it differs from the value stored in the metadata.",
    )


class AnalyzeRequest(BaseModel):
    thresholds: Dict[str, int]


class MouseAverageRecord(BaseModel):
    Group: str
    MouseID: str
    Channel_1_area: float
    Channel_2_area: float
    Channel_3_area: float
    ratios: Dict[str, float] = Field(default_factory=dict)


class IndividualImageRecord(BaseModel):
    group: str
    mouse_id: str
    filename: str
    channel_1_area: float
    channel_2_area: float
    channel_3_area: float
    ratios: Dict[str, float] = Field(default_factory=dict)
    replicate_index: int


class AnalyzeResponse(BaseModel):
    study_id: str
    thresholds: Dict[str, int]
    mouse_averages: List["MouseAverageRecord"]
    individual_images: List["IndividualImageRecord"]


class StatisticsRequest(BaseModel):
    thresholds: Dict[str, int]
    comparison_mode: str = Field("all_vs_one", pattern="^(all_vs_one|pairs|all_pairs)$")
    reference_group: Optional[str] = None
    comparison_pairs: Optional[List[List[str]]] = None
    test_type: str = Field("anova_parametric", pattern="^(anova_parametric|anova_non_parametric|t_test)$")
    significance_display: str = Field("stars", pattern="^(stars|p_values)$")


class StatisticsResponse(BaseModel):
    statistics: Dict[str, Dict[str, object]]
    thresholds: Dict[str, int]
    test_type_used: str
    significance_display: str
    ratios: List[RatioDefinition]


class PreviewRequest(BaseModel):
    thresholds: Dict[str, int]
    groups: Optional[List[str]] = None
    max_samples_per_group: int = Field(1, ge=1, le=6)
    metric: Optional[str] = None
    metrics: Optional[List[str]] = None
    group_sample_limits: Optional[Dict[str, int]] = None


class PreviewImage(BaseModel):
    variant: str
    metric: str
    channel: Optional[int] = None
    group: str
    subject_id: str
    filename: str
    image_path: str


class PreviewResponse(BaseModel):
    study_id: str
    generated_at: datetime
    images: List[PreviewImage]
    nd2_available: bool
    nd2_source: Optional[str] = None
    max_samples_per_group: int
    ratio_definitions: List[RatioDefinition]
    group_sample_counts: Dict[str, int]


class PreviewClearRequest(BaseModel):
    scope: Literal["thresholds", "all"] = "thresholds"
    threshold_key: Optional[str] = None


class PreviewClearResponse(BaseModel):
    removed_directories: List[str]


class DownloadResponse(BaseModel):
    download_path: str
    generated_at: datetime


class UploadResponse(BaseModel):
    stored_path: str
    original_name: str
    category: str


class RatioUpdateRequest(BaseModel):
    ratios: List[RatioDefinition]


class RatioUpdateResponse(BaseModel):
    ratios: List[RatioDefinition]


class AssistantMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class StudyBuilderRequest(BaseModel):
    messages: List[AssistantMessage]
    study_name: Optional[str] = None
    existing_groups: Dict[str, List[str]] = Field(default_factory=dict)
    ratio_definitions: List[RatioDefinition] = Field(default_factory=list)


class StudyBuilderResponse(BaseModel):
    reply: str
    suggested_study_name: Optional[str] = None
    suggested_groups: Dict[str, List[str]] = Field(default_factory=dict)
    next_questions: List[str] = Field(default_factory=list)


class InterpretationGroupSummary(BaseModel):
    group: str
    mean: float
    sd: Optional[float] = None
    count: int


class InterpretationRequest(BaseModel):
    metric_id: str
    metric_label: str
    group_summaries: List[InterpretationGroupSummary]
    reference_group: Optional[str] = None
    significance_notes: Optional[str] = None
    thresholds: Optional[Dict[str, int]] = None


class InterpretationResponse(BaseModel):
    summary: str
    bullets: List[str]


AnalyzeResponse.model_rebuild()
