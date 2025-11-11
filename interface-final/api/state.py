"""In-memory state containers for the Interface-Final API."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, Optional, List

import numpy as np

from threshold_analysis.data_models import ThresholdResults
from .services.ratios import DEFAULT_RATIO_DEFINITIONS


@dataclass
class RunRecord:
    job_id: str
    state: str = "queued"
    message: Optional[str] = None
    input_dir: Optional[Path] = None
    config_path: Optional[Path] = None
    output_path: Optional[Path] = None
    study_name: Optional[str] = None
    is_3d: bool = True
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    sources_latest_mtime: Optional[float] = None
    source_hash: Optional[str] = None
    metadata_path: Optional[Path] = None
    preview_root: Optional[Path] = None
    ratio_definitions: List[Dict[str, object]] = field(default_factory=list)


@dataclass
class StudyRecord:
    study_id: str
    results: ThresholdResults
    source_path: Path
    input_dir: Optional[Path] = None
    replicate_lookup: Dict[str, Dict[str, Path]] = field(default_factory=dict)
    raw_cache: Dict[str, Dict[int, np.ndarray]] = field(default_factory=dict)
    preview_cache: Dict[str, Path] = field(default_factory=dict)
    preview_plane_root: Optional[Path] = None
    is_3d: bool = True
    loaded_at: datetime = field(default_factory=datetime.utcnow)
    ratio_definitions: List[Dict[str, object]] = field(default_factory=lambda: list(DEFAULT_RATIO_DEFINITIONS))


class GlobalState:
    """Thread-safe singleton-like container."""

    def __init__(self) -> None:
        self._runs: Dict[str, RunRecord] = {}
        self._studies: Dict[str, StudyRecord] = {}
        self._lock = Lock()

    def record_run(self, record: RunRecord) -> None:
        with self._lock:
            self._runs[record.job_id] = record

    def update_run(self, job_id: str, **kwargs) -> None:
        with self._lock:
            run = self._runs[job_id]
            for key, value in kwargs.items():
                setattr(run, key, value)

    def get_run(self, job_id: str) -> RunRecord:
        with self._lock:
            return self._runs[job_id]

    def all_runs(self) -> Dict[str, RunRecord]:
        with self._lock:
            return dict(self._runs)

    def add_study(self, record: StudyRecord) -> None:
        with self._lock:
            self._studies[record.study_id] = record

    def get_study(self, study_id: str) -> StudyRecord:
        with self._lock:
            return self._studies[study_id]

    def iter_studies(self) -> Dict[str, StudyRecord]:
        with self._lock:
            return dict(self._studies)


STATE = GlobalState()
