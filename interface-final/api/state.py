"""Durable run and study state for the Interface-Final API."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from threshold_analysis.data_models import ThresholdResults

from .runtime import RuntimeSettings
from .services.channels import DEFAULT_CHANNEL_DEFINITIONS
from .services.ratios import DEFAULT_RATIO_DEFINITIONS


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _default_project_id() -> str:
    return RuntimeSettings.from_env().project_id


def _default_session_id() -> str:
    return RuntimeSettings.from_env().session_id


def _serialize_datetime(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat()


def _deserialize_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _serialize_path(value: Optional[Path]) -> Optional[str]:
    return str(value) if value is not None else None


def _deserialize_path(value: Optional[str]) -> Optional[Path]:
    return Path(value).expanduser().resolve() if value else None


def _serialize_json(value: Any) -> str:
    return json.dumps(value if value is not None else [])


def _deserialize_json(value: Optional[str], default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


@dataclass
class RunRecord:
    job_id: str
    project_id: str = field(default_factory=_default_project_id)
    session_id: str = field(default_factory=_default_session_id)
    state: str = "queued"
    message: Optional[str] = None
    input_dir: Optional[Path] = None
    config_path: Optional[Path] = None
    output_path: Optional[Path] = None
    study_name: Optional[str] = None
    is_3d: bool = True
    started_at: datetime = field(default_factory=utc_now)
    completed_at: Optional[datetime] = None
    sources_latest_mtime: Optional[float] = None
    source_hash: Optional[str] = None
    metadata_path: Optional[Path] = None
    preview_root: Optional[Path] = None
    ratio_definitions: List[Dict[str, object]] = field(default_factory=list)
    channel_definitions: List[Dict[str, object]] = field(default_factory=lambda: list(DEFAULT_CHANNEL_DEFINITIONS))
    pixel_size_um: Optional[float] = None
    progress_completed: int = 0
    progress_total: Optional[int] = None
    marker: Optional[str] = None
    n_jobs: int = 1
    max_threshold: int = 4095
    updated_at: datetime = field(default_factory=utc_now)


@dataclass
class StudyRecord:
    study_id: str
    results: ThresholdResults
    source_path: Path
    project_id: str = field(default_factory=_default_project_id)
    session_id: str = field(default_factory=_default_session_id)
    input_dir: Optional[Path] = None
    replicate_lookup: Dict[str, Dict[str, Path]] = field(default_factory=dict)
    raw_cache: Dict[str, Dict[int, np.ndarray]] = field(default_factory=dict)
    raw_cache_source: Dict[str, str] = field(default_factory=dict)
    preview_cache: Dict[str, Path] = field(default_factory=dict)
    preview_plane_root: Optional[Path] = None
    is_3d: bool = True
    loaded_at: datetime = field(default_factory=utc_now)
    ratio_definitions: List[Dict[str, object]] = field(default_factory=lambda: list(DEFAULT_RATIO_DEFINITIONS))
    channel_definitions: List[Dict[str, object]] = field(default_factory=lambda: list(DEFAULT_CHANNEL_DEFINITIONS))
    pixel_size_um: Optional[float] = None
    analysis_cache: Dict[str, Dict[str, object]] = field(default_factory=dict)
    statistics_cache: Dict[str, Dict[str, object]] = field(default_factory=dict)
    analysis_index: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PersistedStudyRecord:
    study_id: str
    project_id: str
    session_id: str
    study_name: str
    source_path: Path
    input_dir: Optional[Path]
    preview_plane_root: Optional[Path]
    is_3d: bool
    loaded_at: datetime
    ratio_definitions: List[Dict[str, object]]
    channel_definitions: List[Dict[str, object]]
    pixel_size_um: Optional[float]
    groups: List[str]
    image_count: int
    mice_count: int


@dataclass(frozen=True)
class StudySummary:
    study_id: str
    study_name: str
    source_path: Path
    loaded_at: datetime
    groups: List[str]
    image_count: int
    mice_count: int


class GlobalState:
    """SQLite-backed state container with lazy study rehydration."""

    def __init__(self, settings: Optional[RuntimeSettings] = None) -> None:
        self._settings = settings or RuntimeSettings.from_env()
        self._study_cache: Dict[str, StudyRecord] = {}
        self._study_loader: Optional[Callable[[PersistedStudyRecord], StudyRecord]] = None
        self._lock = Lock()
        self._initialized = False
        self.initialize()

    @property
    def settings(self) -> RuntimeSettings:
        return self._settings

    def configure(self, settings: Optional[RuntimeSettings] = None) -> None:
        with self._lock:
            self._settings = settings or RuntimeSettings.from_env()
            self._study_cache.clear()
            self._initialized = False
        self.initialize()

    def set_study_loader(self, loader: Callable[[PersistedStudyRecord], StudyRecord]) -> None:
        self._study_loader = loader

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                return
            self._settings.state_db.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("PRAGMA foreign_keys=ON")
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS runs (
                        job_id TEXT PRIMARY KEY,
                        project_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        state TEXT NOT NULL,
                        message TEXT,
                        input_dir TEXT,
                        config_path TEXT,
                        output_path TEXT,
                        study_name TEXT,
                        is_3d INTEGER NOT NULL,
                        started_at TEXT NOT NULL,
                        completed_at TEXT,
                        sources_latest_mtime REAL,
                        source_hash TEXT,
                        metadata_path TEXT,
                        preview_root TEXT,
                        ratio_definitions_json TEXT NOT NULL,
                        channel_definitions_json TEXT NOT NULL,
                        pixel_size_um REAL,
                        progress_completed INTEGER NOT NULL,
                        progress_total INTEGER,
                        marker TEXT,
                        n_jobs INTEGER NOT NULL,
                        max_threshold INTEGER NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_runs_project_state ON runs (project_id, state, started_at);
                    CREATE INDEX IF NOT EXISTS idx_runs_output_path ON runs (project_id, output_path, started_at);

                    CREATE TABLE IF NOT EXISTS studies (
                        project_id TEXT NOT NULL,
                        study_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        study_name TEXT NOT NULL,
                        source_path TEXT NOT NULL,
                        input_dir TEXT,
                        preview_plane_root TEXT,
                        is_3d INTEGER NOT NULL,
                        loaded_at TEXT NOT NULL,
                        ratio_definitions_json TEXT NOT NULL,
                        channel_definitions_json TEXT NOT NULL,
                        pixel_size_um REAL,
                        groups_json TEXT NOT NULL,
                        image_count INTEGER NOT NULL,
                        mice_count INTEGER NOT NULL,
                        PRIMARY KEY (project_id, study_id)
                    );

                    CREATE TABLE IF NOT EXISTS worker_heartbeats (
                        worker_id TEXT PRIMARY KEY,
                        project_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        heartbeat_at TEXT NOT NULL,
                        status TEXT NOT NULL,
                        details_json TEXT
                    );
                    CREATE INDEX IF NOT EXISTS idx_worker_heartbeats_project_time
                        ON worker_heartbeats (project_id, heartbeat_at);
                    """
                )
            self._initialized = True

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self._settings.state_db), timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        return connection

    def record_run(self, record: RunRecord) -> None:
        self.initialize()
        record.updated_at = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO runs (
                    job_id, project_id, session_id, state, message, input_dir, config_path, output_path, study_name,
                    is_3d, started_at, completed_at, sources_latest_mtime, source_hash, metadata_path, preview_root,
                    ratio_definitions_json, channel_definitions_json, pixel_size_um, progress_completed, progress_total,
                    marker, n_jobs, max_threshold, updated_at
                ) VALUES (
                    :job_id, :project_id, :session_id, :state, :message, :input_dir, :config_path, :output_path, :study_name,
                    :is_3d, :started_at, :completed_at, :sources_latest_mtime, :source_hash, :metadata_path, :preview_root,
                    :ratio_definitions_json, :channel_definitions_json, :pixel_size_um, :progress_completed, :progress_total,
                    :marker, :n_jobs, :max_threshold, :updated_at
                )
                """,
                self._run_payload(record),
            )

    def update_run(self, job_id: str, **kwargs: Any) -> RunRecord:
        record = self.get_run(job_id)
        for key, value in kwargs.items():
            setattr(record, key, value)
        self.record_run(record)
        return record

    def get_run(self, job_id: str) -> RunRecord:
        self.initialize()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE job_id = ? AND project_id = ?",
                (job_id, self._settings.project_id),
            ).fetchone()
        if row is None:
            raise KeyError(job_id)
        return self._row_to_run(row)

    def all_runs(self) -> Dict[str, RunRecord]:
        self.initialize()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM runs WHERE project_id = ? ORDER BY started_at ASC",
                (self._settings.project_id,),
            ).fetchall()
        return {row["job_id"]: self._row_to_run(row) for row in rows}

    def find_run_by_output_path(self, output_path: Path) -> Optional[RunRecord]:
        self.initialize()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM runs
                WHERE project_id = ? AND output_path = ?
                ORDER BY started_at DESC
                LIMIT 1
                """,
                (self._settings.project_id, str(output_path.resolve())),
            ).fetchone()
        return self._row_to_run(row) if row is not None else None

    def claim_next_queued_run(self) -> Optional[RunRecord]:
        self.initialize()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT job_id FROM runs
                WHERE project_id = ? AND state = 'queued'
                ORDER BY started_at ASC, job_id ASC
                LIMIT 1
                """,
                (self._settings.project_id,),
            ).fetchone()
            if row is None:
                connection.rollback()
                return None
            now = utc_now()
            updated = connection.execute(
                """
                UPDATE runs
                SET state = ?, message = ?, updated_at = ?
                WHERE job_id = ? AND project_id = ? AND state = 'queued'
                """,
                ("running", "Processing microscopy files", _serialize_datetime(now), row["job_id"], self._settings.project_id),
            )
            if updated.rowcount != 1:
                connection.rollback()
                return None
            connection.commit()
        return self.get_run(str(row["job_id"]))

    def mark_stale_running_runs_failed(self, message: str) -> int:
        self.initialize()
        now = utc_now()
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE runs
                SET state = ?, message = ?, completed_at = ?, updated_at = ?
                WHERE project_id = ? AND state = 'running'
                """,
                (
                    "failed",
                    message,
                    _serialize_datetime(now),
                    _serialize_datetime(now),
                    self._settings.project_id,
                ),
            )
        return int(updated.rowcount or 0)

    def save_study(self, record: StudyRecord) -> None:
        self.initialize()
        payload = self._study_payload(record)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO studies (
                    project_id, study_id, session_id, study_name, source_path, input_dir, preview_plane_root, is_3d,
                    loaded_at, ratio_definitions_json, channel_definitions_json, pixel_size_um, groups_json,
                    image_count, mice_count
                ) VALUES (
                    :project_id, :study_id, :session_id, :study_name, :source_path, :input_dir, :preview_plane_root, :is_3d,
                    :loaded_at, :ratio_definitions_json, :channel_definitions_json, :pixel_size_um, :groups_json,
                    :image_count, :mice_count
                )
                """,
                payload,
            )
        with self._lock:
            self._study_cache[record.study_id] = record

    def add_study(self, record: StudyRecord) -> None:
        self.save_study(record)

    def get_study(self, study_id: str) -> StudyRecord:
        with self._lock:
            cached = self._study_cache.get(study_id)
        if cached is not None:
            return cached

        persisted = self.get_persisted_study(study_id)
        if persisted is None:
            raise KeyError(study_id)
        if self._study_loader is None:
            raise KeyError(study_id)
        record = self._study_loader(persisted)
        with self._lock:
            self._study_cache[study_id] = record
        return record

    def get_persisted_study(self, study_id: str) -> Optional[PersistedStudyRecord]:
        self.initialize()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM studies
                WHERE project_id = ? AND study_id = ?
                """,
                (self._settings.project_id, study_id),
            ).fetchone()
        return self._row_to_study(row) if row is not None else None

    def list_studies(self) -> Dict[str, StudySummary]:
        self.initialize()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM studies
                WHERE project_id = ?
                ORDER BY loaded_at ASC
                """,
                (self._settings.project_id,),
            ).fetchall()
        summaries: Dict[str, StudySummary] = {}
        for row in rows:
            groups = _deserialize_json(row["groups_json"], [])
            summaries[str(row["study_id"])] = StudySummary(
                study_id=str(row["study_id"]),
                study_name=str(row["study_name"]),
                source_path=Path(str(row["source_path"])).expanduser().resolve(),
                loaded_at=_deserialize_datetime(str(row["loaded_at"])) or utc_now(),
                groups=list(groups),
                image_count=int(row["image_count"]),
                mice_count=int(row["mice_count"]),
            )
        return summaries

    def record_worker_heartbeat(self, worker_id: str, status: str = "alive", details: Optional[Dict[str, Any]] = None) -> None:
        self.initialize()
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO worker_heartbeats (
                    worker_id, project_id, session_id, heartbeat_at, status, details_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    worker_id,
                    self._settings.project_id,
                    self._settings.session_id,
                    _serialize_datetime(now),
                    status,
                    _serialize_json(details or {}),
                ),
            )

    def latest_worker_heartbeat(self) -> Optional[Dict[str, Any]]:
        self.initialize()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM worker_heartbeats
                WHERE project_id = ?
                ORDER BY heartbeat_at DESC
                LIMIT 1
                """,
                (self._settings.project_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "worker_id": str(row["worker_id"]),
            "session_id": str(row["session_id"]),
            "heartbeat_at": _deserialize_datetime(str(row["heartbeat_at"])),
            "status": str(row["status"]),
            "details": _deserialize_json(row["details_json"], {}),
        }

    def health_report(self) -> Dict[str, Any]:
        self.initialize()
        ok = True
        error: Optional[str] = None
        try:
            with self._connect() as connection:
                connection.execute("SELECT 1").fetchone()
        except sqlite3.Error as exc:
            ok = False
            error = str(exc)
        return {
            "status": "ok" if ok else "error",
            "database": "ok" if ok else "error",
            "state_db": str(self._settings.state_db),
            "project_id": self._settings.project_id,
            "session_id": self._settings.session_id,
            "error": error,
        }

    def readiness_report(self) -> Dict[str, Any]:
        self.initialize()
        schema_initialized = self._schema_initialized()
        queue_writable = self._queue_writable()
        heartbeat = self.latest_worker_heartbeat()
        worker_alive = False
        if heartbeat and isinstance(heartbeat.get("heartbeat_at"), datetime):
            worker_alive = heartbeat["heartbeat_at"] >= utc_now() - timedelta(
                seconds=self._settings.worker_heartbeat_ttl_seconds
            )
        ready = schema_initialized and queue_writable and worker_alive
        return {
            "status": "ok" if ready else "error",
            "ready": ready,
            "database": "ok",
            "schema_initialized": schema_initialized,
            "queue_writable": queue_writable,
            "worker_alive": worker_alive,
            "worker_heartbeat_ttl_seconds": self._settings.worker_heartbeat_ttl_seconds,
            "last_worker_heartbeat_at": (
                _serialize_datetime(heartbeat["heartbeat_at"]) if heartbeat and heartbeat.get("heartbeat_at") else None
            ),
            "worker_status": heartbeat.get("status") if heartbeat else None,
            "state_db": str(self._settings.state_db),
            "project_id": self._settings.project_id,
            "session_id": self._settings.session_id,
        }

    def _schema_initialized(self) -> bool:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type = 'table' AND name IN ('runs', 'studies', 'worker_heartbeats')
                """
            ).fetchall()
        return len(rows) == 3

    def _queue_writable(self) -> bool:
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.rollback()
            return True
        except sqlite3.Error:
            return False

    def _run_payload(self, record: RunRecord) -> Dict[str, Any]:
        return {
            "job_id": record.job_id,
            "project_id": record.project_id,
            "session_id": record.session_id,
            "state": record.state,
            "message": record.message,
            "input_dir": _serialize_path(record.input_dir),
            "config_path": _serialize_path(record.config_path),
            "output_path": _serialize_path(record.output_path),
            "study_name": record.study_name,
            "is_3d": int(record.is_3d),
            "started_at": _serialize_datetime(record.started_at),
            "completed_at": _serialize_datetime(record.completed_at),
            "sources_latest_mtime": record.sources_latest_mtime,
            "source_hash": record.source_hash,
            "metadata_path": _serialize_path(record.metadata_path),
            "preview_root": _serialize_path(record.preview_root),
            "ratio_definitions_json": _serialize_json(record.ratio_definitions),
            "channel_definitions_json": _serialize_json(record.channel_definitions),
            "pixel_size_um": record.pixel_size_um,
            "progress_completed": record.progress_completed,
            "progress_total": record.progress_total,
            "marker": record.marker,
            "n_jobs": record.n_jobs,
            "max_threshold": record.max_threshold,
            "updated_at": _serialize_datetime(record.updated_at),
        }

    def _row_to_run(self, row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            job_id=str(row["job_id"]),
            project_id=str(row["project_id"]),
            session_id=str(row["session_id"]),
            state=str(row["state"]),
            message=row["message"],
            input_dir=_deserialize_path(row["input_dir"]),
            config_path=_deserialize_path(row["config_path"]),
            output_path=_deserialize_path(row["output_path"]),
            study_name=row["study_name"],
            is_3d=bool(row["is_3d"]),
            started_at=_deserialize_datetime(str(row["started_at"])) or utc_now(),
            completed_at=_deserialize_datetime(row["completed_at"]),
            sources_latest_mtime=row["sources_latest_mtime"],
            source_hash=row["source_hash"],
            metadata_path=_deserialize_path(row["metadata_path"]),
            preview_root=_deserialize_path(row["preview_root"]),
            ratio_definitions=list(_deserialize_json(row["ratio_definitions_json"], [])),
            channel_definitions=list(_deserialize_json(row["channel_definitions_json"], list(DEFAULT_CHANNEL_DEFINITIONS))),
            pixel_size_um=row["pixel_size_um"],
            progress_completed=int(row["progress_completed"]),
            progress_total=row["progress_total"],
            marker=row["marker"],
            n_jobs=int(row["n_jobs"]),
            max_threshold=int(row["max_threshold"]),
            updated_at=_deserialize_datetime(str(row["updated_at"])) or utc_now(),
        )

    def _study_payload(self, record: StudyRecord) -> Dict[str, Any]:
        return {
            "project_id": record.project_id,
            "study_id": record.study_id,
            "session_id": record.session_id,
            "study_name": record.results.study_name,
            "source_path": str(record.source_path.resolve()),
            "input_dir": _serialize_path(record.input_dir),
            "preview_plane_root": _serialize_path(record.preview_plane_root),
            "is_3d": int(record.is_3d),
            "loaded_at": _serialize_datetime(record.loaded_at),
            "ratio_definitions_json": _serialize_json(record.ratio_definitions),
            "channel_definitions_json": _serialize_json(record.channel_definitions),
            "pixel_size_um": record.pixel_size_um,
            "groups_json": _serialize_json(list(record.results.group_info.keys())),
            "image_count": len(record.results.image_data),
            "mice_count": len({entry.mouse_id for entry in record.results.image_data}),
        }

    def _row_to_study(self, row: sqlite3.Row) -> PersistedStudyRecord:
        return PersistedStudyRecord(
            study_id=str(row["study_id"]),
            project_id=str(row["project_id"]),
            session_id=str(row["session_id"]),
            study_name=str(row["study_name"]),
            source_path=Path(str(row["source_path"])).expanduser().resolve(),
            input_dir=_deserialize_path(row["input_dir"]),
            preview_plane_root=_deserialize_path(row["preview_plane_root"]),
            is_3d=bool(row["is_3d"]),
            loaded_at=_deserialize_datetime(str(row["loaded_at"])) or utc_now(),
            ratio_definitions=list(_deserialize_json(row["ratio_definitions_json"], list(DEFAULT_RATIO_DEFINITIONS))),
            channel_definitions=list(_deserialize_json(row["channel_definitions_json"], list(DEFAULT_CHANNEL_DEFINITIONS))),
            pixel_size_um=row["pixel_size_um"],
            groups=list(_deserialize_json(row["groups_json"], [])),
            image_count=int(row["image_count"]),
            mice_count=int(row["mice_count"]),
        )


STATE = GlobalState()
