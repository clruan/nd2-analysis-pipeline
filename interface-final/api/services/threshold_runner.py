"""Utilities for durable threshold-generation runs."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from fastapi import HTTPException

from data_models import GroupConfig
from image_processing import ensure_microscopy_reader_dependencies, load_nd2_file
from threshold_analysis.data_models import ThresholdResults

from ..logging_utils import log_event
from ..schemas import RunStatus, ThresholdRunRequest
from ..state import RunRecord, STATE, utc_now
from ..utils import ensure_directory, find_nd2_files, normalize_path, preview_plane_filename, slugify
from .channels import DEFAULT_CHANNEL_DEFINITIONS, normalize_channel_definitions
from .ratios import DEFAULT_RATIO_DEFINITIONS, normalize_ratio_definitions
from .studies import PREVIEW_ROOT


LOGGER = logging.getLogger(__name__)
RUN_OUTPUT_ROOT = ensure_directory(Path(__file__).resolve().parent / "generated_results")


def launch_threshold_run(payload: ThresholdRunRequest) -> RunStatus:
    input_dir = normalize_path(payload.input_dir).resolve()
    config_path = normalize_path(payload.config_path).resolve()

    if not input_dir.exists():
        raise HTTPException(status_code=404, detail=f"Input directory does not exist: {input_dir}")
    if not config_path.exists():
        raise HTTPException(status_code=404, detail=f"Config file not found: {config_path}")

    ratio_definitions = list(DEFAULT_RATIO_DEFINITIONS)
    channel_definitions = list(DEFAULT_CHANNEL_DEFINITIONS)
    try:
        group_config = GroupConfig.from_json(str(config_path))
        ratio_definitions = normalize_ratio_definitions(group_config.ratios)
        channel_definitions = normalize_channel_definitions(group_config.channel_definitions)
        pixel_size_um = group_config.pixel_size_um
    except Exception:
        pixel_size_um = None

    job_id = uuid.uuid4().hex
    study_slug = slugify(input_dir.name)
    default_output = RUN_OUTPUT_ROOT / f"threshold_results_{study_slug}.json"
    output_path = normalize_path(payload.output_path).resolve() if payload.output_path else default_output
    ensure_directory(output_path.parent)

    sources_latest_mtime, source_hash, sources = _fingerprint_sources(input_dir, config_path)
    metadata_path = _metadata_path(output_path)
    try:
        microscopy_files = find_nd2_files(input_dir)
        ensure_microscopy_reader_dependencies(microscopy_files)
        progress_total = len(microscopy_files)
    except ImportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        progress_total = None

    preview_root = (PREVIEW_ROOT / study_slug / job_id).resolve()
    record = RunRecord(
        job_id=job_id,
        project_id=STATE.settings.project_id,
        session_id=STATE.settings.session_id,
        state="queued",
        message="Run scheduled",
        input_dir=input_dir,
        config_path=config_path,
        output_path=output_path,
        study_name=study_slug,
        is_3d=payload.is_3d,
        sources_latest_mtime=sources_latest_mtime,
        source_hash=source_hash,
        metadata_path=metadata_path,
        preview_root=preview_root,
        ratio_definitions=ratio_definitions,
        channel_definitions=channel_definitions,
        pixel_size_um=pixel_size_um,
        progress_total=progress_total,
        marker=payload.marker,
        n_jobs=payload.n_jobs,
        max_threshold=payload.max_threshold,
    )
    STATE.record_run(record)

    reuse_result = _maybe_reuse_existing(payload.reuse_existing, record, sources_latest_mtime, source_hash, sources)
    if reuse_result:
        return reuse_result

    log_event(
        LOGGER,
        "run_queued",
        job_id=record.job_id,
        study_id=record.study_name,
        project_id=record.project_id,
        session_id=record.session_id,
        state=record.state,
        input_dir=str(record.input_dir) if record.input_dir else None,
        output_path=str(record.output_path) if record.output_path else None,
    )
    return build_run_status(record)


def describe_run(job_id: str) -> RunStatus:
    try:
        record = STATE.get_run(job_id)
    except KeyError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=404, detail=f"Run not found: {job_id}") from exc
    return build_run_status(record)


def build_run_status(record: RunRecord) -> RunStatus:
    return RunStatus(
        job_id=record.job_id,
        state=record.state,
        message=record.message,
        input_dir=str(record.input_dir) if record.input_dir else None,
        config_path=str(record.config_path) if record.config_path else None,
        output_path=str(record.output_path) if record.output_path else None,
        study_name=record.study_name,
        started_at=record.started_at,
        completed_at=record.completed_at,
        latest_source_mtime=_as_datetime(record.sources_latest_mtime),
        source_hash=record.source_hash,
        progress_completed=record.progress_completed,
        progress_total=record.progress_total,
    )


def prepare_run_preview_root(record: RunRecord) -> Path:
    if record.preview_root is None:
        study_slug = slugify(record.study_name or record.job_id)
        record.preview_root = (PREVIEW_ROOT / study_slug / record.job_id).resolve()
    if record.preview_root.exists():
        shutil.rmtree(record.preview_root)
    ensure_directory(record.preview_root)
    ensure_directory(record.preview_root / "planes")
    return record.preview_root


def cache_run_previews(record: RunRecord, results: ThresholdResults) -> Optional[Path]:
    if not record.preview_root or not record.input_dir:
        return None

    plane_dir = ensure_directory(record.preview_root / "planes")
    try:
        file_map = {path.name: path for path in find_nd2_files(record.input_dir)}
    except Exception:
        file_map = {}

    saved_any = False

    for entry in results.image_data:
        source_path = file_map.get(entry.filename)
        if source_path is None:
            continue
        try:
            channel_arrays = load_nd2_file(str(source_path), is_3d=record.is_3d)
        except Exception:
            continue

        for channel_index, plane in enumerate(channel_arrays, start=1):
            array = np.asarray(plane)
            if array.ndim > 2:
                array = array.max(axis=0)
            target_path = plane_dir / preview_plane_filename(entry.group, entry.mouse_id, entry.filename, channel_index)
            try:
                np.save(target_path, np.clip(array, 0, 65535).astype(np.uint16), allow_pickle=False)
                saved_any = True
            except Exception:
                continue

    if not saved_any:
        return None
    return record.preview_root


def write_run_metadata(
    metadata_path: Optional[Path],
    record: RunRecord,
    latest_mtime: float,
    source_hash: str,
    sources: Iterable[Dict[str, object]],
) -> None:
    _write_metadata(metadata_path, record, latest_mtime, source_hash, sources)


def fingerprint_run_sources(input_dir: Path, config_path: Path) -> Tuple[float, str, List[Dict[str, object]]]:
    return _fingerprint_sources(input_dir, config_path)


def _maybe_reuse_existing(
    reuse_existing: bool,
    record: RunRecord,
    sources_latest_mtime: float,
    source_hash: str,
    sources: List[Dict[str, object]],
) -> Optional[RunStatus]:
    if not reuse_existing or not record.output_path or not record.output_path.exists():
        return None

    metadata = _load_metadata(record.metadata_path)
    ratio_payload = metadata.get("ratio_definitions")
    if ratio_payload:
        record.ratio_definitions = normalize_ratio_definitions(ratio_payload)
    channel_payload = metadata.get("channel_definitions")
    if channel_payload:
        record.channel_definitions = normalize_channel_definitions(channel_payload)
    pixel_meta = metadata.get("pixel_size_um")
    if pixel_meta is not None:
        record.pixel_size_um = float(pixel_meta)
    preview_root = metadata.get("preview_root")
    if preview_root:
        preview_candidate = Path(preview_root).expanduser()
        if preview_candidate.exists():
            record.preview_root = preview_candidate.resolve()
            STATE.record_run(record)

    output_mtime = record.output_path.stat().st_mtime
    cached_mtime = metadata.get("latest_source_mtime") if metadata else None
    cached_hash = metadata.get("source_hash") if metadata else None

    latest_known = max(filter(None, [sources_latest_mtime, cached_mtime or 0.0]))
    hash_matches = cached_hash == source_hash if cached_hash else False
    mtime_valid = output_mtime >= latest_known if latest_known else False

    if hash_matches and mtime_valid:
        completed_at = utc_now()
        record.state = "succeeded"
        record.message = "Reused cached threshold results"
        record.completed_at = completed_at
        record.sources_latest_mtime = sources_latest_mtime
        record.source_hash = source_hash
        record.pixel_size_um = record.pixel_size_um
        STATE.record_run(record)
        log_event(
            LOGGER,
            "run_reused",
            job_id=record.job_id,
            study_id=record.study_name,
            project_id=record.project_id,
            session_id=record.session_id,
            state=record.state,
            output_path=str(record.output_path),
        )
        return build_run_status(record)

    if not metadata and sources_latest_mtime and output_mtime >= sources_latest_mtime:
        completed_at = utc_now()
        record.state = "succeeded"
        record.message = "Reused threshold results based on modification time"
        record.completed_at = completed_at
        record.sources_latest_mtime = sources_latest_mtime
        record.source_hash = source_hash
        STATE.record_run(record)
        _write_metadata(record.metadata_path, record, sources_latest_mtime, source_hash, sources)
        log_event(
            LOGGER,
            "run_reused",
            job_id=record.job_id,
            study_id=record.study_name,
            project_id=record.project_id,
            session_id=record.session_id,
            state=record.state,
            output_path=str(record.output_path),
        )
        return build_run_status(record)

    return None


def _fingerprint_sources(input_dir: Path, config_path: Path) -> Tuple[float, str, List[Dict[str, object]]]:
    paths: List[Path] = []
    if config_path.exists():
        paths.append(config_path)
    try:
        paths.extend(find_nd2_files(input_dir))
    except Exception:
        pass

    hasher = hashlib.sha1()
    latest_mtime = 0.0
    details: List[Dict[str, object]] = []

    for path in sorted(paths):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        latest_mtime = max(latest_mtime, stat.st_mtime)
        hasher.update(str(path).encode("utf-8"))
        hasher.update(str(int(stat.st_mtime_ns)).encode("utf-8"))
        hasher.update(str(stat.st_size).encode("utf-8"))
        details.append({"path": str(path), "mtime": stat.st_mtime, "size": stat.st_size})

    source_hash = hasher.hexdigest()
    return latest_mtime, source_hash, details


def _metadata_path(output_path: Path) -> Path:
    return Path(f"{output_path}.meta.json")


def _load_metadata(metadata_path: Optional[Path]) -> Dict[str, object]:
    if not metadata_path or not metadata_path.exists():
        return {}
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _write_metadata(
    metadata_path: Optional[Path],
    record: RunRecord,
    latest_mtime: float,
    source_hash: str,
    sources: Iterable[Dict[str, object]],
) -> None:
    if not metadata_path:
        return
    try:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": utc_now().isoformat(),
            "input_dir": str(record.input_dir) if record.input_dir else None,
            "config_path": str(record.config_path) if record.config_path else None,
            "output_path": str(record.output_path) if record.output_path else None,
            "preview_root": str(record.preview_root) if record.preview_root else None,
            "latest_source_mtime": latest_mtime,
            "source_hash": source_hash,
            "sources": list(sources),
            "ratio_definitions": record.ratio_definitions or list(DEFAULT_RATIO_DEFINITIONS),
            "channel_definitions": record.channel_definitions or list(DEFAULT_CHANNEL_DEFINITIONS),
            "pixel_size_um": record.pixel_size_um,
        }
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    except Exception:
        pass


def _as_datetime(timestamp: Optional[float]) -> Optional[datetime]:
    if not timestamp:
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)
