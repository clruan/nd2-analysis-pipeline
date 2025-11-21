"""Utilities for launching threshold generation runs."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import shutil
import numpy as np
from fastapi import BackgroundTasks, HTTPException

from image_processing import load_nd2_file
from threshold_analysis.batch_processor import process_directory_all_thresholds
from threshold_analysis.data_models import ThresholdResults

from data_models import GroupConfig

from ..schemas import RunStatus, ThresholdRunRequest
from ..state import RunRecord, STATE
from ..utils import ensure_directory, find_nd2_files, normalize_path, preview_plane_filename, slugify
from .ratios import DEFAULT_RATIO_DEFINITIONS, normalize_ratio_definitions
from .studies import PREVIEW_ROOT


RUN_OUTPUT_ROOT = ensure_directory(Path(__file__).resolve().parent / "generated_results")


def launch_threshold_run(payload: ThresholdRunRequest, background: BackgroundTasks) -> RunStatus:
    input_dir = normalize_path(payload.input_dir).resolve()
    config_path = normalize_path(payload.config_path).resolve()

    if not input_dir.exists():
        raise HTTPException(status_code=404, detail=f"Input directory does not exist: {input_dir}")
    if not config_path.exists():
        raise HTTPException(status_code=404, detail=f"Config file not found: {config_path}")

    ratio_definitions = list(DEFAULT_RATIO_DEFINITIONS)
    try:
        group_config = GroupConfig.from_json(str(config_path))
        ratio_definitions = normalize_ratio_definitions(group_config.ratios)
        pixel_size_um = group_config.pixel_size_um
    except Exception:
        # Fall back to defaults when the config cannot be parsed.
        group_config = None
        pixel_size_um = None

    job_id = uuid.uuid4().hex
    study_slug = slugify(input_dir.name)
    default_output = RUN_OUTPUT_ROOT / f"threshold_results_{study_slug}.json"
    output_path = normalize_path(payload.output_path).resolve() if payload.output_path else default_output
    ensure_directory(output_path.parent)

    sources_latest_mtime, source_hash, sources = _fingerprint_sources(input_dir, config_path)
    metadata_path = _metadata_path(output_path)
    record = RunRecord(
        job_id=job_id,
        state="queued",
        input_dir=input_dir,
        config_path=config_path,
        output_path=output_path,
        study_name=study_slug,
        is_3d=payload.is_3d,
        sources_latest_mtime=sources_latest_mtime,
        source_hash=source_hash,
        metadata_path=metadata_path,
        ratio_definitions=ratio_definitions,
        pixel_size_um=pixel_size_um,
    )
    STATE.record_run(record)

    reuse_result = _maybe_reuse_existing(payload.reuse_existing, record, sources_latest_mtime, source_hash, sources)
    if reuse_result:
        return reuse_result

    study_preview_dir = ensure_directory(PREVIEW_ROOT / study_slug)
    preview_root = study_preview_dir / job_id
    if preview_root.exists():
        shutil.rmtree(preview_root)
    ensure_directory(preview_root)
    record.preview_root = preview_root.resolve()
    STATE.update_run(job_id, preview_root=record.preview_root)

    def _worker() -> None:
        STATE.update_run(job_id, state="running", message="Processing ND2 files")
        try:
            results = process_directory_all_thresholds(
                input_dir=str(record.input_dir),
                config_path=str(record.config_path),
                output_file=str(record.output_path),
                is_3d=payload.is_3d,
                marker=payload.marker,
                n_jobs=payload.n_jobs,
                max_threshold=payload.max_threshold,
                save_intermediate=True,
            )
        except Exception as exc:  # pragma: no cover
            STATE.update_run(job_id, state="failed", message=str(exc), completed_at=datetime.utcnow())
            return

        cached_preview_root = _cache_run_previews(record, results)
        if cached_preview_root is not None:
            record.preview_root = cached_preview_root
            STATE.update_run(job_id, preview_root=record.preview_root)

        final_mtime, final_hash, final_sources = _fingerprint_sources(record.input_dir, record.config_path)
        _write_metadata(record.metadata_path, record, final_mtime, final_hash, final_sources)
        STATE.update_run(
            job_id,
            state="succeeded",
            message="Threshold results generated",
            completed_at=datetime.utcnow(),
            sources_latest_mtime=final_mtime,
            source_hash=final_hash,
            preview_root=record.preview_root,
        )

    background.add_task(_worker)

    return RunStatus(
        job_id=job_id,
        state="queued",
        message="Run scheduled",
        input_dir=str(record.input_dir),
        config_path=str(record.config_path),
        output_path=str(record.output_path),
        study_name=record.study_name,
        started_at=record.started_at,
        latest_source_mtime=_as_datetime(record.sources_latest_mtime),
        source_hash=record.source_hash,
    )


def describe_run(job_id: str) -> RunStatus:
    try:
        record = STATE.get_run(job_id)
    except KeyError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=404, detail=f"Run not found: {job_id}") from exc
    return RunStatus(
        job_id=job_id,
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
    )


def _cache_run_previews(record: RunRecord, results: ThresholdResults) -> Optional[Path]:
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
    pixel_meta = metadata.get("pixel_size_um")
    if pixel_meta is not None:
        record.pixel_size_um = float(pixel_meta)
    preview_root = metadata.get("preview_root")
    if preview_root:
        preview_candidate = Path(preview_root).expanduser()
        if preview_candidate.exists():
            record.preview_root = preview_candidate.resolve()
            STATE.update_run(record.job_id, preview_root=record.preview_root)
    output_mtime = record.output_path.stat().st_mtime
    cached_mtime = metadata.get("latest_source_mtime") if metadata else None
    cached_hash = metadata.get("source_hash") if metadata else None

    latest_known = max(filter(None, [sources_latest_mtime, cached_mtime or 0.0]))
    hash_matches = cached_hash == source_hash if cached_hash else False
    mtime_valid = output_mtime >= latest_known if latest_known else False

    if hash_matches and mtime_valid:
        completed_at = datetime.utcnow()
        STATE.update_run(
            record.job_id,
            state="succeeded",
            message="Reused cached threshold results",
            completed_at=completed_at,
            sources_latest_mtime=sources_latest_mtime,
            source_hash=source_hash,
            pixel_size_um=record.pixel_size_um,
        )
        return RunStatus(
            job_id=record.job_id,
            state="succeeded",
            message="Reused cached threshold results",
            input_dir=str(record.input_dir),
            config_path=str(record.config_path),
            output_path=str(record.output_path),
            study_name=record.study_name,
            started_at=record.started_at,
            completed_at=completed_at,
            latest_source_mtime=_as_datetime(sources_latest_mtime),
            source_hash=source_hash,
        )

    if not metadata and sources_latest_mtime and output_mtime >= sources_latest_mtime:
        completed_at = datetime.utcnow()
        STATE.update_run(
            record.job_id,
            state="succeeded",
            message="Reused threshold results based on modification time",
            completed_at=completed_at,
            sources_latest_mtime=sources_latest_mtime,
            source_hash=source_hash,
        )
        _write_metadata(record.metadata_path, record, sources_latest_mtime, source_hash, sources)
        return RunStatus(
            job_id=record.job_id,
            state="succeeded",
            message="Reused threshold results based on modification time",
            input_dir=str(record.input_dir),
            config_path=str(record.config_path),
            output_path=str(record.output_path),
            study_name=record.study_name,
            started_at=record.started_at,
            completed_at=completed_at,
            latest_source_mtime=_as_datetime(sources_latest_mtime),
            source_hash=source_hash,
        )

    return None


def _fingerprint_sources(input_dir: Path, config_path: Path) -> Tuple[float, str, List[Dict[str, object]]]:
    paths: List[Path] = []
    if config_path.exists():
        paths.append(config_path)
    try:
        paths.extend(find_nd2_files(input_dir))
    except Exception:
        # If traversal fails we proceed with the config file only.
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
    except Exception:  # pragma: no cover - corrupted metadata
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
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "input_dir": str(record.input_dir) if record.input_dir else None,
            "config_path": str(record.config_path) if record.config_path else None,
            "output_path": str(record.output_path) if record.output_path else None,
            "preview_root": str(record.preview_root) if record.preview_root else None,
            "latest_source_mtime": latest_mtime,
            "source_hash": source_hash,
            "sources": list(sources),
            "ratio_definitions": record.ratio_definitions or list(DEFAULT_RATIO_DEFINITIONS),
            "pixel_size_um": record.pixel_size_um,
        }
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    except Exception:  # pragma: no cover - metadata failures should not break runs
        pass


def _as_datetime(timestamp: Optional[float]) -> Optional[datetime]:
    if not timestamp:
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)
