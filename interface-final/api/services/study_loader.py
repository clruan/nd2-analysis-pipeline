"""Study loading, rehydration, and preview-source discovery helpers."""

from __future__ import annotations

from collections import defaultdict
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import HTTPException

from threshold_analysis.batch_processor import load_threshold_results
from threshold_analysis.data_models import ThresholdResults

from data_models import GroupConfig

from ..logging_utils import log_event
from ..schemas import LoadStudyRequest
from ..state import STATE, PersistedStudyRecord, StudyRecord
from ..utils import find_nd2_files, normalize_path, slugify
from .channels import normalize_channel_definitions
from .ratios import normalize_ratio_definitions
from .study_common import (
    LOGGER,
    PREVIEW_ROOT,
    _build_subject_lookup,
    _canonical_subject_id,
    _discover_preview_plane_root,
    _get_record,
    _loose_subject_id,
    _match_subject_from_filename,
    _preview_root_has_npy_planes,
)
from image_processing import detect_pixel_size


def _retokenize_mouse_ids(results: ThresholdResults) -> None:
    if not results.group_info:
        return
    lookup = _build_subject_lookup(results.group_info)
    for entry in results.image_data:
        matched = _match_subject_from_filename(entry.filename, lookup, entry.group)
        if matched:
            subject_id, owner_group = matched
            entry.mouse_id = subject_id
            entry.group = owner_group
            continue

        canonical = _canonical_subject_id(entry.mouse_id)
        if not canonical:
            continue
        if canonical in lookup.global_exact:
            entry.mouse_id = lookup.global_exact[canonical]
            owner = lookup.owners.get(canonical)
            if owner:
                entry.group = owner
            continue
        loose = _loose_subject_id(canonical)
        if loose and loose in lookup.global_loose:
            entry.mouse_id = lookup.global_loose[loose]
            owner = lookup.loose_owners.get(loose)
            if owner:
                entry.group = owner


def _load_threshold_results_or_error(path: Path) -> ThresholdResults:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Threshold results not found: {path}")

    try:
        results = load_threshold_results(str(path))
    except KeyError as exc:
        missing_key = getattr(exc, "args", ["unknown field"])[0]
        suggestions = []
        for candidate in sorted(path.parent.glob("threshold_results_*.json")):
            try:
                load_threshold_results(str(candidate))
            except Exception:
                continue
            suggestions.append(str(candidate))

        message = (
            f"Threshold results file is missing required field '{missing_key}'. "
            "Try regenerating the study or ensure you selected the full JSON output from the pipeline."
        )
        if suggestions:
            message += f" Suggested valid files in the same folder: {', '.join(suggestions[:3])}"
        raise HTTPException(status_code=422, detail=message) from exc
    except Exception as exc:  # pragma: no cover - passthrough for unexpected formats
        raise HTTPException(
            status_code=500,
            detail=f"Unable to load threshold results: {exc}",
        ) from exc

    return results


def _validate_input_dir_override(raw_path: Optional[str]) -> Optional[Path]:
    if not raw_path:
        return None
    override_dir = normalize_path(raw_path).resolve()
    if not override_dir.exists():
        raise HTTPException(status_code=404, detail=f"Input directory does not exist: {override_dir}")
    if not override_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Input directory is not a folder: {override_dir}")
    try:
        sample_nd2 = find_nd2_files(override_dir)
    except Exception as exc:  # pragma: no cover - filesystem edge cases
        raise HTTPException(status_code=500, detail=f"Unable to scan input directory {override_dir}: {exc}") from exc
    if not sample_nd2:
        raise HTTPException(
            status_code=404,
            detail=f"No supported microscopy files found under {override_dir}. Select the folder that contains your study data.",
        )
    return override_dir


def _build_replicate_lookup(results: ThresholdResults, input_dir: Path) -> Dict[str, Dict[str, Path]]:
    lookup: Dict[str, Dict[str, Path]] = {}
    try:
        nd2_files = find_nd2_files(input_dir)
    except Exception:
        nd2_files = []

    file_map: Dict[str, List[Path]] = defaultdict(list)
    for path in nd2_files:
        file_map[path.name].append(path)

    for entry in results.image_data:
        subject_map = lookup.setdefault(entry.mouse_id, {})
        candidates = file_map.get(entry.filename)
        if candidates:
            chosen = next(
                (candidate for candidate in candidates if entry.mouse_id in candidate.parts[-2:]),
                candidates[0],
            )
            subject_map[entry.filename] = chosen
    return lookup


def _record_has_preview_sources(record: StudyRecord) -> bool:
    for subject_map in record.replicate_lookup.values():
        if subject_map:
            return True
    if record.preview_plane_root is None or not _preview_root_has_npy_planes(record.preview_plane_root):
        discovered_root = _discover_preview_plane_root(record.study_id)
        if discovered_root is not None:
            record.preview_plane_root = discovered_root
    if _preview_root_has_npy_planes(record.preview_plane_root):
        return True
    raw_dir = PREVIEW_ROOT / record.study_id / "raw"
    if raw_dir.exists():
        try:
            next(raw_dir.glob("*.png"))
            return True
        except StopIteration:
            pass
        except Exception:
            pass
    return False


def study_has_preview_sources(study_id: str) -> bool:
    record = _get_record(study_id)
    return _record_has_preview_sources(record)


def _detect_pixel_size_from_dir(input_dir: Path) -> Optional[float]:
    try:
        nd2_files = find_nd2_files(input_dir)
    except Exception:
        return None
    for path in nd2_files:
        detected = detect_pixel_size(str(path))
        if detected:
            return detected
    return None


def _materialize_study_record(
    path: Path,
    input_dir_override: Optional[Path] = None,
    persisted: Optional[PersistedStudyRecord] = None,
) -> StudyRecord:
    results = _load_threshold_results_or_error(path)
    if not results.image_data:
        raise HTTPException(
            status_code=422,
            detail="Threshold results contain no image data. Please rerun threshold generation for this study.",
        )

    _retokenize_mouse_ids(results)

    study_id = persisted.study_id if persisted else slugify(results.study_name)
    run_lookup = STATE.find_run_by_output_path(path)

    input_dir: Optional[Path] = persisted.input_dir if persisted else None
    is_3d = persisted.is_3d if persisted else True
    preview_plane_root: Optional[Path] = persisted.preview_plane_root if persisted else None
    ratio_definitions: Optional[List[Dict[str, object]]] = getattr(results, "ratio_definitions", None) or (
        persisted.ratio_definitions if persisted else None
    )
    channel_definitions: Optional[List[Dict[str, object]]] = getattr(results, "channel_definitions", None) or (
        persisted.channel_definitions if persisted else None
    )
    pixel_size_um: Optional[float] = getattr(results, "pixel_size_um", None)
    if pixel_size_um is None and persisted is not None:
        pixel_size_um = persisted.pixel_size_um

    meta: Dict[str, object] = {}
    meta_path = Path(str(path) + ".meta.json")
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}
    meta_pixel = meta.get("pixel_size_um")
    if meta_pixel is not None:
        try:
            pixel_size_um = float(meta_pixel)
        except (TypeError, ValueError):
            pixel_size_um = None

    if ratio_definitions is None and run_lookup and run_lookup.ratio_definitions:
        ratio_definitions = run_lookup.ratio_definitions
    elif ratio_definitions is None and meta.get("ratio_definitions"):
        ratio_definitions = meta.get("ratio_definitions")
    if channel_definitions is None and run_lookup and run_lookup.channel_definitions:
        channel_definitions = run_lookup.channel_definitions
    elif channel_definitions is None and meta.get("channel_definitions"):
        channel_definitions = meta.get("channel_definitions")

    if pixel_size_um is None and run_lookup and run_lookup.pixel_size_um:
        pixel_size_um = run_lookup.pixel_size_um

    if input_dir_override:
        input_dir = input_dir_override
    elif input_dir is not None:
        input_dir = input_dir
    elif run_lookup and run_lookup.input_dir:
        input_dir = run_lookup.input_dir
        if persisted is None:
            is_3d = run_lookup.is_3d
        preview_plane_root = preview_plane_root or run_lookup.preview_root
    elif meta.get("input_dir"):
        input_dir = Path(str(meta["input_dir"])).expanduser().resolve()
        preview_value = meta.get("preview_root")
        if preview_value:
            preview_candidate = Path(str(preview_value)).expanduser()
            if preview_candidate.exists():
                preview_plane_root = preview_candidate.resolve()
    else:
        candidates = [path.parent]
        if path.parent.parent:
            candidates.append(path.parent.parent)
        found: Optional[Path] = None
        for cand in candidates:
            try:
                if find_nd2_files(cand):
                    found = cand
                    break
            except Exception:
                continue
        input_dir = found or path.parent
    if input_dir_override:
        preview_plane_root = _discover_preview_plane_root(study_id)

    if preview_plane_root is not None and not _preview_root_has_npy_planes(preview_plane_root):
        discovered_root = _discover_preview_plane_root(study_id)
        if discovered_root is not None:
            preview_plane_root = discovered_root
    if preview_plane_root is None:
        preview_plane_root = _discover_preview_plane_root(study_id)

    config_candidate: Optional[Path] = None
    if run_lookup and run_lookup.config_path:
        config_candidate = run_lookup.config_path
    elif meta.get("config_path"):
        config_candidate = Path(str(meta["config_path"])).expanduser()

    if config_candidate and config_candidate.exists() and (
        ratio_definitions is None or channel_definitions is None or pixel_size_um is None
    ):
        try:
            group_config = GroupConfig.from_json(str(config_candidate))
            if ratio_definitions is None:
                ratio_definitions = group_config.ratios
            if channel_definitions is None:
                channel_definitions = group_config.channel_definitions
            if pixel_size_um is None and group_config.pixel_size_um:
                pixel_size_um = group_config.pixel_size_um
        except Exception:
            if ratio_definitions is None:
                ratio_definitions = None

    ratio_definitions = normalize_ratio_definitions(ratio_definitions)
    channel_definitions = normalize_channel_definitions(channel_definitions)
    if pixel_size_um is None and input_dir:
        pixel_size_um = _detect_pixel_size_from_dir(input_dir)
    replicate_lookup = _build_replicate_lookup(results, input_dir) if input_dir else {}

    return StudyRecord(
        study_id=study_id,
        project_id=persisted.project_id if persisted else STATE.settings.project_id,
        session_id=persisted.session_id if persisted else STATE.settings.session_id,
        results=results,
        source_path=path,
        input_dir=input_dir,
        replicate_lookup=replicate_lookup,
        is_3d=is_3d,
        preview_plane_root=preview_plane_root,
        loaded_at=persisted.loaded_at if persisted else datetime.now(timezone.utc),
        ratio_definitions=ratio_definitions,
        channel_definitions=channel_definitions,
        pixel_size_um=pixel_size_um,
    )


def load_study(request: LoadStudyRequest) -> StudyRecord:
    path = normalize_path(request.file_path).resolve()
    override_dir = _validate_input_dir_override(request.input_dir_override)
    record = _materialize_study_record(path, input_dir_override=override_dir)
    STATE.add_study(record)
    log_event(
        LOGGER,
        "study_loaded",
        study_id=record.study_id,
        project_id=record.project_id,
        session_id=record.session_id,
        source_path=str(record.source_path),
        input_dir=str(record.input_dir) if record.input_dir else None,
    )
    return record


def _rehydrate_study(persisted: PersistedStudyRecord) -> StudyRecord:
    record = _materialize_study_record(
        persisted.source_path,
        input_dir_override=None,
        persisted=persisted,
    )
    log_event(
        LOGGER,
        "study_rehydrated",
        study_id=record.study_id,
        project_id=record.project_id,
        session_id=STATE.settings.session_id,
        source_path=str(record.source_path),
    )
    return record
