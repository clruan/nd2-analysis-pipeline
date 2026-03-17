"""Study loading and analysis helpers."""

from __future__ import annotations

from collections import defaultdict, Counter
import hashlib
import json
from datetime import datetime, timedelta
import re
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, List, NamedTuple, Optional, Tuple, Set
import zipfile
import logging

import numpy as np
import pandas as pd
from fastapi import HTTPException
import matplotlib.pyplot as plt
import shutil
from PIL import Image

from threshold_analysis.batch_processor import load_threshold_results
from threshold_analysis.data_models import ThresholdData, ThresholdResults

from data_models import GroupConfig, VisualizationConfig

from ..schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    IndividualImageRecord,
    LoadStudyRequest,
    MouseAverageRecord,
    PreviewImage,
    PreviewRequest,
    PreviewResponse,
    PreviewClearResponse,
    StatisticsRequest,
    StatisticsResponse,
    DownloadResponse,
    PixelSizeUpdateRequest,
    PixelSizeUpdateResponse,
    PreviewDownloadRequest,
    PreviewDownloadResponse,
)
from ..state import STATE, StudyRecord
from ..utils import ensure_directory, find_nd2_files, normalize_path, preview_plane_filename, slugify
from .channels import (
    DEFAULT_CHANNEL_DEFINITIONS,
    channel_color_rgb,
    channel_definition_map,
    normalize_channel_definitions,
)
from .ratios import DEFAULT_RATIO_DEFINITIONS, normalize_ratio_definitions
from image_processing import load_nd2_file, detect_pixel_size
from visualization import ND2Visualizer


DOWNLOAD_ROOT = ensure_directory(Path(__file__).resolve().parent / "generated_downloads")
PREVIEW_ROOT = ensure_directory(Path(__file__).resolve().parent / "generated_previews")

LOGGER = logging.getLogger(__name__)
_PREVIEW_REVISION_LOCK = Lock()
_PREVIEW_LATEST_REVISION: Dict[str, str] = {}

CHANNEL_METRICS: Tuple[Dict[str, object], ...] = (
    {"id": "channel_1_area", "kind": "channel", "channel": 1},
    {"id": "channel_2_area", "kind": "channel", "channel": 2},
    {"id": "channel_3_area", "kind": "channel", "channel": 3},
)

SUBJECT_TOKEN_PATTERN = re.compile(r"([A-Za-z]+)(\d{1,4})")


def _channel_defs_for_record(record: StudyRecord) -> Dict[int, Dict[str, object]]:
    return channel_definition_map(record.channel_definitions)


def _channel_label(record: StudyRecord, channel: int) -> str:
    defs = _channel_defs_for_record(record)
    return str(defs.get(channel, {}).get("label", f"Channel {channel}"))


def _channel_color(record: StudyRecord, channel: int) -> str:
    defs = _channel_defs_for_record(record)
    return str(defs.get(channel, {}).get("color", "#ffffff"))


def _channel_area_label(record: StudyRecord, channel: int) -> str:
    return f"{_channel_label(record, channel)} Area (%)"


def _discover_preview_plane_root(study_id: str) -> Optional[Path]:
    study_root = PREVIEW_ROOT / study_id
    if not study_root.exists():
        return None
    direct_plane_dir = study_root / "planes"
    if direct_plane_dir.exists():
        try:
            next(direct_plane_dir.glob("*.npy"))
            return study_root
        except StopIteration:
            pass
        except Exception:
            pass

    candidates: List[Tuple[int, float, Path]] = []
    try:
        for plane_dir in study_root.rglob("planes"):
            if not plane_dir.is_dir():
                continue
            try:
                npy_count = sum(1 for _ in plane_dir.glob("*.npy"))
            except Exception:
                continue
            if npy_count <= 0:
                continue
            try:
                mtime = plane_dir.stat().st_mtime
            except Exception:
                mtime = 0.0
            candidates.append((npy_count, mtime, plane_dir.parent))
    except Exception:
        return None

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def _preview_root_has_npy_planes(preview_root: Optional[Path]) -> bool:
    if not preview_root:
        return False
    plane_dir = preview_root / "planes"
    if not plane_dir.exists():
        return False
    try:
        next(plane_dir.glob("*.npy"))
        return True
    except StopIteration:
        return False
    except Exception:
        return False


def _canonical_subject_id(value: str) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", "", str(value)).upper()


def _loose_subject_id(value: str) -> str:
    match = re.match(r"^([A-Z]+)(\d+)$", value)
    if not match:
        return value
    prefix, digits = match.groups()
    stripped = digits.lstrip("0") or "0"
    return f"{prefix}{stripped}"


class _SubjectLookup(NamedTuple):
    group_exact: Dict[str, Dict[str, str]]
    group_loose: Dict[str, Dict[str, str]]
    global_exact: Dict[str, str]
    global_loose: Dict[str, str]
    owners: Dict[str, str]
    loose_owners: Dict[str, str]


def _build_subject_lookup(group_info: Dict[str, Iterable[str]]) -> _SubjectLookup:
    group_exact: Dict[str, Dict[str, str]] = {}
    group_loose: Dict[str, Dict[str, str]] = {}
    global_exact: Dict[str, str] = {}
    global_loose: Dict[str, str] = {}
    owners: Dict[str, str] = {}
    loose_owners: Dict[str, str] = {}
    loose_conflicts: Set[str] = set()

    for group, subjects in group_info.items():
        exact_map: Dict[str, str] = {}
        loose_map: Dict[str, str] = {}
        for subject in subjects or []:
            canonical = _canonical_subject_id(subject)
            if not canonical:
                continue
            exact_map.setdefault(canonical, str(subject))
            if canonical not in global_exact:
                global_exact[canonical] = str(subject)
                owners[canonical] = group

            loose = _loose_subject_id(canonical)
            if loose and loose not in loose_map:
                loose_map[loose] = str(subject)

            if loose:
                if loose in loose_conflicts:
                    continue
                if loose in global_loose and global_loose[loose] != str(subject):
                    loose_conflicts.add(loose)
                    global_loose.pop(loose, None)
                    loose_owners.pop(loose, None)
                elif loose not in global_loose:
                    global_loose[loose] = str(subject)
                    loose_owners[loose] = group
        group_exact[group] = exact_map
        group_loose[group] = loose_map

    return _SubjectLookup(
        group_exact=group_exact,
        group_loose=group_loose,
        global_exact=global_exact,
        global_loose=global_loose,
        owners=owners,
        loose_owners=loose_owners,
    )


def _match_subject_from_filename(
    filename: str,
    lookups: _SubjectLookup,
    group: str,
) -> Optional[Tuple[str, str]]:
    best: Optional[Tuple[int, int, str, str]] = None
    exact_group = lookups.group_exact.get(group, {})
    loose_group = lookups.group_loose.get(group, {})
    for match in SUBJECT_TOKEN_PATTERN.finditer(filename):
        token = _canonical_subject_id(match.group(0))
        loose = _loose_subject_id(token)
        for key, source, owners in (
            (token, exact_group, lookups.owners),
            (token, lookups.global_exact, lookups.owners),
            (loose, loose_group, lookups.loose_owners),
            (loose, lookups.global_loose, lookups.loose_owners),
        ):
            if not key:
                continue
            candidate = source.get(key)
            if not candidate:
                continue
            owner_group = owners.get(key, group)
            digit_count = len(match.group(2))
            position = match.start()
            if best is None or digit_count > best[0] or (digit_count == best[0] and position < best[1]):
                best = (digit_count, position, candidate, owner_group)
    if best:
        return best[2], best[3]
    return None


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


def load_study(request: LoadStudyRequest) -> StudyRecord:
    path = normalize_path(request.file_path).resolve()
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Threshold results not found: {path}")

    try:
        results = load_threshold_results(str(path))
    except KeyError as exc:
        missing_key = getattr(exc, "args", ["unknown field"])[0]
        # Try to help the user by suggesting nearby candidate files
        suggestions = []
        for candidate in sorted(path.parent.glob("threshold_results_*.json")):
            try:
                _ = load_threshold_results(str(candidate))
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

    if not results.image_data:
        raise HTTPException(
            status_code=422,
            detail="Threshold results contain no image data. Please rerun threshold generation for this study.",
        )

    _retokenize_mouse_ids(results)

    study_id = slugify(results.study_name)
    run_lookup = next(
        (
            record
            for record in STATE.all_runs().values()
            if record.output_path and Path(record.output_path).resolve() == path
        ),
        None,
    )

    # Resolve source image root directory used to generate the study so we can build previews after restarts
    input_dir: Path
    is_3d: bool = True
    preview_plane_root: Optional[Path] = None
    ratio_definitions: Optional[List[Dict[str, object]]] = getattr(results, "ratio_definitions", None)
    channel_definitions: Optional[List[Dict[str, object]]] = getattr(results, "channel_definitions", None)
    pixel_size_um: Optional[float] = getattr(results, "pixel_size_um", None)

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

    override_dir: Optional[Path] = None
    if request.input_dir_override:
        override_dir = normalize_path(request.input_dir_override).resolve()
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

    if override_dir:
        input_dir = override_dir
    elif run_lookup and run_lookup.input_dir:
        input_dir = run_lookup.input_dir
        is_3d = run_lookup.is_3d
        preview_plane_root = run_lookup.preview_root
    elif meta.get("input_dir"):
        input_dir = Path(str(meta["input_dir"])).expanduser().resolve()
        preview_value = meta.get("preview_root")
        if preview_value:
            preview_candidate = Path(str(preview_value)).expanduser()
            if preview_candidate.exists():
                preview_plane_root = preview_candidate.resolve()
    else:
        # Last resort: search parents for any supported microscopy files (shallow)
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
    if override_dir:
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
    replicate_lookup = _build_replicate_lookup(results, input_dir)

    record = StudyRecord(
        study_id=study_id,
        results=results,
        source_path=path,
        input_dir=input_dir,
        replicate_lookup=replicate_lookup,
        is_3d=is_3d,
        preview_plane_root=preview_plane_root,
        ratio_definitions=ratio_definitions,
        channel_definitions=channel_definitions,
        pixel_size_um=pixel_size_um,
    )
    STATE.add_study(record)
    return record


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
            # Prefer files whose parent folder name contains the mouse_id; fall back to the first occurrence.
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


def _metric_definitions(record: StudyRecord) -> List[Dict[str, object]]:
    metrics: List[Dict[str, object]] = []
    for defn in CHANNEL_METRICS:
        metric = dict(defn)
        channel = int(metric.get("channel", 1))
        metric["label"] = _channel_area_label(record, channel)
        metrics.append(metric)
    for ratio in record.ratio_definitions:
        metrics.append(
            {
                "id": ratio["id"],
                "label": ratio["label"],
                "kind": "ratio",
                "numerator_channel": ratio["numerator_channel"],
                "denominator_channel": ratio["denominator_channel"],
            }
        )
    return metrics


def _ensure_ratio_columns(mouse_df: pd.DataFrame, ratios: List[Dict[str, object]]) -> None:
    for ratio in ratios:
        numerator_col = f"Channel_{ratio['numerator_channel']}_area"
        denominator_col = f"Channel_{ratio['denominator_channel']}_area"
        if numerator_col in mouse_df.columns and denominator_col in mouse_df.columns:
            mouse_df[ratio["id"]] = mouse_df[numerator_col] / (mouse_df[denominator_col] + 1e-3)
        else:
            mouse_df[ratio["id"]] = np.nan


def _threshold_dict(thresholds: Dict[str, int]) -> Dict[str, int]:
    defaulted = {
        "channel_1": thresholds.get("channel_1", 0),
        "channel_2": thresholds.get("channel_2", 0),
        "channel_3": thresholds.get("channel_3", 0),
    }
    return defaulted


def _get_record(study_id: str) -> StudyRecord:
    try:
        return STATE.get_study(study_id)
    except KeyError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=404, detail=f"Study not loaded: {study_id}") from exc


def _analysis_cache_key(thresholds: Dict[str, int]) -> str:
    return f"{thresholds['channel_1']}-{thresholds['channel_2']}-{thresholds['channel_3']}"


def _statistics_cache_key(thresholds: Dict[str, int], request: StatisticsRequest) -> str:
    pairs = request.comparison_pairs or []
    pairs_token = "|".join(
        "::".join(sorted((str(pair[0]), str(pair[1]))))
        for pair in sorted((pair for pair in pairs if len(pair) == 2), key=lambda pair: tuple(sorted(pair)))
    ) or "none"
    reference_group = request.reference_group or "none"
    return (
        f"{_analysis_cache_key(thresholds)}|{request.comparison_mode}|{reference_group}|"
        f"{pairs_token}|{request.test_type}|{request.significance_display}"
    )


def _analysis_tables_for_thresholds(
    record: StudyRecord,
    thresholds: Dict[str, int],
) -> Tuple[pd.DataFrame, List[IndividualImageRecord]]:
    cache_key = _analysis_cache_key(thresholds)
    cached = record.analysis_cache.get(cache_key)
    if cached:
        cached_mouse = cached.get("mouse_averages")
        cached_images = cached.get("individual_images")
        if isinstance(cached_mouse, pd.DataFrame) and isinstance(cached_images, list):
            return cached_mouse, cached_images

    mouse_averages_df = record.results.get_mouse_averages(thresholds)
    _ensure_ratio_columns(mouse_averages_df, record.ratio_definitions)
    individual_images = _collect_replicate_metrics(record.results.image_data, thresholds, record.ratio_definitions)
    record.analysis_cache[cache_key] = {
        "mouse_averages": mouse_averages_df,
        "individual_images": individual_images,
    }
    return mouse_averages_df, individual_images


def analyze_study(study_id: str, request: AnalyzeRequest) -> AnalyzeResponse:
    record = _get_record(study_id)
    thresholds = _threshold_dict(request.thresholds)
    uses_ratio_metrics = True
    mouse_averages_df, individual_images = _analysis_tables_for_thresholds(record, thresholds)

    mouse_records: List[MouseAverageRecord] = []
    for row in mouse_averages_df.to_dict(orient="records"):
        ch1 = float(row.get("Channel_1_area", 0.0))
        ch2 = float(row.get("Channel_2_area", 0.0))
        ch3 = float(row.get("Channel_3_area", 0.0))
        ratio_values: Dict[str, float] = {}
        if uses_ratio_metrics:
            for ratio in record.ratio_definitions:
                value = float(row.get(ratio["id"], 0.0))
                ratio_values[ratio["id"]] = value
        mouse_records.append(
            MouseAverageRecord(
                Group=str(row.get("Group", "")),
                MouseID=str(row.get("MouseID", "")),
                Channel_1_area=ch1,
                Channel_2_area=ch2,
                Channel_3_area=ch3,
                ratios=ratio_values,
            )
        )

    return AnalyzeResponse(
        study_id=study_id,
        thresholds=thresholds,
        mouse_averages=mouse_records,
        individual_images=individual_images,
    )


def _collect_replicate_metrics(
    image_data: Iterable[ThresholdData],
    thresholds: Dict[str, int],
    ratio_definitions: List[Dict[str, object]],
) -> List[IndividualImageRecord]:
    records: List[IndividualImageRecord] = []
    replicate_counters: Counter[Tuple[str, str]] = Counter()

    for entry in image_data:
        key = (entry.group, entry.mouse_id)
        replicate_counters[key] += 1
        replicate_index = replicate_counters[key]

        ch1 = entry.get_percentage_at_threshold(1, thresholds["channel_1"])
        ch2 = entry.get_percentage_at_threshold(2, thresholds["channel_2"])
        ch3 = entry.get_percentage_at_threshold(3, thresholds["channel_3"])
        ratio_values: Dict[str, float] = {}
        for ratio in ratio_definitions:
            num_idx = int(ratio["numerator_channel"])
            den_idx = int(ratio["denominator_channel"])
            numerator = entry.get_percentage_at_threshold(num_idx, thresholds[f"channel_{num_idx}"])
            denominator = entry.get_percentage_at_threshold(den_idx, thresholds[f"channel_{den_idx}"])
            ratio_values[ratio["id"]] = float(numerator / (denominator + 1e-3))
        records.append(
            IndividualImageRecord(
                group=str(entry.group),
                mouse_id=str(entry.mouse_id),
                filename=str(entry.filename),
                channel_1_area=float(ch1),
                channel_2_area=float(ch2),
                channel_3_area=float(ch3),
                ratios=ratio_values,
                replicate_index=replicate_index,
            )
        )
    return records


def _resolve_source_image_path(record: StudyRecord, mouse_id: str, filename: str) -> Optional[Path]:
    direct = record.replicate_lookup.get(mouse_id, {}).get(filename)
    if direct and direct.exists():
        return direct
    for subject_map in record.replicate_lookup.values():
        candidate = subject_map.get(filename)
        if candidate and candidate.exists():
            return candidate
    return None


def _entry_has_channel_source(record: StudyRecord, group: str, subject_id: str, filename: str) -> bool:
    source_path = _resolve_source_image_path(record, subject_id, filename)
    if source_path is not None and source_path.exists():
        return True

    if record.preview_plane_root is None or not _preview_root_has_npy_planes(record.preview_plane_root):
        discovered_root = _discover_preview_plane_root(record.study_id)
        if discovered_root is not None:
            record.preview_plane_root = discovered_root

    if record.preview_plane_root:
        plane_dir = record.preview_plane_root / "planes"
        if plane_dir.exists():
            for channel_index in (1, 2, 3):
                plane_path = plane_dir / preview_plane_filename(group, subject_id, filename, channel_index)
                if plane_path.exists():
                    return True

    raw_dir = PREVIEW_ROOT / record.study_id / "raw"
    if raw_dir.exists():
        safe_base = f"{slugify(group)}_{slugify(subject_id)}_{slugify(filename)}"
        for channel_index in (1, 2, 3):
            if (raw_dir / f"{safe_base}_raw_ch{channel_index}.png").exists():
                return True
    return False


def _apply_optional_denoise(channel: np.ndarray, sigma: Optional[float]) -> np.ndarray:
    if sigma is None:
        return channel.astype(np.float32, copy=False)
    try:
        from scipy.ndimage import gaussian_filter
    except Exception:
        return channel.astype(np.float32, copy=False)
    return gaussian_filter(channel.astype(np.float32, copy=False), sigma=sigma)


def _positive_intensity(channel: np.ndarray, threshold: int) -> float:
    if channel is None:
        return 0.0
    array = np.asarray(channel, dtype=np.float32)
    positive = array[array > float(threshold)]
    if positive.size == 0:
        return 0.0
    return float(np.sum(positive))


def _colocalization_intensity_metric(
    channel_a: np.ndarray, channel_b: np.ndarray, threshold_a: int, threshold_b: int
) -> float:
    a = np.asarray(channel_a, dtype=np.float32)
    b = np.asarray(channel_b, dtype=np.float32)
    mask = (a > float(threshold_a)) | (b > float(threshold_b))
    if not np.any(mask):
        return 0.0
    values_a = a[mask].ravel()
    values_b = b[mask].ravel()
    if values_a.size < 3:
        return 0.0
    std_a = float(np.std(values_a))
    std_b = float(np.std(values_b))
    if std_a < 1e-8 or std_b < 1e-8:
        return 0.0
    corr = float(np.corrcoef(values_a, values_b)[0, 1])
    if not np.isfinite(corr):
        return 0.0
    return corr


def _colocalization_overlap_metric(
    channel_a: np.ndarray, channel_b: np.ndarray, threshold_a: int, threshold_b: int
) -> float:
    a = np.asarray(channel_a)
    b = np.asarray(channel_b)
    mask_a = a > float(threshold_a)
    mask_b = b > float(threshold_b)
    union = np.logical_or(mask_a, mask_b)
    union_count = int(np.sum(union))
    if union_count == 0:
        return 0.0
    intersection_count = int(np.sum(np.logical_and(mask_a, mask_b)))
    return float((intersection_count / union_count) * 100.0)


def _collect_replicate_mode_metrics(
    record: StudyRecord,
    thresholds: Dict[str, int],
    ratio_definitions: List[Dict[str, object]],
    analysis_mode: str,
    denoise_sigma: Optional[float],
) -> List[IndividualImageRecord]:
    records: List[IndividualImageRecord] = []
    replicate_counters: Counter[Tuple[str, str]] = Counter()
    stack_modes = {"positive_intensity_stack_sum", "coloc_intensity_3d", "coloc_overlap_3d"}
    projection = "none" if analysis_mode in stack_modes else "max"
    uses_ratio_metrics = analysis_mode.startswith("positive_")

    for entry in record.results.image_data:
        source_path = _resolve_source_image_path(record, entry.mouse_id, entry.filename)
        if source_path is None:
            continue

        try:
            ch1, ch2, ch3 = load_nd2_file(str(source_path), is_3d=record.is_3d, projection=projection)
        except Exception:
            continue

        channels = {
            1: _apply_optional_denoise(np.asarray(ch1), denoise_sigma),
            2: _apply_optional_denoise(np.asarray(ch2), denoise_sigma),
            3: _apply_optional_denoise(np.asarray(ch3), denoise_sigma),
        }

        if analysis_mode.startswith("positive_intensity_"):
            channel_values = {
                1: _positive_intensity(channels[1], thresholds["channel_1"]),
                2: _positive_intensity(channels[2], thresholds["channel_2"]),
                3: _positive_intensity(channels[3], thresholds["channel_3"]),
            }
        elif analysis_mode.startswith("coloc_intensity_"):
            channel_values = {
                1: _colocalization_intensity_metric(channels[1], channels[2], thresholds["channel_1"], thresholds["channel_2"]),
                2: _colocalization_intensity_metric(channels[1], channels[3], thresholds["channel_1"], thresholds["channel_3"]),
                3: _colocalization_intensity_metric(channels[2], channels[3], thresholds["channel_2"], thresholds["channel_3"]),
            }
        elif analysis_mode.startswith("coloc_overlap_"):
            channel_values = {
                1: _colocalization_overlap_metric(channels[1], channels[2], thresholds["channel_1"], thresholds["channel_2"]),
                2: _colocalization_overlap_metric(channels[1], channels[3], thresholds["channel_1"], thresholds["channel_3"]),
                3: _colocalization_overlap_metric(channels[2], channels[3], thresholds["channel_2"], thresholds["channel_3"]),
            }
        else:
            channel_values = {
                1: float(entry.get_percentage_at_threshold(1, thresholds["channel_1"])),
                2: float(entry.get_percentage_at_threshold(2, thresholds["channel_2"])),
                3: float(entry.get_percentage_at_threshold(3, thresholds["channel_3"])),
            }

        ratio_values: Dict[str, float] = {}
        if uses_ratio_metrics:
            for ratio in ratio_definitions:
                num_idx = int(ratio["numerator_channel"])
                den_idx = int(ratio["denominator_channel"])
                numerator = channel_values.get(num_idx, 0.0)
                denominator = channel_values.get(den_idx, 0.0)
                ratio_values[ratio["id"]] = float(numerator / (denominator + 1e-6))

        key = (entry.group, entry.mouse_id)
        replicate_counters[key] += 1
        replicate_index = replicate_counters[key]
        records.append(
            IndividualImageRecord(
                group=str(entry.group),
                mouse_id=str(entry.mouse_id),
                filename=str(entry.filename),
                channel_1_area=float(channel_values.get(1, 0.0)),
                channel_2_area=float(channel_values.get(2, 0.0)),
                channel_3_area=float(channel_values.get(3, 0.0)),
                ratios=ratio_values,
                replicate_index=replicate_index,
            )
        )

    return records


def perform_statistics(study_id: str, request: StatisticsRequest) -> StatisticsResponse:
    record = _get_record(study_id)
    thresholds = _threshold_dict(request.thresholds)
    cache_key = _statistics_cache_key(thresholds, request)
    cached = record.statistics_cache.get(cache_key)
    if isinstance(cached, StatisticsResponse):
        return cached
    uses_ratio_metrics = True
    mouse_averages_df, _ = _analysis_tables_for_thresholds(record, thresholds)

    channels = [
        ("channel_1", "Channel_1_area"),
        ("channel_2", "Channel_2_area"),
        ("channel_3", "Channel_3_area"),
    ]

    statistics = {}
    for channel_key, column in channels:
        groups_data = {
            group: group_df[column].dropna().tolist()
            for group, group_df in mouse_averages_df.groupby("Group")
        }

        statistics[channel_key] = _analyze_groups(
            groups_data,
            request.comparison_mode,
            request.reference_group,
            request.comparison_pairs,
            request.test_type,
            request.significance_display,
        )

    if uses_ratio_metrics:
        for ratio in record.ratio_definitions:
            column = ratio["id"]
            groups_data = {
                group: group_df[column].dropna().tolist()
                for group, group_df in mouse_averages_df.groupby("Group")
            }
            statistics[column] = _analyze_groups(
                groups_data,
                request.comparison_mode,
                request.reference_group,
                request.comparison_pairs,
                request.test_type,
                request.significance_display,
            )

    response = StatisticsResponse(
        statistics=statistics,
        thresholds=thresholds,
        test_type_used=request.test_type,
        significance_display=request.significance_display,
        ratios=record.ratio_definitions,
    )
    record.statistics_cache[cache_key] = response
    return response


def _analyze_groups(
    groups_data: Dict[str, List[float]],
    comparison_mode: str,
    reference_group: Optional[str],
    comparison_pairs: Optional[List[List[str]]],
    test_type: str,
    significance_display: str,
) -> Dict[str, object]:
    cleaned: Dict[str, List[float]] = {
        group: [value for value in values if np.isfinite(value)]
        for group, values in groups_data.items()
    }
    cleaned = {group: values for group, values in cleaned.items() if values}

    if len(cleaned) < 2:
        return {
            "comparison_mode": comparison_mode,
            "pairwise_comparisons": [],
            "note": "Not enough samples per group to compute statistics.",
        }

    if comparison_mode == "pairs":
        if not comparison_pairs:
            raise HTTPException(status_code=400, detail="Comparison pairs required for pairs mode")
        comparisons = []
        for pair in comparison_pairs:
            if len(pair) != 2:
                continue
            g1, g2 = pair
            if g1 not in cleaned or g2 not in cleaned:
                continue
            statistic, p_value = _perform_statistical_test(cleaned[g1], cleaned[g2], test_type)
            comparisons.append(
                {
                    "group1": g1,
                    "group2": g2,
                    "statistic": statistic,
                    "p_value": p_value,
                    "significance": _format_significance(p_value, significance_display),
                }
            )
        return {
            "comparison_mode": comparison_mode,
            "pairwise_comparisons": comparisons,
            "note": "Comparisons skipped for groups without samples." if not comparisons else None,
        }

    if comparison_mode == "all_pairs":
        group_names = sorted(cleaned.keys())
        comparisons = []
        for idx, group_a in enumerate(group_names):
            for group_b in group_names[idx + 1 :]:
                statistic, p_value = _perform_statistical_test(cleaned[group_a], cleaned[group_b], test_type)
                comparisons.append(
                    {
                        "group1": group_a,
                        "group2": group_b,
                        "statistic": statistic,
                        "p_value": p_value,
                        "significance": _format_significance(p_value, significance_display),
                    }
                )

        overall_stat, overall_p = _perform_anova(cleaned, test_type)

        overall_block = None
        if test_type != "t_test":
            overall_block = {
                "statistic": overall_stat,
                "p_value": overall_p,
                "significance": _format_significance(overall_p, significance_display),
            }

        return {
            "comparison_mode": comparison_mode,
            "overall_test": overall_block,
            "pairwise_comparisons": comparisons,
            "note": "Comparisons skipped for groups without samples." if not comparisons else None,
        }

    reference = reference_group or next(iter(cleaned))
    if reference not in cleaned:
        reference = next(iter(cleaned))

    comparisons = []
    for group_name, data in cleaned.items():
        if group_name == reference:
            continue
        statistic, p_value = _perform_statistical_test(cleaned[reference], data, test_type)
        comparisons.append(
            {
                "group1": reference,
                "group2": group_name,
                "statistic": statistic,
                "p_value": p_value,
                "significance": _format_significance(p_value, significance_display),
            }
        )

    overall_stat, overall_p = _perform_anova(cleaned, test_type)

    overall_block = None
    if test_type != "t_test":
        overall_block = {
            "statistic": overall_stat,
            "p_value": overall_p,
            "significance": _format_significance(overall_p, significance_display),
        }

    return {
        "comparison_mode": comparison_mode,
        "reference_group": reference,
        "overall_test": overall_block,
        "pairwise_comparisons": comparisons,
    }


def _perform_statistical_test(group1: List[float], group2: List[float], test_type: str) -> Tuple[float, float]:
    from scipy import stats

    group1 = [value for value in group1 if np.isfinite(value)]
    group2 = [value for value in group2 if np.isfinite(value)]

    if len(group1) < 2 or len(group2) < 2:
        return 0.0, 1.0

    if not group1 or not group2:
        return 0.0, 1.0

    if test_type in {"anova_parametric", "t_test"}:
        statistic, p_value = stats.ttest_ind(group1, group2, equal_var=False)
    elif test_type == "anova_non_parametric":
        statistic, p_value = stats.mannwhitneyu(group1, group2, alternative="two-sided")
    else:
        use_parametric = _is_normal(group1) and _is_normal(group2)
        if use_parametric:
            statistic, p_value = stats.ttest_ind(group1, group2, equal_var=False)
        else:
            statistic, p_value = stats.mannwhitneyu(group1, group2, alternative="two-sided")

    return float(statistic), float(p_value)


def _perform_anova(groups_data: Dict[str, List[float]], test_type: str) -> Tuple[float, float]:
    from scipy import stats

    clean_groups = [
        [value for value in values if np.isfinite(value)]
        for values in groups_data.values()
        if values
    ]
    clean_groups = [group for group in clean_groups if len(group) >= 2]
    if len(clean_groups) < 2:
        return 0.0, 1.0

    if test_type == "t_test":
        return 0.0, 1.0
    if test_type == "anova_parametric":
        statistic, p_value = stats.f_oneway(*clean_groups)
    elif test_type == "anova_non_parametric":
        statistic, p_value = stats.kruskal(*clean_groups)
    else:
        all_normal = all(_is_normal(group) for group in clean_groups)
        if all_normal:
            statistic, p_value = stats.f_oneway(*clean_groups)
        else:
            statistic, p_value = stats.kruskal(*clean_groups)

    return float(statistic), float(p_value)


def _is_normal(data: Iterable[float]) -> bool:
    from scipy import stats

    data = [value for value in data if np.isfinite(value)]
    if len(data) < 3:
        return True
    _, p_value = stats.shapiro(data)
    return p_value > 0.05


def _format_significance(p_value: float, mode: str) -> str:
    if mode == "p_values":
        return f"p={p_value:.4f}"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


ALL_PREVIEW_METRICS: Tuple[str, ...] = (
    "channel_1_area",
    "channel_2_area",
    "channel_3_area",
    "channel_1_3_ratio",
    "channel_2_3_ratio",
)

DEFAULT_PREVIEW_METRIC = ALL_PREVIEW_METRICS[0]
DEFAULT_PANEL_ORDER: Tuple[str, ...] = ("channel_1", "channel_2", "channel_3", "composite")

GRAPH_PAD_EXPORT_KEYS: Tuple[str, ...] = ("Channel_1_area", "Channel_2_area", "Channel_3_area")


class PreviewVariant(NamedTuple):
    variant: str
    channels: Tuple[int, ...]
    cacheable: bool


def _normalize_metric(metric: Optional[str]) -> str:
    if metric in ALL_PREVIEW_METRICS:
        return metric
    return DEFAULT_PREVIEW_METRIC


def _normalize_metrics(
    metrics: Optional[Iterable[str]],
    fallback: Optional[str] = None,
    available: Optional[Iterable[str]] = None,
) -> List[str]:
    allowed = list(available) if available else list(ALL_PREVIEW_METRICS)
    if metrics:
        ordered: List[str] = []
        for metric in metrics:
            if metric in allowed and metric not in ordered:
                ordered.append(metric)
        if ordered:
            return ordered
    if fallback:
        normalized = _normalize_metric(fallback)
        return [normalized]
    return allowed


def _preview_variants_for_metric(metric: Dict[str, object]) -> List[PreviewVariant]:
    if metric.get("kind") == "channel":
        channel = int(metric.get("channel", 1))
        return [
            PreviewVariant("raw", (channel,), True),
            PreviewVariant("mask", (channel,), False),
            PreviewVariant("overlay", (channel,), False),
        ]

    numerator = int(metric.get("numerator_channel", 1))
    denominator = int(metric.get("denominator_channel", 3))
    return [
        PreviewVariant("raw", (numerator,), True),
        PreviewVariant("raw", (denominator,), True),
        PreviewVariant("mask", (numerator,), False),
        PreviewVariant("mask", (denominator,), False),
        PreviewVariant("overlay", (numerator, denominator), False),
    ]


def _sanitize_panel_order(order: Optional[Iterable[str]]) -> List[str]:
    if not order:
        return list(DEFAULT_PANEL_ORDER)
    normalized: List[str] = []
    for panel in order:
        if panel in DEFAULT_PANEL_ORDER and panel not in normalized:
            normalized.append(panel)
    return normalized or list(DEFAULT_PANEL_ORDER)

def _normalize_channel_ranges(payload: Optional[Dict[str, object]]) -> Dict[int, Tuple[float, float]]:
    """Convert client-provided channel range payload into numeric overrides."""
    if not payload:
        return {}
    normalized: Dict[int, Tuple[float, float]] = {}

    def _extract_bounds(candidate: object) -> Optional[Tuple[float, float]]:
        # Accept plain dicts, pydantic models, or lightweight objects with vmin/vmax attributes.
        raw_min = None
        raw_max = None
        if isinstance(candidate, dict):
            raw_min = candidate.get("vmin")
            raw_max = candidate.get("vmax")
        else:
            raw_min = getattr(candidate, "vmin", None)
            raw_max = getattr(candidate, "vmax", None)
            if raw_min is None and hasattr(candidate, "model_dump"):
                try:
                    data = candidate.model_dump()
                except Exception:
                    data = {}
                raw_min = data.get("vmin", raw_min)
                raw_max = data.get("vmax", raw_max)

        if raw_min is None and raw_max is None:
            return None

        vmin = float(raw_min if raw_min is not None else 0)
        vmax = float(raw_max if raw_max is not None else vmin + 1)
        if vmax <= vmin:
            vmax = vmin + 1
        return max(0.0, vmin), max(vmax, vmin + 1e-3)

    for key, values in payload.items():
        channel_match = re.match(r"channel_(\d+)", key)
        if not channel_match:
            continue
        bounds = _extract_bounds(values)
        if bounds is None:
            continue
        channel_index = int(channel_match.group(1))
        normalized[channel_index] = bounds
    return normalized


def _normalize_channel_ids(channel_ids: Optional[Iterable[int]] = None) -> Tuple[int, ...]:
    if channel_ids is None:
        return (1, 2, 3)
    normalized = sorted({int(channel_id) for channel_id in channel_ids if int(channel_id) in {1, 2, 3}})
    return tuple(normalized) or (1, 2, 3)


def _channel_range_token(
    channel_ranges: Dict[int, Tuple[float, float]],
    channel_ids: Optional[Iterable[int]] = None,
) -> str:
    active_channels = _normalize_channel_ids(channel_ids)
    parts = []
    for channel_index in active_channels:
        bounds = channel_ranges.get(channel_index)
        if bounds is None:
            continue
        vmin, vmax = bounds
        parts.append(f"{channel_index}:{int(vmin)}-{int(vmax)}")
    if not parts:
        return "default"
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:10]
    return f"rng-{digest}"


def _channel_color_token(
    channel_definitions: List[Dict[str, object]],
    channel_ids: Optional[Iterable[int]] = None,
) -> str:
    active_channels = _normalize_channel_ids(channel_ids)
    normalized = normalize_channel_definitions(channel_definitions)
    defaults = normalize_channel_definitions(DEFAULT_CHANNEL_DEFINITIONS)
    normalized_map = {int(entry["channel"]): entry for entry in normalized}
    default_map = {int(entry["channel"]): entry for entry in defaults}
    parts = []
    for channel_index in active_channels:
        current = normalized_map.get(channel_index)
        default = default_map.get(channel_index)
        if current is None or default is None:
            continue
        if current == default:
            continue
        parts.append(f"{channel_index}:{current['color']}")
    if not parts:
        return "default"
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:10]
    return f"clr-{digest}"


def _preview_style_token(
    channel_ranges: Dict[int, Tuple[float, float]],
    channel_definitions: List[Dict[str, object]],
    channel_ids: Optional[Iterable[int]] = None,
) -> str:
    range_token = _channel_range_token(channel_ranges, channel_ids)
    color_token = _channel_color_token(channel_definitions, channel_ids)
    if range_token == "default" and color_token == "default":
        return "default"
    digest = hashlib.sha1(f"{range_token}|{color_token}".encode("utf-8")).hexdigest()[:10]
    return f"style-{digest}"


def _threshold_dependency_token(
    thresholds: Dict[str, int],
    channel_ids: Optional[Iterable[int]] = None,
) -> str:
    active_channels = _normalize_channel_ids(channel_ids)
    if not active_channels:
        return "default"
    parts = [f"{channel_index}:{thresholds.get(f'channel_{channel_index}', 0)}" for channel_index in active_channels]
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:10]
    return f"thr-{digest}"


def _preview_dependency_token(
    thresholds: Dict[str, int],
    channel_ranges: Dict[int, Tuple[float, float]],
    channel_definitions: List[Dict[str, object]],
    variant: PreviewVariant,
) -> str:
    active_channels = variant.channels or _normalize_channel_ids()
    tokens: List[str] = []
    if variant.variant in {"mask", "overlay"}:
        tokens.append(_threshold_dependency_token(thresholds, active_channels))
    if variant.variant in {"raw", "overlay"}:
        tokens.append(_channel_range_token(channel_ranges, active_channels))
    tokens.append(_channel_color_token(channel_definitions, active_channels))
    non_default = [token for token in tokens if token != "default"]
    if not non_default:
        return "default"
    digest = hashlib.sha1("|".join(non_default).encode("utf-8")).hexdigest()[:10]
    return f"dep-{digest}"


def _visualizer_range_payload(channel_ranges: Dict[int, Tuple[float, float]]) -> Dict[str, Dict[str, float]]:
    payload: Dict[str, Dict[str, float]] = {}
    for channel_index, (vmin, vmax) in channel_ranges.items():
        payload[f"channel_{channel_index}"] = {"vmin": vmin, "vmax": vmax}
    return payload


def _build_variant_cache_key(
    cache_base: str,
    variant: PreviewVariant,
    metric: str,
    dependency_token: str,
) -> str:
    channel_tag = "-".join(str(ch) for ch in variant.channels) or "all"
    token = dependency_token or "default"
    if variant.cacheable:
        return f"{cache_base}|{variant.variant}|{channel_tag}|{token}"
    return f"{cache_base}|{metric}|{variant.variant}|{channel_tag}|{token}"


def _build_variant_filename(
    safe_base: str,
    variant: PreviewVariant,
    metric_slug: str,
    cacheable: bool,
    dependency_token: str,
) -> str:
    channel_tag = "-".join(f"ch{ch}" for ch in variant.channels) or "all"
    parts = [safe_base, variant.variant, channel_tag]
    if not cacheable:
        parts.append(metric_slug)
    if dependency_token and dependency_token != "default":
        parts.append(dependency_token)
    return "_".join(parts) + ".png"


def _is_valid_preview_file(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with Image.open(path) as candidate:
            return max(candidate.size) <= 320
    except Exception:
        return False


def _metadata_path_for(image_path: Path) -> Path:
    return Path(str(image_path) + ".meta.json")


def _write_preview_metadata(image_path: Path, metric_id: str, variant: PreviewVariant, dependency_token: str) -> None:
    metadata = {
        "variant": variant.variant,
        "channels": list(variant.channels),
        "cache_scope": "global" if variant.cacheable else metric_id,
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "dependency_token": dependency_token or "default",
    }
    meta_path = _metadata_path_for(image_path)
    try:
        meta_path.write_text(json.dumps(metadata))
    except Exception:
        # Metadata is advisory; ignore failures but ensure stale files are not reused later.
        if meta_path.exists():
            meta_path.unlink(missing_ok=True)


def _metadata_matches(image_path: Path, metric_id: str, variant: PreviewVariant, dependency_token: str) -> bool:
    meta_path = _metadata_path_for(image_path)
    if not meta_path.exists():
        return False
    try:
        data = json.loads(meta_path.read_text())
    except Exception:
        return False
    expected_scope = "global" if variant.cacheable else metric_id
    if data.get("variant") != variant.variant:
        return False
    if data.get("channels") != list(variant.channels):
        return False
    if data.get("cache_scope") != expected_scope:
        return False
    stored_token = data.get("dependency_token", data.get("range_token", "default"))
    return stored_token == (dependency_token or "default")


def _cached_variant_valid(path: Path, metric_id: str, variant: PreviewVariant, dependency_token: str) -> bool:
    path = Path(path)
    if not _is_valid_preview_file(path):
        return False
    return _metadata_matches(path, metric_id, variant, dependency_token)


def _preview_cache_token(path: Path) -> str:
    try:
        return str(path.stat().st_mtime_ns)
    except OSError:
        return "0"


def _prune_preview_thresholds(preview_dir: Path, retain_keys: Set[str], max_sets: int = 5, max_age_days: int = 7) -> None:
    try:
        candidates = [
            child
            for child in preview_dir.iterdir()
            if child.is_dir()
            and child.name not in {"raw", "planes"}
            and child.name not in retain_keys
            and not _preview_root_has_npy_planes(child)
        ]
    except FileNotFoundError:
        return

    now = datetime.utcnow()
    max_age = timedelta(days=max_age_days)
    scored: List[Tuple[datetime, Path]] = []
    for child in candidates:
        try:
            mtime = datetime.fromtimestamp(child.stat().st_mtime)
        except OSError:
            continue
        scored.append((mtime, child))

    scored.sort(key=lambda item: item[0], reverse=True)
    slots = max(max_sets - len(retain_keys), 0)
    for index, (mtime, path) in enumerate(scored):
        if index >= slots or (now - mtime) > max_age:
            shutil.rmtree(path, ignore_errors=True)


def clear_preview_cache(study_id: str, scope: str = "thresholds", threshold_key: Optional[str] = None) -> PreviewClearResponse:
    record = _get_record(study_id)
    preview_dir = ensure_directory(PREVIEW_ROOT / study_id)
    allowed_scopes = {"thresholds", "all"}
    if scope not in allowed_scopes:
        raise HTTPException(status_code=400, detail=f"Invalid scope '{scope}'. Expected one of {sorted(allowed_scopes)}.")

    def removable_children(filter_raw: bool) -> List[Path]:
        targets: List[Path] = []
        try:
            for child in preview_dir.iterdir():
                if not child.is_dir():
                    continue
                if not filter_raw or child.name not in {"raw", "planes"}:
                    targets.append(child)
        except FileNotFoundError:
            return []
        return targets

    removed: List[str] = []

    if scope == "thresholds":
        targets: List[Path]
        if threshold_key:
            candidate = preview_dir / threshold_key
            targets = [candidate] if candidate.exists() and candidate.is_dir() else []
        else:
            targets = removable_children(filter_raw=True)
        for path in targets:
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path.name)
    else:  # scope == "all"
        targets = removable_children(filter_raw=False)
        for path in targets:
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path.name)
        record.preview_cache.clear()
        record.raw_cache.clear()
        record.preview_plane_root = None

    return PreviewClearResponse(removed_directories=removed)


def _set_latest_preview_revision(study_id: str, revision_key: str) -> None:
    with _PREVIEW_REVISION_LOCK:
        _PREVIEW_LATEST_REVISION[study_id] = revision_key


def _is_latest_preview_revision(study_id: str, revision_key: str) -> bool:
    with _PREVIEW_REVISION_LOCK:
        return _PREVIEW_LATEST_REVISION.get(study_id) == revision_key


def generate_previews(study_id: str, request: PreviewRequest) -> PreviewResponse:
    record = _get_record(study_id)
    thresholds = _threshold_dict(request.thresholds)
    metric_defs = _metric_definitions(record)
    metric_map = {metric["id"]: metric for metric in metric_defs}

    preview_dir = ensure_directory(PREVIEW_ROOT / study_id)
    raw_dir = ensure_directory(preview_dir / "raw")
    if record.preview_plane_root is None or not _preview_root_has_npy_planes(record.preview_plane_root):
        discovered_root = _discover_preview_plane_root(study_id)
        if discovered_root is not None:
            record.preview_plane_root = discovered_root
        elif record.preview_plane_root is None:
            record.preview_plane_root = preview_dir

    threshold_key = f"{thresholds['channel_1']}-{thresholds['channel_2']}-{thresholds['channel_3']}"
    threshold_dir = ensure_directory(preview_dir / threshold_key)
    _prune_preview_thresholds(preview_dir, retain_keys={threshold_key})
    available_metric_ids = [metric["id"] for metric in metric_defs]
    metric_candidates = _normalize_metrics(request.metrics, request.metric, available_metric_ids)
    metric_ids = [metric for metric in metric_candidates if metric in metric_map] or available_metric_ids
    metric_slugs = {metric_id: slugify(metric_id) for metric_id in metric_ids}
    metric_dirs = {metric_id: ensure_directory(threshold_dir / metric_slugs[metric_id]) for metric_id in metric_ids}
    variant_plans = {metric_id: _preview_variants_for_metric(metric_map[metric_id]) for metric_id in metric_ids}

    groups_requested = set(request.groups) if request.groups else None
    channel_ranges = _normalize_channel_ranges(request.channel_ranges)
    style_token = _preview_style_token(channel_ranges, record.channel_definitions)
    revision_key = request.revision_key or f"{threshold_key}|{style_token}"
    _set_latest_preview_revision(study_id, revision_key)
    group_counts: Dict[str, int] = defaultdict(int)
    group_items_seen: Dict[str, Set[str]] = defaultdict(set)
    per_group_limits: Dict[str, int] = {
        group: max(1, min(20, int(value)))
        for group, value in (request.group_sample_limits or {}).items()
        if isinstance(value, int)
    }
    max_samples = max(1, min(20, request.max_samples_per_group))
    nd2_available = _record_has_preview_sources(record)
    preview_images: List[PreviewImage] = []
    priority_keys: Set[Tuple[str, str, str]] = set()
    priority_entries: List[ThresholdData] = []
    if request.priority_subjects:
        entry_lookup: Dict[Tuple[str, str, str], ThresholdData] = {
            (entry.group, entry.mouse_id, entry.filename): entry for entry in record.results.image_data
        }
        for subject in request.priority_subjects:
            key = (subject.group, subject.subject_id, subject.filename)
            if key in priority_keys:
                continue
            match = entry_lookup.get(key)
            if match is None:
                continue
            priority_keys.add(key)
            priority_entries.append(match)
    ordered_entries: List[ThresholdData] = priority_entries + [
        entry for entry in record.results.image_data if (entry.group, entry.mouse_id, entry.filename) not in priority_keys
    ]
    stale_request = False
    for entry in ordered_entries:
        if not _is_latest_preview_revision(study_id, revision_key):
            stale_request = True
            break
        if groups_requested and entry.group not in groups_requested:
            continue
        limit = per_group_limits.get(entry.group, max_samples)
        item_key = f"{entry.mouse_id}|{entry.filename}"
        items_seen = group_items_seen[entry.group]
        if item_key in items_seen:
            continue
        if len(items_seen) >= limit:
            continue

        cache_base = f"{entry.group}|{entry.mouse_id}|{entry.filename}"
        safe_base = f"{slugify(entry.group)}_{slugify(entry.mouse_id)}_{slugify(entry.filename)}"
        channel_arrays: Optional[Dict[int, np.ndarray]] = None
        generated_for_entry = False

        for metric_id in metric_ids:
            if not _is_latest_preview_revision(study_id, revision_key):
                stale_request = True
                break
            variant_plan = variant_plans[metric_id]
            metric_dir = metric_dirs[metric_id]
            metric_slug = metric_slugs[metric_id]
            existing_files: Dict[str, Path] = {}

            for variant in variant_plan:
                if not _is_latest_preview_revision(study_id, revision_key):
                    stale_request = True
                    break
                cacheable_variant = variant.cacheable
                dependency_token = _preview_dependency_token(
                    thresholds,
                    channel_ranges,
                    record.channel_definitions,
                    variant,
                )
                cache_key = _build_variant_cache_key(cache_base, variant, metric_id, dependency_token)
                output_dir = raw_dir if cacheable_variant else metric_dir
                safe_name = _build_variant_filename(safe_base, variant, metric_slug, cacheable_variant, dependency_token)
                image_path = output_dir / safe_name

                cached_path = record.preview_cache.get(cache_key)
                if cached_path and _cached_variant_valid(cached_path, metric_id, variant, dependency_token):
                    existing_files[cache_key] = cached_path
                    continue

                if _cached_variant_valid(image_path, metric_id, variant, dependency_token):
                    existing_files[cache_key] = image_path
                    record.preview_cache[cache_key] = image_path
                    continue

                if channel_arrays is None:
                    allow_source_read = not bool(request.prefer_generated_assets)
                    channel_arrays = _load_channels(
                        record,
                        entry.group,
                        entry.mouse_id,
                        entry.filename,
                        allow_png_fallback=True,
                        allow_source_read=allow_source_read,
                    )
                if not channel_arrays:
                    continue

                image_array = _render_preview_variant(
                    channel_arrays,
                    thresholds,
                    variant,
                    channel_ranges,
                    record.channel_definitions,
                )
                if image_array is None:
                    continue

                _write_image(image_array, image_path)
                _write_preview_metadata(image_path, metric_id, variant, dependency_token)
                record.preview_cache[cache_key] = image_path
                existing_files[cache_key] = image_path
            if stale_request:
                break

            if not existing_files:
                continue

            for variant in variant_plan:
                dependency_token = _preview_dependency_token(
                    thresholds,
                    channel_ranges,
                    record.channel_definitions,
                    variant,
                )
                cache_key = _build_variant_cache_key(cache_base, variant, metric_id, dependency_token)
                image_path = existing_files.get(cache_key)
                if not image_path:
                    continue
                channel = variant.channels[0] if len(variant.channels) == 1 else None
                preview_images.append(
                    PreviewImage(
                        metric=metric_id,
                        variant=variant.variant,
                        channel=channel,
                        group=entry.group,
                        subject_id=entry.mouse_id,
                        filename=entry.filename,
                        image_path=str(image_path),
                        cache_token=_preview_cache_token(image_path),
                    )
                )
                generated_for_entry = True
        if stale_request:
            break

        if generated_for_entry:
            group_items_seen[entry.group].add(item_key)
            group_counts[entry.group] = len(group_items_seen[entry.group])

    if stale_request:
        LOGGER.debug("Cancelled stale preview generation for study %s revision %s", study_id, revision_key)

    metric_rank = {metric["id"]: index for index, metric in enumerate(metric_defs)}
    preview_images.sort(
        key=lambda item: (
            metric_rank.get(item.metric, len(metric_rank)),
            item.group,
            item.subject_id,
            item.variant,
            item.channel or 0,
        )
    )

    group_sample_counts = dict(group_counts)
    nd2_available = nd2_available or bool(preview_images)

    return PreviewResponse(
        study_id=study_id,
        generated_at=datetime.utcnow(),
        images=preview_images,
        nd2_available=nd2_available,
        nd2_source=str(record.input_dir),
        max_samples_per_group=max_samples,
        ratio_definitions=record.ratio_definitions,
        group_sample_counts=group_sample_counts,
    )


def render_preview_panel(study_id: str, request: PreviewDownloadRequest) -> PreviewDownloadResponse:
    record = _get_record(study_id)
    thresholds = _threshold_dict(request.thresholds)
    panel_order = _sanitize_panel_order(request.panel_order)
    channel_ranges = _normalize_channel_ranges(request.channel_ranges)
    visualizer_ranges = _visualizer_range_payload(channel_ranges)

    channel_arrays = _load_channels(
        record,
        request.group,
        request.subject_id,
        request.filename,
        allow_png_fallback=False,
    )
    if not channel_arrays:
        raise HTTPException(status_code=404, detail="Source image data unavailable for the requested preview.")

    def _project(channel_id: int) -> Optional[np.ndarray]:
        array = channel_arrays.get(channel_id)
        if array is None:
            return None
        if array.ndim == 2:
            return array
        return np.max(array, axis=0)

    channel_1 = _project(1)
    channel_2 = _project(2)
    channel_3 = _project(3)
    if channel_1 is None or channel_2 is None or channel_3 is None:
        raise HTTPException(status_code=422, detail="One or more channels are missing for this preview.")

    panel_dir = ensure_directory(PREVIEW_ROOT / study_id / "custom_panels")
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    file_name = f"{slugify(request.group)}_{slugify(request.subject_id)}_{timestamp}.png"
    target_path = panel_dir / file_name

    requested_scale_bar = request.scale_bar_um
    add_scale_bar = requested_scale_bar is None or requested_scale_bar > 0
    vis_config = VisualizationConfig(scale_bar_um=requested_scale_bar or 50)
    visualizer = ND2Visualizer(vis_config, pixel_size_um=record.pixel_size_um, channel_definitions=record.channel_definitions)
    figure = visualizer.visualize_channels(
        channel_1,
        channel_2,
        channel_3,
        save_path=str(target_path),
        add_scale_bar=add_scale_bar,
        panel_order=panel_order,
        channel_ranges=visualizer_ranges,
    )
    plt.close(figure)

    return PreviewDownloadResponse(
        image_path=str(target_path),
        panel_order=panel_order,
    )


def list_ratio_definitions(study_id: str) -> List[Dict[str, object]]:
    record = _get_record(study_id)
    return record.ratio_definitions


def list_channel_definitions(study_id: str) -> List[Dict[str, object]]:
    record = _get_record(study_id)
    return record.channel_definitions


def update_ratio_definitions(study_id: str, ratios: List[Dict[str, object]]) -> List[Dict[str, object]]:
    record = _get_record(study_id)
    normalized = normalize_ratio_definitions(ratios)
    record.ratio_definitions = normalized
    record.results.ratio_definitions = normalized
    record.preview_cache.clear()
    record.analysis_cache.clear()
    record.statistics_cache.clear()
    _persist_study_annotations(record)
    return normalized


def update_channel_definitions(study_id: str, channels: List[Dict[str, object]]) -> List[Dict[str, object]]:
    record = _get_record(study_id)
    normalized = normalize_channel_definitions(channels)
    record.channel_definitions = normalized
    record.results.channel_definitions = normalized
    record.preview_cache.clear()
    _persist_study_annotations(record)
    return normalized


def update_pixel_size(study_id: str, pixel_size_um: Optional[float]) -> Optional[float]:
    record = _get_record(study_id)
    if pixel_size_um is not None and pixel_size_um <= 0:
        raise HTTPException(status_code=400, detail="Pixel size must be positive.")
    record.pixel_size_um = float(pixel_size_um) if pixel_size_um else None
    record.results.pixel_size_um = record.pixel_size_um
    _persist_study_annotations(record)
    return record.pixel_size_um


def _load_json_payload(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _write_json_payload(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _persist_study_annotations(record: StudyRecord) -> None:
    _persist_result_annotations(record)
    _persist_study_metadata(record)


def _persist_result_annotations(record: StudyRecord) -> None:
    result_path = Path(record.source_path)
    try:
        payload = _load_json_payload(result_path)
        if not payload:
            return
        payload["ratio_definitions"] = record.ratio_definitions
        payload["channel_definitions"] = record.channel_definitions
        payload["pixel_size_um"] = record.pixel_size_um
        _write_json_payload(result_path, payload)
    except Exception:
        pass


def _persist_study_metadata(record: StudyRecord) -> None:
    meta_path = Path(str(record.source_path) + ".meta.json")
    try:
        payload = _load_json_payload(meta_path)
        if not payload:
            payload = {}
        payload["ratio_definitions"] = record.ratio_definitions
        payload["channel_definitions"] = record.channel_definitions
        payload["pixel_size_um"] = record.pixel_size_um
        if record.input_dir:
            payload.setdefault("input_dir", str(record.input_dir))
        if record.preview_plane_root:
            payload.setdefault("preview_root", str(record.preview_plane_root))
        _write_json_payload(meta_path, payload)
    except Exception:
        pass


def _graphpad_group_order(mouse_df: pd.DataFrame) -> List[str]:
    if mouse_df.empty or "Group" not in mouse_df:
        return []
    order: List[str] = []
    for value in mouse_df["Group"]:
        group = str(value)
        if group not in order:
            order.append(group)
    return order


def _build_graphpad_block(
    mouse_df: pd.DataFrame, metric_key: str, label: str, group_order: List[str]
) -> Tuple[Optional[pd.DataFrame], Dict[str, int]]:
    if metric_key not in mouse_df.columns or not group_order:
        return None, {}
    columns: Dict[str, List[float]] = {}
    counts: Dict[str, int] = {}
    max_len = 0
    for group in group_order:
        mask = mouse_df["Group"] == group
        values = mouse_df.loc[mask, metric_key].dropna().tolist()
        column_name = f"{label} - {group}"
        columns[column_name] = values
        counts[column_name] = len(values)
        max_len = max(max_len, len(values))
    if not columns:
        return None, {}
    if max_len == 0:
        max_len = 1
    padded = {name: values + [np.nan] * (max_len - len(values)) for name, values in columns.items()}
    return pd.DataFrame(padded), counts


def _build_graphpad_dataframe(
    mouse_df: pd.DataFrame,
    ratio_defs: List[Dict[str, object]],
    channel_defs: List[Dict[str, object]],
) -> pd.DataFrame:
    if mouse_df.empty:
        return pd.DataFrame()
    group_order = _graphpad_group_order(mouse_df)
    if not group_order:
        return pd.DataFrame()
    blocks: List[pd.DataFrame] = []
    counts: Dict[str, int] = {}
    channel_map = channel_definition_map(channel_defs)
    for metric_key in GRAPH_PAD_EXPORT_KEYS:
        channel_index = int(metric_key.split("_")[1])
        label = str(channel_map.get(channel_index, {}).get("label", f"Channel {channel_index}"))
        block, block_counts = _build_graphpad_block(mouse_df, metric_key, label, group_order)
        if block is None:
            continue
        blocks.append(block)
        counts.update(block_counts)
    for ratio in ratio_defs:
        column = ratio["id"]
        block, block_counts = _build_graphpad_block(mouse_df, column, ratio["label"], group_order)
        if block is None:
            continue
        blocks.append(block)
        counts.update(block_counts)
    if not blocks:
        return pd.DataFrame()
    combined = pd.concat(blocks, axis=1)
    combined.reset_index(drop=True, inplace=True)
    count_row = pd.DataFrame([{col: counts.get(col, 0) for col in combined.columns}])
    graphpad_df = pd.concat([count_row, combined], ignore_index=True)
    index_labels = ["n"] + [str(i) for i in range(1, len(graphpad_df))]
    graphpad_df.index = index_labels
    return graphpad_df


def _format_replicates_dataframe(
    individual_df: pd.DataFrame,
    ratios: List[Dict[str, object]],
    channel_defs: List[Dict[str, object]],
) -> pd.DataFrame:
    if individual_df.empty:
        return individual_df

    table = individual_df.copy()
    channel_map = channel_definition_map(channel_defs)
    ch1_label = str(channel_map.get(1, {}).get("label", "Channel 1"))
    ch2_label = str(channel_map.get(2, {}).get("label", "Channel 2"))
    ch3_label = str(channel_map.get(3, {}).get("label", "Channel 3"))
    rename_map = {
        "group": "Group",
        "mouse_id": "Mouse ID",
        "replicate_index": "Replicate #",
        "filename": "Filename",
        "channel_1_area": f"{ch1_label} Area (%)",
        "channel_2_area": f"{ch2_label} Area (%)",
        "channel_3_area": f"{ch3_label} Area (%)",
        "channel_1_3_ratio": "Channel 1 / Channel 3",
        "channel_2_3_ratio": "Channel 2 / Channel 3",
    }
    ratio_labels: List[str] = []
    for ratio in ratios:
        ratio_id = ratio.get("id")
        if not ratio_id:
            continue
        label = ratio.get("label") or ratio_id
        rename_map[ratio_id] = label
        ratio_labels.append(label)

    table.rename(columns=rename_map, inplace=True)
    ordered_columns = [
        "Group",
        "Mouse ID",
        "Replicate #",
        "Filename",
        f"{ch1_label} Area (%)",
        f"{ch2_label} Area (%)",
        f"{ch3_label} Area (%)",
    ]
    ordered_columns.extend(ratio_labels)
    existing_columns = [column for column in ordered_columns if column in table.columns]
    table = table[existing_columns]
    metric_columns = [
        column
        for column in existing_columns
        if column.startswith("Channel") or column in ratio_labels
    ]
    for column in metric_columns:
        table[column] = table[column].astype(float).round(4)
    sort_columns = [column for column in ("Group", "Mouse ID", "Replicate #") if column in table.columns]
    if sort_columns:
        table.sort_values(sort_columns, inplace=True, kind="mergesort")
    table.reset_index(drop=True, inplace=True)
    return table


def generate_downloads(study_id: str, thresholds: Dict[str, int]) -> DownloadResponse:
    record = _get_record(study_id)
    thresholds = _threshold_dict(thresholds)

    mouse_averages_df = record.results.get_mouse_averages(thresholds)
    _ensure_ratio_columns(mouse_averages_df, record.ratio_definitions)
    individual_images = pd.DataFrame(
        _collect_replicate_metrics(record.results.image_data, thresholds, record.ratio_definitions)
    )
    if not individual_images.empty and "ratios" in individual_images.columns:
        ratio_values = individual_images["ratios"].apply(pd.Series)
        individual_images = pd.concat([individual_images.drop(columns=["ratios"]), ratio_values], axis=1)

    download_dir = ensure_directory(DOWNLOAD_ROOT / study_id)
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    excel_path = download_dir / f"{study_id}_thresholds_{timestamp}.xlsx"

    graphpad_df = _build_graphpad_dataframe(mouse_averages_df, record.ratio_definitions, record.channel_definitions)
    replicates_df = _format_replicates_dataframe(individual_images, record.ratio_definitions, record.channel_definitions)

    try:
        _write_excel_workbook(excel_path, graphpad_df, mouse_averages_df, replicates_df)
        final_path = excel_path
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.warning("Excel export failed (%s). Falling back to CSV bundle.", exc)
        final_path = _write_csv_bundle(
            download_dir, study_id, timestamp, graphpad_df, mouse_averages_df, replicates_df
        )

    return DownloadResponse(download_path=str(final_path), generated_at=datetime.utcnow())


def _write_excel_workbook(
    excel_path: Path, graphpad_df: pd.DataFrame, mouse_df: pd.DataFrame, replicates_df: pd.DataFrame
) -> None:
    excel_engine: Optional[str]
    try:  # pragma: no cover - optional dependency
        import openpyxl  # type: ignore  # noqa: F401

        excel_engine = "openpyxl"
    except Exception:  # pragma: no cover - optional dependency
        try:
            import xlsxwriter  # type: ignore  # noqa: F401

            excel_engine = "xlsxwriter"
        except Exception:
            excel_engine = None

    try:
        writer_factory = pd.ExcelWriter(excel_path, engine=excel_engine) if excel_engine else pd.ExcelWriter(excel_path)
    except ValueError as exc:  # pragma: no cover - optional dependency
        raise HTTPException(
            status_code=500,
            detail="Excel export requires openpyxl or xlsxwriter. Install one of these packages and restart the API.",
        ) from exc

    with writer_factory as writer:
        if not graphpad_df.empty:
            graphpad_df.to_excel(writer, sheet_name="GraphPad Data", index_label="Row")
        mouse_df.to_excel(writer, sheet_name="Mouse Averages", index=False)
        replicates_df.to_excel(writer, sheet_name="Replicates", index=False)
        if excel_engine == "openpyxl":
            replicates_sheet = writer.sheets.get("Replicates")
            if replicates_sheet is not None:
                try:
                    replicates_sheet.freeze_panes = replicates_sheet["E2"]
                except Exception:
                    pass


def _write_csv_bundle(
    base_dir: Path,
    study_id: str,
    timestamp: str,
    graphpad_df: pd.DataFrame,
    mouse_df: pd.DataFrame,
    replicates_df: pd.DataFrame,
) -> Path:
    zip_path = base_dir / f"{study_id}_thresholds_{timestamp}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if not graphpad_df.empty:
            archive.writestr("graphpad_data.csv", graphpad_df.to_csv(index=True))
        archive.writestr("mouse_averages.csv", mouse_df.to_csv(index=False))
        archive.writestr("replicates.csv", replicates_df.to_csv(index=False))
    return zip_path


def resolve_preview_path(study_id: str, file_path: str) -> Path:
    _get_record(study_id)
    target = Path(file_path).expanduser().resolve()
    allowed_root = ensure_directory(PREVIEW_ROOT / study_id)
    if not str(target).startswith(str(allowed_root)):
        raise HTTPException(status_code=403, detail="Preview path is outside of the study directory")
    if not target.exists():
        raise HTTPException(status_code=404, detail="Preview image not found")
    return target


def resolve_download_path(study_id: str, file_path: str) -> Path:
    _get_record(study_id)
    target = Path(file_path).expanduser().resolve()
    allowed_root = ensure_directory(DOWNLOAD_ROOT / study_id)
    if not str(target).startswith(str(allowed_root)):
        raise HTTPException(status_code=403, detail="Download path is outside of the study directory")
    if not target.exists():
        raise HTTPException(status_code=404, detail="Download file not found")
    return target


def _load_channels(
    record: StudyRecord,
    group: str,
    subject_id: str,
    filename: str,
    *,
    allow_png_fallback: bool = True,
    allow_source_read: bool = True,
) -> Optional[Dict[int, np.ndarray]]:
    cache_key = f"{group}|{subject_id}|{filename}"
    cached_source = record.raw_cache_source.get(cache_key)
    if cache_key in record.raw_cache:
        # When a LUT override is requested, refuse to reuse PNG-sourced cache (clipped to 8-bit).
        if not allow_png_fallback and cached_source not in {"nd2", "planes"}:
            record.raw_cache.pop(cache_key, None)
            record.raw_cache_source.pop(cache_key, None)
        else:
            return record.raw_cache[cache_key]

    channel_arrays: Dict[int, np.ndarray] = {}
    if record.preview_plane_root is None or not _preview_root_has_npy_planes(record.preview_plane_root):
        discovered_root = _discover_preview_plane_root(record.study_id)
        if discovered_root is not None:
            record.preview_plane_root = discovered_root

    if record.preview_plane_root:
        plane_dir = record.preview_plane_root / "planes"
        for channel_index in (1, 2, 3):
            plane_path = plane_dir / preview_plane_filename(group, subject_id, filename, channel_index)
            if not plane_path.exists():
                channel_arrays.clear()
                break
            try:
                array = np.load(plane_path, allow_pickle=False)
                channel_arrays[channel_index] = array.astype(np.uint16, copy=False)
            except Exception:
                channel_arrays.clear()
                break

    source_tag = "nd2"
    if not channel_arrays and allow_png_fallback:
        channel_arrays = _load_channels_from_cached_png(record, group, subject_id, filename)
        if channel_arrays:
            _persist_preview_planes(record, group, subject_id, filename, channel_arrays)
            source_tag = "png"
    if not channel_arrays and allow_source_read:
        subject_path: Optional[Path] = None
        for subject_map in record.replicate_lookup.values():
            if filename in subject_map:
                subject_path = subject_map[filename]
                break
        if subject_path and subject_path.exists():
            try:
                ch1, ch2, ch3 = load_nd2_file(str(subject_path), is_3d=record.is_3d)
                channel_arrays = {1: ch1, 2: ch2, 3: ch3}
                _persist_preview_planes(record, group, subject_id, filename, channel_arrays)
                source_tag = "nd2"
            except Exception:
                channel_arrays = {}
    elif channel_arrays and source_tag != "png":
        source_tag = "planes" if record.preview_plane_root else "nd2"

    if not channel_arrays:
        return None

    record.raw_cache[cache_key] = channel_arrays
    record.raw_cache_source[cache_key] = source_tag
    return channel_arrays


def _persist_preview_planes(
    record: StudyRecord, group: str, subject_id: str, filename: str, channel_arrays: Dict[int, np.ndarray]
) -> None:
    if not record.preview_plane_root:
        return
    try:
        plane_dir = ensure_directory(record.preview_plane_root / "planes")
    except Exception:
        return

    for channel_index, array in channel_arrays.items():
        plane_path = plane_dir / preview_plane_filename(group, subject_id, filename, channel_index)
        if plane_path.exists():
            continue
        try:
            np.save(plane_path, array.astype(np.uint16, copy=False), allow_pickle=False)
        except Exception:
            continue


def _load_channels_from_cached_png(
    record: StudyRecord, group: str, subject_id: str, filename: str
) -> Optional[Dict[int, np.ndarray]]:
    preview_dir = PREVIEW_ROOT / record.study_id
    raw_dir = preview_dir / "raw"
    if not raw_dir.exists():
        return None

    safe_base = f"{slugify(group)}_{slugify(subject_id)}_{slugify(filename)}"
    channel_arrays: Dict[int, np.ndarray] = {}

    for channel_index in (1, 2, 3):
        image_name = f"{safe_base}_raw_ch{channel_index}.png"
        image_path = raw_dir / image_name
        if not image_path.exists():
            continue
        try:
            with Image.open(image_path) as image:
                grayscale = image.convert("L")
                array = np.asarray(grayscale, dtype=np.float32)
                # Raw previews are scaled to 0-255; re-expand to 16-bit space so thresholds still apply.
                rescaled = (array / 255.0) * 4095.0
                channel_arrays[channel_index] = rescaled.astype(np.uint16)
        except Exception:
            continue

    return channel_arrays or None


def _normalize_channel(channel: np.ndarray, range_override: Optional[Tuple[float, float]] = None) -> np.ndarray:
    channel = channel.astype(np.float32)
    channel = np.clip(channel, 0, None)
    if range_override:
        vmin, vmax = range_override
        channel = np.clip(channel, vmin, vmax)
        channel -= vmin
        denom = max(vmax - vmin, 1e-6)
    else:
        channel -= channel.min()
        denom = channel.max()
    if denom <= 0:
        return np.zeros_like(channel, dtype=np.uint8)
    channel /= denom
    channel *= 255.0
    return channel.astype(np.uint8)


def _generate_raw_image(
    channels: Dict[int, np.ndarray],
    channel_ranges: Dict[int, Tuple[float, float]],
    channel_definitions: List[Dict[str, object]],
) -> Optional[np.ndarray]:
    sample = next((array for array in channels.values() if array is not None), None)
    if sample is None:
        return None
    rgb = np.zeros((*sample.shape, 3), dtype=np.float32)
    color_map = channel_definition_map(channel_definitions)
    for channel_id in (1, 2, 3):
        source = channels.get(channel_id)
        if source is None:
            continue
        normalized = _normalize_channel(source, channel_ranges.get(channel_id)).astype(np.float32) / 255.0
        channel_color = str(color_map.get(channel_id, {}).get("color", "#ffffff"))
        red, green, blue = channel_color_rgb(channel_color)
        rgb[..., 0] += normalized * red
        rgb[..., 1] += normalized * green
        rgb[..., 2] += normalized * blue
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def _build_binary_mask(
    channels: Dict[int, np.ndarray], thresholds: Dict[str, int], channel_id: int
) -> Optional[np.ndarray]:
    channel = channels.get(channel_id)
    threshold = thresholds.get(f"channel_{channel_id}")
    if channel is None or threshold is None:
        return None
    return (channel > threshold).astype(np.uint8)


def _generate_channel_raw_image(
    channels: Dict[int, np.ndarray],
    channel_id: int,
    channel_ranges: Dict[int, Tuple[float, float]],
    channel_definitions: List[Dict[str, object]],
) -> Optional[np.ndarray]:
    channel = channels.get(channel_id)
    if channel is None:
        return None
    normalized = _normalize_channel(channel, channel_ranges.get(channel_id)).astype(np.float32) / 255.0
    color_map = channel_definition_map(channel_definitions)
    channel_color = str(color_map.get(channel_id, {}).get("color", "#ffffff"))
    red, green, blue = channel_color_rgb(channel_color)
    rgb = np.zeros((*normalized.shape, 3), dtype=np.float32)
    rgb[..., 0] = normalized * red
    rgb[..., 1] = normalized * green
    rgb[..., 2] = normalized * blue
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def _generate_channel_mask_image(
    channels: Dict[int, np.ndarray],
    thresholds: Dict[str, int],
    channel_id: int,
    channel_definitions: List[Dict[str, object]],
) -> Optional[np.ndarray]:
    binary = _build_binary_mask(channels, thresholds, channel_id)
    if binary is None:
        return None
    color_map = channel_definition_map(channel_definitions)
    channel_color = str(color_map.get(channel_id, {}).get("color", "#ffffff"))
    red, green, blue = channel_color_rgb(channel_color)
    mask = binary.astype(np.float32)
    rgb = np.zeros((*mask.shape, 3), dtype=np.float32)
    rgb[..., 0] = mask * red
    rgb[..., 1] = mask * green
    rgb[..., 2] = mask * blue
    return (rgb * 255.0).astype(np.uint8)


def _generate_mask_image(
    channels: Dict[int, np.ndarray],
    thresholds: Dict[str, int],
    channel_definitions: List[Dict[str, object]],
) -> Optional[np.ndarray]:
    masks = {channel_id: _build_binary_mask(channels, thresholds, channel_id) for channel_id in (1, 2, 3)}
    valid_masks = [mask for mask in masks.values() if mask is not None]
    if not valid_masks:
        sample = next(iter(channels.values()), None)
        if sample is None:
            return None
        shape = sample.shape
        return np.zeros((*shape, 3), dtype=np.uint8)
    color_map = channel_definition_map(channel_definitions)
    rgb = np.zeros((*valid_masks[0].shape, 3), dtype=np.float32)
    for channel_id in (1, 2, 3):
        mask = masks.get(channel_id)
        if mask is None:
            continue
        channel_color = str(color_map.get(channel_id, {}).get("color", "#ffffff"))
        red, green, blue = channel_color_rgb(channel_color)
        binary = mask.astype(np.float32)
        rgb[..., 0] += binary * red
        rgb[..., 1] += binary * green
        rgb[..., 2] += binary * blue
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def _apply_highlight(raw: np.ndarray, mask: np.ndarray, strength: float = 0.45) -> np.ndarray:
    base = raw.astype(np.float32)
    alpha = (mask[..., :1].astype(np.float32) / 255.0) * strength
    highlighted = np.clip(base * (1.0 - alpha) + 255.0 * alpha, 0, 255)
    result = np.where(mask[..., :1] > 0, highlighted, base)
    return result.astype(np.uint8)


def _apply_white_mask(raw: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    """Cheap overlay: masked pixels become white; others keep the raw color."""
    if raw.ndim != 3 or raw.shape[-1] != 3:
        raise ValueError("Expected RGB uint8 image for raw overlay")
    if binary_mask.ndim != 2:
        raise ValueError("Expected 2D mask")
    overlay = raw.copy()
    overlay[binary_mask.astype(bool)] = 255
    return overlay


def _generate_channel_overlay_image(
    channels: Dict[int, np.ndarray],
    thresholds: Dict[str, int],
    channel_id: int,
    channel_ranges: Dict[int, Tuple[float, float]],
    channel_definitions: List[Dict[str, object]],
) -> Optional[np.ndarray]:
    raw = _generate_channel_raw_image(channels, channel_id, channel_ranges, channel_definitions)
    if raw is None:
        return None
    binary = _build_binary_mask(channels, thresholds, channel_id)
    if binary is None:
        return raw
    return _apply_white_mask(raw, binary)


def _generate_ratio_overlay_image(
    channels: Dict[int, np.ndarray],
    thresholds: Dict[str, int],
    channel_pair: Tuple[int, ...],
    channel_ranges: Dict[int, Tuple[float, float]],
    channel_definitions: List[Dict[str, object]],
) -> Optional[np.ndarray]:
    if len(channel_pair) != 2:
        return None
    raw_images: List[np.ndarray] = []
    for channel_id in channel_pair:
        raw_image = _generate_channel_raw_image(channels, channel_id, channel_ranges, channel_definitions)
        if raw_image is None:
            return None
        raw_images.append(raw_image.astype(np.float32))
    stack = np.stack(raw_images, axis=0)
    base = np.max(stack, axis=0).astype(np.uint8)
    masks = [_build_binary_mask(channels, thresholds, channel_id) for channel_id in channel_pair]
    masks = [mask for mask in masks if mask is not None]
    if not masks:
        return base
    combined = np.maximum.reduce(masks)
    return _apply_white_mask(base, combined)


def _generate_overlay_image(
    channels: Dict[int, np.ndarray],
    thresholds: Dict[str, int],
    channel_ranges: Dict[int, Tuple[float, float]],
    channel_definitions: List[Dict[str, object]],
) -> Optional[np.ndarray]:
    raw = _generate_raw_image(channels, channel_ranges, channel_definitions)
    if raw is None:
        return None
    masks = [_build_binary_mask(channels, thresholds, channel_id) for channel_id in (1, 2, 3)]
    masks = [mask for mask in masks if mask is not None]
    if not masks:
        return raw
    combined = np.maximum.reduce(masks)
    return _apply_white_mask(raw, combined)


def _render_preview_variant(
    channels: Dict[int, np.ndarray],
    thresholds: Dict[str, int],
    variant: PreviewVariant,
    channel_ranges: Dict[int, Tuple[float, float]],
    channel_definitions: List[Dict[str, object]],
) -> Optional[np.ndarray]:
    if variant.variant == "raw":
        if len(variant.channels) == 1:
            return _generate_channel_raw_image(channels, variant.channels[0], channel_ranges, channel_definitions)
        return _generate_raw_image(channels, channel_ranges, channel_definitions)
    if variant.variant == "mask":
        if len(variant.channels) == 1:
            return _generate_channel_mask_image(channels, thresholds, variant.channels[0], channel_definitions)
        return _generate_mask_image(channels, thresholds, channel_definitions)
    if variant.variant == "overlay":
        if len(variant.channels) == 1:
            return _generate_channel_overlay_image(channels, thresholds, variant.channels[0], channel_ranges, channel_definitions)
        if len(variant.channels) == 2:
            return _generate_ratio_overlay_image(channels, thresholds, variant.channels, channel_ranges, channel_definitions)
        return _generate_overlay_image(channels, thresholds, channel_ranges, channel_definitions)
    return None


def _write_image(image_array: np.ndarray, path: Path) -> None:
    image = Image.fromarray(image_array)
    max_dim = max(image.size)
    if max_dim > 320:
        scale = 320 / max_dim
        new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    image.save(path)
