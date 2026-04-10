"""Preview generation, rendering, and file resolution helpers."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Set, Tuple
import shutil

import matplotlib.pyplot as plt
import numpy as np
from fastapi import HTTPException
from PIL import Image

from data_models import VisualizationConfig

from ..schemas import (
    PreviewClearResponse,
    PreviewDownloadRequest,
    PreviewDownloadResponse,
    PreviewImage,
    PreviewRequest,
    PreviewResponse,
)
from ..utils import ensure_directory, preview_plane_filename, slugify
from .analysis_service import _resolve_source_image_path
from .preview_rendering import (
    _build_variant_cache_key,
    _build_variant_filename,
    _cached_variant_valid,
    _normalize_channel_ranges,
    _normalize_metrics,
    _preview_cache_token,
    _preview_dependency_token,
    _preview_style_token,
    _preview_variants_for_metric,
    _render_preview_variant,
    _visualizer_range_payload,
    _write_preview_metadata,
    _write_image,
)
from .study_common import (
    LOGGER,
    PREVIEW_ROOT,
    _discover_preview_plane_root,
    _get_record,
    _metric_definitions,
    _preview_root_has_npy_planes,
    _sanitize_panel_order,
    _threshold_dict,
)
from .study_loader import _record_has_preview_sources
from image_processing import load_nd2_file
from visualization import ND2Visualizer


_PREVIEW_REVISION_LOCK = Lock()
_PREVIEW_LATEST_REVISION: Dict[str, str] = {}


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
        if threshold_key:
            candidate = preview_dir / threshold_key
            targets = [candidate] if candidate.exists() and candidate.is_dir() else []
        else:
            targets = removable_children(filter_raw=True)
        for path in targets:
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path.name)
    else:
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
    priority_entries = []
    if request.priority_subjects:
        entry_lookup = {
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
    ordered_entries = priority_entries + [
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
                dependency_token = _preview_dependency_token(
                    thresholds,
                    channel_ranges,
                    record.channel_definitions,
                    variant,
                )
                cache_key = _build_variant_cache_key(cache_base, variant, metric_id, dependency_token)
                output_dir = raw_dir if variant.cacheable else metric_dir
                safe_name = _build_variant_filename(safe_base, variant, metric_slug, variant.cacheable, dependency_token)
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


def resolve_preview_path(study_id: str, file_path: str) -> Path:
    _get_record(study_id)
    target = Path(file_path).expanduser().resolve()
    allowed_root = ensure_directory(PREVIEW_ROOT / study_id)
    if not str(target).startswith(str(allowed_root)):
        raise HTTPException(status_code=403, detail="Preview path is outside of the study directory")
    if not target.exists():
        raise HTTPException(status_code=404, detail="Preview image not found")
    return target


def _load_channels(
    record,
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
        subject_path = _resolve_source_image_path(record, subject_id, filename)
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


def _persist_preview_planes(record, group: str, subject_id: str, filename: str, channel_arrays: Dict[int, np.ndarray]) -> None:
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


def _load_channels_from_cached_png(record, group: str, subject_id: str, filename: str) -> Optional[Dict[int, np.ndarray]]:
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
                rescaled = (array / 255.0) * 4095.0
                channel_arrays[channel_index] = rescaled.astype(np.uint16)
        except Exception:
            continue

    return channel_arrays or None
