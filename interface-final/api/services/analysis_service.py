"""Analysis helpers for loaded studies."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from threshold_analysis.data_models import ThresholdData

from ..schemas import AnalyzeRequest, AnalyzeResponse, IndividualImageRecord, MouseAverageRecord
from ..state import StudyRecord
from .study_common import _analysis_cache_key, _ensure_ratio_columns, _get_record, _record_channel_ids, _threshold_dict


_ANALYSIS_LOCKS: Dict[str, Lock] = {}
_ANALYSIS_LOCKS_GUARD = Lock()


def _analysis_lock_for(study_id: str) -> Lock:
    with _ANALYSIS_LOCKS_GUARD:
        lock = _ANALYSIS_LOCKS.get(study_id)
        if lock is None:
            lock = Lock()
            _ANALYSIS_LOCKS[study_id] = lock
        return lock


def _build_image_index(image_data: Iterable[ThresholdData], channel_ids: Iterable[int]) -> Dict[str, object]:
    entries = list(image_data)
    channel_ids = list(channel_ids)
    image_count = len(entries)
    image_groups = np.empty(image_count, dtype=object)
    image_mouse_ids = np.empty(image_count, dtype=object)
    image_filenames = np.empty(image_count, dtype=object)
    image_subject_indices = np.empty(image_count, dtype=np.int32)
    replicate_indices = np.empty(image_count, dtype=np.int32)
    replicate_counters: Counter[Tuple[str, str]] = Counter()
    subject_index_by_key: Dict[Tuple[str, str], int] = {}
    subject_groups: List[str] = []
    subject_mouse_ids: List[str] = []

    for index, entry in enumerate(entries):
        group = str(entry.group)
        mouse_id = str(entry.mouse_id)
        image_groups[index] = group
        image_mouse_ids[index] = mouse_id
        image_filenames[index] = str(entry.filename)

        key = (group, mouse_id)
        subject_index = subject_index_by_key.get(key)
        if subject_index is None:
            subject_index = len(subject_groups)
            subject_index_by_key[key] = subject_index
            subject_groups.append(group)
            subject_mouse_ids.append(mouse_id)
        image_subject_indices[index] = subject_index

        replicate_counters[key] += 1
        replicate_indices[index] = replicate_counters[key]

    subject_counts = np.bincount(image_subject_indices, minlength=len(subject_groups)).astype(np.float32, copy=False)
    channel_arrays = {
        channel: tuple(entry.channel_percentages.get(channel) for entry in entries)
        for channel in channel_ids
    }
    return {
        "channel_ids": channel_ids,
        "image_count": image_count,
        "image_groups": image_groups,
        "image_mouse_ids": image_mouse_ids,
        "image_filenames": image_filenames,
        "image_subject_indices": image_subject_indices,
        "replicate_indices": replicate_indices,
        "subject_groups": np.asarray(subject_groups, dtype=object),
        "subject_mouse_ids": np.asarray(subject_mouse_ids, dtype=object),
        "subject_counts": subject_counts,
        "channel_arrays": channel_arrays,
    }


def _ensure_analysis_index(record: StudyRecord) -> Dict[str, object]:
    channel_ids = _record_channel_ids(record)
    cached_channels = record.analysis_index.get("channel_ids")
    if record.analysis_index and cached_channels == channel_ids:
        return record.analysis_index
    record.analysis_index = _build_image_index(record.results.image_data, channel_ids)
    return record.analysis_index


def _value_at_threshold(values: Optional[np.ndarray], threshold: int) -> float:
    if values is None or threshold < 0:
        return 0.0
    if threshold >= len(values):
        return 0.0
    return float(values[threshold])


def _channel_values_for_threshold(index: Dict[str, object], channel: int, threshold: int) -> np.ndarray:
    channel_arrays = index["channel_arrays"].get(channel, ())
    image_count = int(index["image_count"])
    return np.fromiter((_value_at_threshold(values, threshold) for values in channel_arrays), dtype=np.float32, count=image_count)


def _channel_value_map(index: Dict[str, object], thresholds: Dict[str, int]) -> Dict[int, np.ndarray]:
    return {
        channel: _channel_values_for_threshold(index, channel, thresholds.get(f"channel_{channel}", 0))
        for channel in index["channel_ids"]
    }


def _mouse_averages_dataframe(
    index: Dict[str, object],
    channel_values: Dict[int, np.ndarray],
) -> pd.DataFrame:
    subject_indices = index["image_subject_indices"]
    subject_counts = index["subject_counts"]
    subject_count = int(subject_counts.shape[0])
    channel_ids: List[int] = list(index["channel_ids"])
    subject_totals = np.zeros((subject_count, len(channel_ids)), dtype=np.float32)

    for column_index, channel in enumerate(channel_ids):
        np.add.at(subject_totals[:, column_index], subject_indices, channel_values[channel])

    subject_means = subject_totals / subject_counts[:, None]
    payload: Dict[str, object] = {
        "Group": index["subject_groups"],
        "MouseID": index["subject_mouse_ids"],
    }
    for column_index, channel in enumerate(channel_ids):
        payload[f"Channel_{channel}_area"] = subject_means[:, column_index]
    return pd.DataFrame(payload)


def _ratio_value_arrays(
    channel_values: Dict[int, np.ndarray],
    ratio_definitions: List[Dict[str, object]],
    epsilon: float = 1e-3,
) -> Dict[str, np.ndarray]:
    ratio_arrays: Dict[str, np.ndarray] = {}
    for ratio in ratio_definitions:
        num_idx = int(ratio["numerator_channel"])
        den_idx = int(ratio["denominator_channel"])
        numerator = channel_values.get(num_idx)
        denominator = channel_values.get(den_idx)
        if numerator is None or denominator is None:
            ratio_arrays[str(ratio["id"])] = np.full(0, np.nan, dtype=np.float32)
            continue
        ratio_arrays[str(ratio["id"])] = numerator / (denominator + epsilon)
    return ratio_arrays


def _channel_area_payload(channel_values: Dict[int, float]) -> Dict[str, float]:
    return {
        f"channel_{channel}_area": float(value)
        for channel, value in sorted(channel_values.items())
    }


def _build_individual_image_records(
    index: Dict[str, object],
    channel_values: Dict[int, np.ndarray],
    ratio_definitions: List[Dict[str, object]],
) -> List[IndividualImageRecord]:
    image_count = int(index["image_count"])
    ratio_arrays = _ratio_value_arrays(channel_values, ratio_definitions)
    image_groups = index["image_groups"]
    image_mouse_ids = index["image_mouse_ids"]
    image_filenames = index["image_filenames"]
    replicate_indices = index["replicate_indices"]
    channel_ids: List[int] = list(index["channel_ids"])

    return [
        IndividualImageRecord(
            group=str(image_groups[row]),
            mouse_id=str(image_mouse_ids[row]),
            filename=str(image_filenames[row]),
            channel_areas=_channel_area_payload({channel: float(channel_values[channel][row]) for channel in channel_ids}),
            ratios={
                ratio_id: float(values[row]) if row < len(values) else float("nan")
                for ratio_id, values in ratio_arrays.items()
            },
            replicate_index=int(replicate_indices[row]),
        )
        for row in range(image_count)
    ]


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

    with _analysis_lock_for(record.study_id):
        cached = record.analysis_cache.get(cache_key)
        if cached:
            cached_mouse = cached.get("mouse_averages")
            cached_images = cached.get("individual_images")
            if isinstance(cached_mouse, pd.DataFrame) and isinstance(cached_images, list):
                return cached_mouse, cached_images

        index = _ensure_analysis_index(record)
        channel_values = _channel_value_map(index, thresholds)
        mouse_averages_df = _mouse_averages_dataframe(index, channel_values)
        _ensure_ratio_columns(mouse_averages_df, record.ratio_definitions)
        individual_images = _build_individual_image_records(index, channel_values, record.ratio_definitions)
        record.analysis_cache[cache_key] = {
            "mouse_averages": mouse_averages_df,
            "individual_images": individual_images,
        }
        return mouse_averages_df, individual_images


def analyze_study(study_id: str, request: AnalyzeRequest) -> AnalyzeResponse:
    record = _get_record(study_id)
    thresholds = _threshold_dict(request.thresholds, channel_ids=_record_channel_ids(record))
    mouse_averages_df, individual_images = _analysis_tables_for_thresholds(record, thresholds)

    mouse_records: List[MouseAverageRecord] = []
    channel_ids = _record_channel_ids(record)
    for row in mouse_averages_df.to_dict(orient="records"):
        ratio_values: Dict[str, float] = {}
        for ratio in record.ratio_definitions:
            value = row.get(ratio["id"])
            if isinstance(value, (int, float, np.floating)):
                ratio_values[str(ratio["id"])] = float(value)
        channel_areas = {
            f"channel_{channel}_area": float(row.get(f"Channel_{channel}_area", 0.0))
            for channel in channel_ids
        }
        mouse_records.append(
            MouseAverageRecord(
                Group=str(row.get("Group", "")),
                MouseID=str(row.get("MouseID", "")),
                channel_areas=channel_areas,
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
    entries = list(image_data)
    channel_ids = sorted({channel for entry in entries for channel in entry.channel_ids}) or [1, 2, 3]
    normalized_thresholds = _threshold_dict(thresholds, channel_ids=channel_ids)
    index = _build_image_index(entries, channel_ids)
    channel_values = _channel_value_map(index, normalized_thresholds)
    return _build_individual_image_records(index, channel_values, ratio_definitions)


def _resolve_source_image_path(record: StudyRecord, mouse_id: str, filename: str) -> Optional[Path]:
    direct = record.replicate_lookup.get(mouse_id, {}).get(filename)
    if direct and direct.exists():
        return direct
    for subject_map in record.replicate_lookup.values():
        candidate = subject_map.get(filename)
        if candidate and candidate.exists():
            return candidate
    return None
