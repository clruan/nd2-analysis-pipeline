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
from .study_common import _analysis_cache_key, _ensure_ratio_columns, _get_record, _threshold_dict
from image_processing import load_nd2_file


_ANALYSIS_LOCKS: Dict[str, Lock] = {}
_ANALYSIS_LOCKS_GUARD = Lock()


def _analysis_lock_for(study_id: str) -> Lock:
    with _ANALYSIS_LOCKS_GUARD:
        lock = _ANALYSIS_LOCKS.get(study_id)
        if lock is None:
            lock = Lock()
            _ANALYSIS_LOCKS[study_id] = lock
        return lock


def _build_image_index(image_data: Iterable[ThresholdData]) -> Dict[str, object]:
    entries = list(image_data)
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
        1: tuple(entry.channel_1_percentages for entry in entries),
        2: tuple(entry.channel_2_percentages for entry in entries),
        3: tuple(entry.channel_3_percentages for entry in entries),
    }
    return {
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
    if record.analysis_index:
        return record.analysis_index
    record.analysis_index = _build_image_index(record.results.image_data)
    return record.analysis_index


def _channel_values_for_threshold(index: Dict[str, object], channel: int, threshold: int) -> np.ndarray:
    channel_arrays = index["channel_arrays"][channel]
    image_count = int(index["image_count"])
    return np.fromiter((values[threshold] for values in channel_arrays), dtype=np.float32, count=image_count)


def _channel_value_map(index: Dict[str, object], thresholds: Dict[str, int]) -> Dict[int, np.ndarray]:
    return {
        1: _channel_values_for_threshold(index, 1, thresholds["channel_1"]),
        2: _channel_values_for_threshold(index, 2, thresholds["channel_2"]),
        3: _channel_values_for_threshold(index, 3, thresholds["channel_3"]),
    }


def _mouse_averages_dataframe(index: Dict[str, object], channel_values: Dict[int, np.ndarray]) -> pd.DataFrame:
    subject_indices = index["image_subject_indices"]
    subject_counts = index["subject_counts"]
    subject_count = int(subject_counts.shape[0])
    subject_totals = np.zeros((subject_count, 3), dtype=np.float32)

    np.add.at(subject_totals[:, 0], subject_indices, channel_values[1])
    np.add.at(subject_totals[:, 1], subject_indices, channel_values[2])
    np.add.at(subject_totals[:, 2], subject_indices, channel_values[3])

    subject_means = subject_totals / subject_counts[:, None]
    return pd.DataFrame(
        {
            "Group": index["subject_groups"],
            "MouseID": index["subject_mouse_ids"],
            "Channel_1_area": subject_means[:, 0],
            "Channel_2_area": subject_means[:, 1],
            "Channel_3_area": subject_means[:, 2],
        }
    )


def _ratio_value_arrays(
    channel_values: Dict[int, np.ndarray],
    ratio_definitions: List[Dict[str, object]],
    epsilon: float = 1e-3,
) -> Dict[str, np.ndarray]:
    ratio_arrays: Dict[str, np.ndarray] = {}
    for ratio in ratio_definitions:
        num_idx = int(ratio["numerator_channel"])
        den_idx = int(ratio["denominator_channel"])
        numerator = channel_values[num_idx]
        denominator = channel_values[den_idx]
        ratio_arrays[str(ratio["id"])] = numerator / (denominator + epsilon)
    return ratio_arrays


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

    return [
        IndividualImageRecord(
            group=str(image_groups[row]),
            mouse_id=str(image_mouse_ids[row]),
            filename=str(image_filenames[row]),
            channel_1_area=float(channel_values[1][row]),
            channel_2_area=float(channel_values[2][row]),
            channel_3_area=float(channel_values[3][row]),
            ratios={ratio_id: float(values[row]) for ratio_id, values in ratio_arrays.items()},
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
    index = _build_image_index(image_data)
    channel_values = _channel_value_map(index, thresholds)
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
