"""Shared constants and helpers for study-oriented services."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Set, Tuple

import numpy as np
import pandas as pd
from fastapi import HTTPException

from ..state import STATE, StudyRecord
from ..utils import ensure_directory
from .channels import channel_definition_map


DOWNLOAD_ROOT = ensure_directory(Path(__file__).resolve().parent / "generated_downloads")
PREVIEW_ROOT = ensure_directory(Path(__file__).resolve().parent / "generated_previews")

LOGGER = logging.getLogger(__name__)

SUBJECT_TOKEN_PATTERN = re.compile(r"([A-Za-z]+)(\d{1,4})")


def _channel_defs_for_record(record: StudyRecord) -> Dict[int, Dict[str, object]]:
    return channel_definition_map(record.channel_definitions, channel_ids=_record_channel_ids(record))


def _record_channel_ids(record: StudyRecord) -> List[int]:
    channel_ids = sorted({int(entry["channel"]) for entry in (record.channel_definitions or []) if int(entry["channel"]) > 0})
    if channel_ids:
        return channel_ids
    return list(record.results.channel_ids)


def _channel_metric_definitions(record: StudyRecord) -> List[Dict[str, object]]:
    return [
        {"id": f"channel_{channel}_area", "kind": "channel", "channel": channel}
        for channel in _record_channel_ids(record)
    ]


def _preview_metric_ids(record: StudyRecord) -> List[str]:
    metrics = [metric["id"] for metric in _channel_metric_definitions(record)]
    metrics.extend(str(ratio["id"]) for ratio in record.ratio_definitions)
    return metrics


def _default_preview_metric(record: StudyRecord) -> str:
    metrics = _preview_metric_ids(record)
    return metrics[0] if metrics else "channel_1_area"


def _default_panel_order(channel_ids: Iterable[int]) -> List[str]:
    normalized_ids = sorted({int(channel) for channel in channel_ids if int(channel) > 0})
    order = [f"channel_{channel}" for channel in normalized_ids]
    order.append("composite")
    return order or ["composite"]


def _graph_pad_export_keys(record: StudyRecord) -> List[str]:
    return [f"Channel_{channel}_area" for channel in _record_channel_ids(record)]


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


def _metric_definitions(record: StudyRecord) -> List[Dict[str, object]]:
    metrics: List[Dict[str, object]] = []
    for defn in _channel_metric_definitions(record):
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


def _threshold_dict(thresholds: Dict[str, int], channel_ids: Optional[Iterable[int]] = None) -> Dict[str, int]:
    normalized: Dict[str, int] = {}
    expected_ids = sorted({int(channel) for channel in (channel_ids or []) if int(channel) > 0})
    for channel in expected_ids:
        key = f"channel_{channel}"
        value = thresholds.get(key, 0)
        normalized[key] = max(0, int(value))

    for key, value in thresholds.items():
        match = re.match(r"^channel_(\d+)$", str(key))
        if not match:
            continue
        normalized[f"channel_{int(match.group(1))}"] = max(0, int(value))

    if normalized:
        return dict(sorted(normalized.items(), key=lambda item: int(item[0].split("_")[1])))

    return {"channel_1": 0, "channel_2": 0, "channel_3": 0}


def _get_record(study_id: str) -> StudyRecord:
    try:
        return STATE.get_study(study_id)
    except KeyError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=404, detail=f"Study not loaded: {study_id}") from exc


def _analysis_cache_key(thresholds: Dict[str, int]) -> str:
    parts = [f"{key}={thresholds[key]}" for key in sorted(thresholds, key=lambda item: int(item.split("_")[1]))]
    return "|".join(parts)


def _sanitize_panel_order(order: Optional[Iterable[str]], channel_ids: Optional[Iterable[int]] = None) -> List[str]:
    default_order = _default_panel_order(channel_ids or [1, 2, 3])
    allowed_panels = set(default_order)
    if not order:
        return default_order
    normalized: List[str] = []
    for panel in order:
        if panel in allowed_panels and panel not in normalized:
            normalized.append(panel)
    return normalized or default_order
