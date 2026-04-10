"""Study metadata persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import HTTPException

from ..state import STATE, StudyRecord
from .channels import normalize_channel_definitions
from .ratios import normalize_ratio_definitions
from .study_common import _get_record


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
    STATE.save_study(record)


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
