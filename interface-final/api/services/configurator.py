"""Helpers for configuration discovery and creation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from fastapi import HTTPException

from data_models import GroupConfig

from ..schemas import (
    ConfigCreateRequest,
    ConfigCreateResponse,
    ConfigReadResponse,
    ConfigScanRequest,
    ConfigScanResponse,
    GroupInfo,
    ReplicaInfo,
    SubjectInfo,
)
from ..utils import ensure_directory, find_nd2_files, guess_subject_id, normalize_path, slugify
from .ratios import normalize_ratio_definitions


def scan_input_directory(request: ConfigScanRequest) -> ConfigScanResponse:
    input_dir = normalize_path(request.input_dir).resolve()
    try:
        nd2_files = find_nd2_files(input_dir, recursive=request.recursive)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except NotADirectoryError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=f"Permission denied while reading {input_dir}: {exc}") from exc
    except Exception as exc:  # pragma: no cover - unexpected filesystem errors
        raise HTTPException(status_code=500, detail=f"Unable to scan directory: {exc}") from exc

    group_map: Dict[str, Dict[str, List[ReplicaInfo]]] = defaultdict(lambda: defaultdict(list))

    for file_path in nd2_files:
        relative = file_path.relative_to(input_dir)
        group_name = relative.parts[0] if len(relative.parts) > 1 else "Ungrouped"
        subject_id = guess_subject_id(file_path.stem)
        group_map[group_name][subject_id].append(
            ReplicaInfo(filename=file_path.name, absolute_path=str(file_path))
        )

    group_infos: List[GroupInfo] = []
    for group_name, subjects in sorted(group_map.items()):
        subject_infos = [
            SubjectInfo(subject_id=subject_id, replicates=replicas)
            for subject_id, replicas in sorted(subjects.items())
        ]
        group_infos.append(GroupInfo(group_name=group_name, subjects=subject_infos))

    study_name = input_dir.name

    return ConfigScanResponse(
        study_name=study_name,
        input_dir=str(input_dir),
        nd2_files=[str(path) for path in nd2_files],
        groups=group_infos,
    )


def create_config(request: ConfigCreateRequest) -> ConfigCreateResponse:
    input_dir = normalize_path(request.input_dir).resolve()
    if not input_dir.exists():
        raise HTTPException(status_code=404, detail=f"Input directory does not exist: {input_dir}")

    slug = slugify(request.study_name)
    config_dir = ensure_directory(Path(__file__).resolve().parent / "generated_configs")
    output_path = normalize_path(request.output_path).resolve() if request.output_path else config_dir / f"{slug}.json"

    ratio_entries = [ratio.dict() for ratio in request.ratios] if request.ratios else None
    normalized_ratios = normalize_ratio_definitions(ratio_entries)

    group_config = GroupConfig(
        groups=request.groups,
        thresholds=request.thresholds,
        pixel_size_um=request.pixel_size_um,
        ratios=normalized_ratios,
    )
    group_config.to_json(str(output_path))

    return ConfigCreateResponse(
        config_path=str(output_path),
        study_name=request.study_name,
        groups=request.groups,
        ratios=group_config.ratios,
    )


def read_config(config_path: str) -> ConfigReadResponse:
    path = normalize_path(config_path).resolve()
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Config not found: {path}")
    config = GroupConfig.from_json(str(path))
    study_name = path.stem
    return ConfigReadResponse(
        config_path=str(path),
        study_name=study_name,
        groups=config.groups,
        pixel_size_um=config.pixel_size_um,
        thresholds=config.thresholds,
        ratios=config.ratios,
    )
