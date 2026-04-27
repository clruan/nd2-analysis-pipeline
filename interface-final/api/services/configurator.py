"""Helpers for configuration discovery and creation."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from fastapi import HTTPException

from data_models import GroupConfig

from ..schemas import (
    ConfigCreateRequest,
    ConfigCreateResponse,
    ConfigAutoGroupRequest,
    ConfigAutoGroupResponse,
    ConfigReadResponse,
    ConfigScanRequest,
    ConfigScanResponse,
    GroupInfo,
    ReplicaInfo,
    SubjectInfo,
)
from ..utils import assign_subject_ids, ensure_directory, find_nd2_files, normalize_path, slugify
from .channels import detect_channel_definitions_from_dir, normalize_channel_definitions
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
    subject_ids = assign_subject_ids(nd2_files, strategy=request.subject_strategy)

    for file_path in nd2_files:
        relative = file_path.relative_to(input_dir)
        group_name = relative.parts[0] if len(relative.parts) > 1 else "Ungrouped"
        subject_id = subject_ids.get(file_path, file_path.stem)
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
    channel_definitions = detect_channel_definitions_from_dir(input_dir)

    return ConfigScanResponse(
        study_name=study_name,
        input_dir=str(input_dir),
        nd2_files=[str(path) for path in nd2_files],
        groups=group_infos,
        channel_definitions=channel_definitions,
    )


def create_config(request: ConfigCreateRequest) -> ConfigCreateResponse:
    input_dir = normalize_path(request.input_dir).resolve()
    if not input_dir.exists():
        raise HTTPException(status_code=404, detail=f"Input directory does not exist: {input_dir}")

    slug = slugify(request.study_name)
    config_dir = ensure_directory(Path(__file__).resolve().parent / "generated_configs")
    output_path = normalize_path(request.output_path).resolve() if request.output_path else config_dir / f"{slug}.json"

    ratio_entries = [ratio.dict() for ratio in request.ratios] if request.ratios else None
    channel_entries = [channel.dict() for channel in request.channel_definitions] if request.channel_definitions else None
    detected_channels = detect_channel_definitions_from_dir(input_dir) if not channel_entries else None
    normalized_ratios = normalize_ratio_definitions(ratio_entries)
    normalized_channels = normalize_channel_definitions(channel_entries or detected_channels)

    group_config = GroupConfig(
        groups=request.groups,
        thresholds=request.thresholds,
        pixel_size_um=request.pixel_size_um,
        ratios=normalized_ratios,
        channel_definitions=normalized_channels,
    )
    group_config.to_json(str(output_path))

    return ConfigCreateResponse(
        config_path=str(output_path),
        study_name=request.study_name,
        groups=request.groups,
        ratios=group_config.ratios,
        channel_definitions=group_config.channel_definitions,
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
        channel_definitions=normalize_channel_definitions(config.channel_definitions),
    )


def suggest_groups_with_llm(request: ConfigAutoGroupRequest) -> ConfigAutoGroupResponse:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY is not set on the API server.")

    scan = scan_input_directory(ConfigScanRequest(input_dir=request.input_dir, recursive=True))
    if not scan.groups:
        raise HTTPException(status_code=404, detail="No supported microscopy files found to infer group mappings.")

    subject_entries = [
        {
            "detected_group": group.group_name,
            "subject_id": subject.subject_id,
            "replicates": [rep.filename for rep in subject.replicates],
        }
        for group in scan.groups
        for subject in group.subjects
    ]
    known_subjects = {entry["subject_id"] for entry in subject_entries}
    model = request.model or os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"

    system_prompt = (
        "You are helping a microscopy analysis workflow build group mappings. "
        "Return strict JSON with this shape only: "
        '{"groups":{"Group Name":["SUBJECT1","SUBJECT2"]},"notes":"short optional note"}. '
        "Use subject_id strings exactly as provided."
    )
    user_prompt = (
        f"User instructions:\n{request.instructions.strip()}\n\n"
        f"Detected subjects and files:\n{json.dumps(subject_entries, indent=2)}\n\n"
        "Return only JSON."
    )

    payload = {
        "model": model,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise HTTPException(status_code=502, detail=f"OpenAI request failed: {detail}") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Unable to call OpenAI API: {exc}") from exc

    try:
        content = body["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except Exception as exc:
        raise HTTPException(status_code=502, detail="OpenAI response did not contain valid JSON.") from exc

    raw_groups = parsed.get("groups")
    if not isinstance(raw_groups, dict):
        raise HTTPException(status_code=422, detail="LLM output missing 'groups' object.")

    normalized: Dict[str, List[str]] = {}
    for group_name, subjects in raw_groups.items():
        if not isinstance(group_name, str):
            continue
        if not isinstance(subjects, list):
            continue
        cleaned = sorted(
            {str(subject).strip() for subject in subjects if isinstance(subject, str) and str(subject).strip() in known_subjects}
        )
        if cleaned:
            normalized[group_name.strip() or "Group"] = cleaned

    if not normalized:
        raise HTTPException(status_code=422, detail="LLM output did not produce any valid subject-group assignments.")

    notes = parsed.get("notes")
    return ConfigAutoGroupResponse(
        groups=normalized,
        model=model,
        notes=notes if isinstance(notes, str) and notes.strip() else None,
    )
