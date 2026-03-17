"""Utility helpers for the Interface-Final API."""

from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Sequence

SUPPORTED_MICROSCOPY_EXTENSIONS = (".nd2", ".czi", ".oib", ".oif")
SUBJECT_TOKEN_PATTERN = re.compile(r"[A-Za-z]+#?\d{1,4}")
SUBJECT_CANONICAL_PATTERN = re.compile(r"^([A-Za-z]+)(\d{1,4})$")

def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value.strip("-") or "study"


def build_preview_base(group: str, subject_id: str, filename: str) -> str:
    return f"{slugify(group)}_{slugify(subject_id)}_{slugify(filename)}"


def preview_plane_filename(group: str, subject_id: str, filename: str, channel_index: int) -> str:
    base = build_preview_base(group, subject_id, filename)
    return f"{base}_ch{channel_index}.npy"


def preview_raw_filename(group: str, subject_id: str, filename: str, variant: str) -> str:
    base = build_preview_base(group, subject_id, filename)
    return f"{base}_{variant}.png"


def find_nd2_files(input_dir: Path, recursive: bool = True) -> List[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_dir}")
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_MICROSCOPY_EXTENSIONS
    )


def _normalize_subject_token(token: str) -> str:
    candidate = re.sub(r"[^A-Za-z0-9]", "", token)
    match = SUBJECT_CANONICAL_PATTERN.match(candidate)
    if not match:
        return ""
    return f"{match.group(1)}{match.group(2)}"


def _extract_subject_candidates(filename_stem: str) -> List[tuple[str, str, int]]:
    candidates: List[tuple[str, str, int]] = []
    seen: set[str] = set()
    for match in SUBJECT_TOKEN_PATTERN.finditer(filename_stem):
        token = _normalize_subject_token(match.group(0))
        if not token:
            continue
        key = token.upper()
        if key in seen:
            continue
        seen.add(key)
        prefix = re.match(r"^[A-Za-z]+", token)
        if prefix:
            candidates.append((token, prefix.group(0).lower(), match.start()))
    return candidates


def _fallback_unique_subject_ids(file_paths: Sequence[Path]) -> Dict[Path, str]:
    counts: Dict[str, int] = defaultdict(int)
    subject_ids: Dict[Path, str] = {}
    for file_path in sorted(file_paths, key=str):
        base = re.sub(r"\s+", " ", file_path.stem).strip() or file_path.name
        counts[base] += 1
        subject_ids[file_path] = base if counts[base] == 1 else f"{base}#{counts[base]}"
    return subject_ids


def infer_subject_ids(file_paths: Sequence[Path]) -> Dict[Path, str]:
    if not file_paths:
        return {}

    per_file: Dict[Path, List[tuple[str, str, int]]] = {}
    prefix_tokens: Dict[str, set[str]] = defaultdict(set)
    prefix_file_hits: Dict[str, int] = defaultdict(int)
    prefix_position_sum: Dict[str, int] = defaultdict(int)
    prefix_position_count: Dict[str, int] = defaultdict(int)

    for file_path in file_paths:
        candidates = _extract_subject_candidates(file_path.stem)
        per_file[file_path] = candidates
        seen_prefixes: set[str] = set()
        for token, prefix, position in candidates:
            prefix_tokens[prefix].add(token)
            prefix_position_sum[prefix] += position
            prefix_position_count[prefix] += 1
            if prefix not in seen_prefixes:
                prefix_file_hits[prefix] += 1
                seen_prefixes.add(prefix)

    total_files = len(file_paths)
    best_prefix: str | None = None
    best_score: tuple[float, float, float] | None = None
    for prefix, tokens in prefix_tokens.items():
        if len(tokens) < 2:
            continue
        coverage = prefix_file_hits[prefix] / total_files
        avg_position = prefix_position_sum[prefix] / max(prefix_position_count[prefix], 1)
        score = (float(len(tokens)), coverage, -avg_position)
        if best_score is None or score > best_score:
            best_prefix = prefix
            best_score = score

    if not best_prefix:
        return _fallback_unique_subject_ids(file_paths)

    subject_ids: Dict[Path, str] = {}
    for file_path in file_paths:
        matches = [item for item in per_file[file_path] if item[1] == best_prefix]
        if not matches:
            return _fallback_unique_subject_ids(file_paths)
        matches.sort(key=lambda item: item[2])
        subject_ids[file_path] = matches[0][0]
    return subject_ids


def assign_subject_ids(file_paths: Sequence[Path], strategy: Literal["per_file", "auto"] = "per_file") -> Dict[Path, str]:
    if strategy == "auto":
        return infer_subject_ids(file_paths)
    return _fallback_unique_subject_ids(file_paths)


def guess_subject_id(filename_stem: str) -> str:
    candidates = _extract_subject_candidates(filename_stem)
    if candidates:
        return candidates[0][0]
    return filename_stem


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_path(path_value: str) -> Path:
    """
    Normalize user-provided filesystem paths by trimming whitespace and
    stripping surrounding quotes before expanding any user home shortcut.
    """
    if path_value is None:
        raise ValueError("Path value cannot be None")
    candidate = path_value.strip()
    while candidate and candidate[0] in {"'", '"', "“", "”", "‘", "’"}:
        candidate = candidate[1:].lstrip()
    while candidate and candidate[-1] in {"'", '"', "“", "”", "‘", "’"}:
        candidate = candidate[:-1].rstrip()
    return Path(candidate).expanduser()


def common_prefix(parts: Iterable[str]) -> str:
    try:
        return os.path.commonprefix(list(parts))
    except ValueError:
        return ""
