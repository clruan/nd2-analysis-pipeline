"""Utility helpers for the Interface-Final API."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, List


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

    pattern = "**/*.nd2" if recursive else "*.nd2"
    return sorted(input_dir.glob(pattern))


SUBJECT_PATTERN = re.compile(r"^[A-Za-z]+\d{1,4}$")


def guess_subject_id(filename_stem: str) -> str:
    tokens = re.split(r"[\s_\-]+", filename_stem)
    for token in tokens:
        candidate = token.strip()
        if candidate and SUBJECT_PATTERN.match(candidate):
            return candidate
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
    if len(candidate) >= 2 and candidate[0] == candidate[-1] and candidate[0] in {"'", '"'}:
        candidate = candidate[1:-1]
    return Path(candidate).expanduser()


def common_prefix(parts: Iterable[str]) -> str:
    try:
        return os.path.commonprefix(list(parts))
    except ValueError:
        return ""
