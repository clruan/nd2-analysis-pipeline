"""Helpers for handling file uploads from the web interface."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import UploadFile

from ..utils import ensure_directory, slugify

UPLOAD_ROOT = ensure_directory(Path(__file__).resolve().parent / "uploaded_inputs")

UploadCategory = Literal["config", "threshold_results"]


def store_upload(file: UploadFile, category: UploadCategory) -> Path:
    """Persist an uploaded file on disk and return the stored path."""
    target_dir = ensure_directory(UPLOAD_ROOT / slugify(category))
    suffix = Path(file.filename or "").suffix or ".dat"
    unique_name = f"{uuid4().hex}{suffix}"
    target_path = target_dir / unique_name

    with target_path.open("wb") as destination:
        shutil.copyfileobj(file.file, destination)

    return target_path.resolve()
