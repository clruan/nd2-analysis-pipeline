"""Compatibility facade for study-oriented backend services."""

from __future__ import annotations

from ..state import STATE
from .analysis_service import analyze_study
from .download_service import generate_downloads, resolve_download_path
from .preview_rendering import (
    PreviewVariant,
    _generate_channel_raw_image,
    _normalize_channel_ranges,
    _preview_dependency_token,
)
from .preview_service import clear_preview_cache, generate_previews, render_preview_panel, resolve_preview_path
from .statistics_service import perform_statistics
from .study_common import PREVIEW_ROOT
from .study_loader import _rehydrate_study, _retokenize_mouse_ids, load_study, study_has_preview_sources
from .study_metadata_service import (
    _persist_study_annotations,
    list_channel_definitions,
    list_ratio_definitions,
    update_channel_definitions,
    update_pixel_size,
    update_ratio_definitions,
)


__all__ = [
    "PreviewVariant",
    "PREVIEW_ROOT",
    "_generate_channel_raw_image",
    "_normalize_channel_ranges",
    "_persist_study_annotations",
    "_preview_dependency_token",
    "_rehydrate_study",
    "_retokenize_mouse_ids",
    "analyze_study",
    "clear_preview_cache",
    "generate_downloads",
    "generate_previews",
    "list_channel_definitions",
    "list_ratio_definitions",
    "load_study",
    "perform_statistics",
    "render_preview_panel",
    "resolve_download_path",
    "resolve_preview_path",
    "study_has_preview_sources",
    "update_channel_definitions",
    "update_pixel_size",
    "update_ratio_definitions",
]


STATE.set_study_loader(_rehydrate_study)
