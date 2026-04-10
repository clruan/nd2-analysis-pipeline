"""Compatibility checks for the study-service facade."""

from __future__ import annotations

from pathlib import Path
import sys

INTERFACE_ROOT = Path(__file__).resolve().parents[2]
if str(INTERFACE_ROOT) not in sys.path:
    sys.path.append(str(INTERFACE_ROOT))

from api.services.studies import (  # noqa: E402
    analyze_study,
    clear_preview_cache,
    generate_downloads,
    generate_previews,
    list_channel_definitions,
    list_ratio_definitions,
    load_study,
    perform_statistics,
    render_preview_panel,
    resolve_download_path,
    resolve_preview_path,
    study_has_preview_sources,
    update_channel_definitions,
    update_pixel_size,
    update_ratio_definitions,
)


def test_studies_facade_exports_main_entrypoints() -> None:
    exports = [
        analyze_study,
        clear_preview_cache,
        generate_downloads,
        generate_previews,
        list_channel_definitions,
        list_ratio_definitions,
        load_study,
        perform_statistics,
        render_preview_panel,
        resolve_download_path,
        resolve_preview_path,
        study_has_preview_sources,
        update_channel_definitions,
        update_pixel_size,
        update_ratio_definitions,
    ]

    assert all(callable(exported) for exported in exports)
