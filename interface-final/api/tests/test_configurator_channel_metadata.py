"""Tests for propagating metadata-derived channel labels into config workflows."""

from __future__ import annotations

import json
from pathlib import Path
import sys

INTERFACE_ROOT = Path(__file__).resolve().parents[2]
if str(INTERFACE_ROOT) not in sys.path:
    sys.path.append(str(INTERFACE_ROOT))

from api.schemas import ConfigCreateRequest, ConfigScanRequest  # noqa: E402
from api.services import configurator  # noqa: E402


def test_scan_input_directory_returns_detected_channel_definitions(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "study"
    input_dir.mkdir()
    sample = input_dir / "sample.czi"
    sample.write_text("placeholder", encoding="utf-8")

    detected_channels = [
        {"channel": 1, "label": "DAPI", "color": "#0000ff"},
        {"channel": 2, "label": "VCAM", "color": "#00ff00"},
        {"channel": 3, "label": "FActin", "color": "#ff0000"},
    ]

    monkeypatch.setattr(configurator, "find_nd2_files", lambda *_args, **_kwargs: [sample])
    monkeypatch.setattr(configurator, "assign_subject_ids", lambda paths, strategy: {paths[0]: "A1"})
    monkeypatch.setattr(configurator, "detect_channel_definitions_from_dir", lambda _dir: detected_channels)

    response = configurator.scan_input_directory(ConfigScanRequest(input_dir=str(input_dir), recursive=True))

    assert response.channel_definitions is not None
    assert response.channel_definitions[0].label == "DAPI"
    assert response.channel_definitions[1].label == "VCAM"
    assert response.channel_definitions[2].label == "FActin"


def test_create_config_uses_detected_channel_definitions_when_request_omits_them(
    tmp_path: Path, monkeypatch
) -> None:
    input_dir = tmp_path / "study"
    input_dir.mkdir()
    output_path = tmp_path / "config.json"

    detected_channels = [
        {"channel": 1, "label": "DAPI", "color": "#0000ff"},
        {"channel": 2, "label": "VCAM", "color": "#00ff00"},
        {"channel": 3, "label": "FActin", "color": "#ff0000"},
    ]

    monkeypatch.setattr(configurator, "detect_channel_definitions_from_dir", lambda _dir: detected_channels)

    response = configurator.create_config(
        ConfigCreateRequest(
            input_dir=str(input_dir),
            study_name="demo-study",
            groups={"Control": ["A1"]},
            output_path=str(output_path),
        )
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert response.channel_definitions is not None
    assert response.channel_definitions[0].label == "DAPI"
    assert payload["channel_definitions"][0]["label"] == "DAPI"
