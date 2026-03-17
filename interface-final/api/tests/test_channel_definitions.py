"""Tests for channel definition normalization."""

from __future__ import annotations

from pathlib import Path
import sys

INTERFACE_ROOT = Path(__file__).resolve().parents[2]
if str(INTERFACE_ROOT) not in sys.path:
    sys.path.append(str(INTERFACE_ROOT))

from api.services.channels import channel_color_rgb, normalize_channel_definitions  # noqa: E402


def test_normalize_channel_definitions_defaults() -> None:
    normalized = normalize_channel_definitions(None)
    assert [entry["channel"] for entry in normalized] == [1, 2, 3]
    assert [entry["color"] for entry in normalized] == ["#00ff00", "#ff0000", "#0000ff"]


def test_normalize_channel_definitions_accepts_custom_labels_and_colors() -> None:
    normalized = normalize_channel_definitions(
        [
            {"channel": 1, "label": "DAPI", "color": "#3366ff"},
            {"channel": 2, "label": "MPO", "color": "green"},
            {"channel": 3, "label": "Citrulline", "color": "#f0a"},
        ]
    )
    assert normalized[0]["label"] == "DAPI"
    assert normalized[0]["color"] == "#3366ff"
    assert normalized[1]["label"] == "MPO"
    assert normalized[1]["color"] == "#00ff00"
    assert normalized[2]["label"] == "Citrulline"
    assert normalized[2]["color"] == "#ff00aa"


def test_channel_color_rgb_parses_hex() -> None:
    red, green, blue = channel_color_rgb("#00ff00")
    assert red == 0.0
    assert green == 1.0
    assert blue == 0.0
