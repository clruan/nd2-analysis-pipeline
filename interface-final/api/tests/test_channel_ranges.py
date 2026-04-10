"""Tests for channel range normalization used by preview rendering."""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

INTERFACE_ROOT = Path(__file__).resolve().parents[2]
if str(INTERFACE_ROOT) not in sys.path:
    sys.path.append(str(INTERFACE_ROOT))

from api.schemas import ChannelRange  # noqa: E402
from api.services.studies import (  # noqa: E402
    PreviewVariant,
    _generate_channel_raw_image,
    _normalize_channel_ranges,
    _preview_dependency_token,
)
from api.services.preview_rendering import _generate_channel_mask_image, _generate_mask_image  # noqa: E402


def test_normalize_channel_ranges_accepts_pydantic_models() -> None:
    payload = {
        "channel_1": ChannelRange(vmin=25, vmax=125),
        "channel_3": ChannelRange(vmin=0, vmax=1024),
    }
    normalized = _normalize_channel_ranges(payload)

    assert normalized[1] == (25.0, 125.0)
    assert normalized[3] == (0.0, 1024.0)


def test_normalize_channel_ranges_clamps_and_orders_bounds() -> None:
    normalized = _normalize_channel_ranges({"channel_2": {"vmin": 900, "vmax": 100}})

    assert normalized[2][0] == 900.0
    assert normalized[2][1] > 900.0


def test_channel_raw_image_uses_configured_color() -> None:
    channels = {
        1: np.array([[0, 1000]], dtype=np.uint16),
        2: np.array([[0, 0]], dtype=np.uint16),
        3: np.array([[0, 0]], dtype=np.uint16),
    }
    definitions = [
        {"channel": 1, "label": "DAPI", "color": "#ff0000"},
        {"channel": 2, "label": "Marker A", "color": "#00ff00"},
        {"channel": 3, "label": "Marker B", "color": "#0000ff"},
    ]
    image = _generate_channel_raw_image(channels, 1, {}, definitions)

    assert image is not None
    assert image.shape == (1, 2, 3)
    assert int(image[0, 1, 0]) > 0
    assert int(image[0, 1, 1]) == 0
    assert int(image[0, 1, 2]) == 0


def test_channel_mask_image_is_binary_black_and_white() -> None:
    channels = {
        1: np.array([[0, 1200]], dtype=np.uint16),
        2: np.array([[0, 0]], dtype=np.uint16),
        3: np.array([[0, 0]], dtype=np.uint16),
    }
    thresholds = {"channel_1": 1000, "channel_2": 1000, "channel_3": 1000}
    definitions = [
        {"channel": 1, "label": "DAPI", "color": "#ff0000"},
        {"channel": 2, "label": "Marker A", "color": "#00ff00"},
        {"channel": 3, "label": "Marker B", "color": "#0000ff"},
    ]

    image = _generate_channel_mask_image(channels, thresholds, 1, definitions)

    assert image is not None
    assert image.shape == (1, 2, 3)
    assert tuple(image[0, 0]) == (0, 0, 0)
    assert tuple(image[0, 1]) == (255, 255, 255)


def test_combined_mask_image_is_binary_black_and_white() -> None:
    channels = {
        1: np.array([[0, 1200]], dtype=np.uint16),
        2: np.array([[1500, 0]], dtype=np.uint16),
        3: np.array([[0, 0]], dtype=np.uint16),
    }
    thresholds = {"channel_1": 1000, "channel_2": 1000, "channel_3": 1000}
    definitions = [
        {"channel": 1, "label": "DAPI", "color": "#ff0000"},
        {"channel": 2, "label": "Marker A", "color": "#00ff00"},
        {"channel": 3, "label": "Marker B", "color": "#0000ff"},
    ]

    image = _generate_mask_image(channels, thresholds, definitions)

    assert image is not None
    assert image.shape == (1, 2, 3)
    assert tuple(image[0, 0]) == (255, 255, 255)
    assert tuple(image[0, 1]) == (255, 255, 255)


def test_preview_dependency_token_ignores_unrelated_threshold_changes() -> None:
    variant = PreviewVariant("mask", (2,), False)
    thresholds_a = {"channel_1": 100, "channel_2": 250, "channel_3": 400}
    thresholds_b = {"channel_1": 900, "channel_2": 250, "channel_3": 400}
    channel_ranges = {1: (0.0, 1000.0), 2: (50.0, 1200.0)}
    definitions = [
        {"channel": 1, "label": "A", "color": "#00ff00"},
        {"channel": 2, "label": "B", "color": "#ff0000"},
        {"channel": 3, "label": "C", "color": "#0000ff"},
    ]

    token_a = _preview_dependency_token(thresholds_a, channel_ranges, definitions, variant)
    token_b = _preview_dependency_token(thresholds_b, channel_ranges, definitions, variant)

    assert token_a == token_b


def test_preview_dependency_token_ignores_unrelated_range_changes() -> None:
    variant = PreviewVariant("raw", (2,), True)
    thresholds = {"channel_1": 100, "channel_2": 250, "channel_3": 400}
    channel_ranges_a = {1: (0.0, 1000.0), 2: (50.0, 1200.0)}
    channel_ranges_b = {1: (200.0, 1800.0), 2: (50.0, 1200.0)}
    definitions = [
        {"channel": 1, "label": "A", "color": "#00ff00"},
        {"channel": 2, "label": "B", "color": "#ff0000"},
        {"channel": 3, "label": "C", "color": "#0000ff"},
    ]

    token_a = _preview_dependency_token(thresholds, channel_ranges_a, definitions, variant)
    token_b = _preview_dependency_token(thresholds, channel_ranges_b, definitions, variant)

    assert token_a == token_b
