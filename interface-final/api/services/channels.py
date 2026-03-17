"""Helpers for channel naming and pseudo-color configuration."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple

VALID_CHANNELS = (1, 2, 3)

DEFAULT_CHANNEL_DEFINITIONS: List[Dict[str, object]] = [
    {"channel": 1, "label": "Channel 1", "color": "#00ff00"},
    {"channel": 2, "label": "Channel 2", "color": "#ff0000"},
    {"channel": 3, "label": "Channel 3", "color": "#0000ff"},
]

NAMED_COLORS: Dict[str, str] = {
    "red": "#ff0000",
    "green": "#00ff00",
    "blue": "#0000ff",
    "cyan": "#00ffff",
    "magenta": "#ff00ff",
    "yellow": "#ffff00",
    "orange": "#ff8800",
    "white": "#ffffff",
    "gray": "#9ca3af",
    "grey": "#9ca3af",
}

HEX_COLOR_PATTERN = re.compile(r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def normalize_channel_definitions(channels: Iterable[Dict[str, Any]] | None) -> List[Dict[str, object]]:
    """Validate and standardize channel definitions, falling back to defaults."""
    normalized = {entry["channel"]: dict(entry) for entry in DEFAULT_CHANNEL_DEFINITIONS}
    if not channels:
        return [normalized[channel] for channel in VALID_CHANNELS]

    for entry in channels:
        try:
            channel = int(entry.get("channel"))
        except (TypeError, ValueError, AttributeError):
            continue
        if channel not in VALID_CHANNELS:
            continue

        fallback = normalized[channel]
        raw_label = entry.get("label")
        label = str(raw_label).strip() if isinstance(raw_label, str) else ""
        if not label:
            label = str(fallback["label"])

        color = normalize_channel_color(entry.get("color"), str(fallback["color"]))
        normalized[channel] = {
            "channel": channel,
            "label": label,
            "color": color,
        }

    return [normalized[channel] for channel in VALID_CHANNELS]


def normalize_channel_color(value: Any, fallback: str = "#ffffff") -> str:
    if not isinstance(value, str):
        return _normalize_hex_color(fallback)
    candidate = value.strip().lower()
    if not candidate:
        return _normalize_hex_color(fallback)
    if candidate in NAMED_COLORS:
        return NAMED_COLORS[candidate]
    match = HEX_COLOR_PATTERN.match(candidate)
    if not match:
        return _normalize_hex_color(fallback)
    hex_value = match.group(1).lower()
    if len(hex_value) == 3:
        hex_value = "".join(ch * 2 for ch in hex_value)
    return f"#{hex_value}"


def channel_definition_map(channels: Iterable[Dict[str, Any]] | None) -> Dict[int, Dict[str, object]]:
    return {int(entry["channel"]): dict(entry) for entry in normalize_channel_definitions(channels)}


def channel_color_rgb(color: str) -> Tuple[float, float, float]:
    hex_value = normalize_channel_color(color)
    red = int(hex_value[1:3], 16) / 255.0
    green = int(hex_value[3:5], 16) / 255.0
    blue = int(hex_value[5:7], 16) / 255.0
    return red, green, blue


def _normalize_hex_color(value: str) -> str:
    match = HEX_COLOR_PATTERN.match(value.strip())
    if not match:
        return "#ffffff"
    hex_value = match.group(1).lower()
    if len(hex_value) == 3:
        hex_value = "".join(ch * 2 for ch in hex_value)
    return f"#{hex_value}"
