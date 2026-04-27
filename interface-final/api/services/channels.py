"""Helpers for channel naming and pseudo-color configuration."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from image_processing import detect_channel_definitions as detect_channel_definitions_from_file

from ..utils import find_nd2_files

DEFAULT_CHANNEL_COLORS: Tuple[str, ...] = (
    "#00ff00",
    "#ff0000",
    "#0000ff",
    "#ffff00",
    "#00ffff",
    "#ff00ff",
    "#ffffff",
    "#ff8800",
)

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


def _default_channel_color(channel: int) -> str:
    if 1 <= channel <= len(DEFAULT_CHANNEL_COLORS):
        return DEFAULT_CHANNEL_COLORS[channel - 1]
    return DEFAULT_CHANNEL_COLORS[(channel - 1) % len(DEFAULT_CHANNEL_COLORS)]


def default_channel_definitions(channel_ids: Optional[Iterable[int]] = None) -> List[Dict[str, object]]:
    normalized_ids = sorted({int(channel) for channel in (channel_ids or [1, 2, 3]) if int(channel) > 0})
    if not normalized_ids:
        normalized_ids = [1, 2, 3]
    return [
        {"channel": channel, "label": f"Channel {channel}", "color": _default_channel_color(channel)}
        for channel in normalized_ids
    ]


DEFAULT_CHANNEL_DEFINITIONS: List[Dict[str, object]] = default_channel_definitions()


def normalize_channel_definitions(
    channels: Iterable[Dict[str, Any]] | None,
    channel_ids: Optional[Iterable[int]] = None,
) -> List[Dict[str, object]]:
    """Validate and standardize channel definitions, falling back to defaults."""
    target_ids = sorted(
        {
            int(channel)
            for channel in (channel_ids or [])
            if channel is not None and int(channel) > 0
        }
    )

    provided_ids: List[int] = []
    for entry in channels or []:
        try:
            channel = int(entry.get("channel"))
        except (TypeError, ValueError, AttributeError):
            continue
        if channel > 0:
            provided_ids.append(channel)

    if not target_ids:
        target_ids = sorted(set(provided_ids)) or [1, 2, 3]

    normalized = {entry["channel"]: dict(entry) for entry in default_channel_definitions(target_ids)}
    for entry in channels or []:
        try:
            channel = int(entry.get("channel"))
        except (TypeError, ValueError, AttributeError):
            continue
        if channel <= 0 or channel not in normalized:
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

    return [normalized[channel] for channel in sorted(normalized)]


def channel_definitions_are_default(
    channels: Iterable[Dict[str, Any]] | None,
    channel_ids: Optional[Iterable[int]] = None,
) -> bool:
    normalized = normalize_channel_definitions(channels, channel_ids=channel_ids)
    defaults = default_channel_definitions(entry["channel"] for entry in normalized)
    return normalized == defaults


def detect_channel_definitions_from_dir(input_dir: Path) -> List[Dict[str, object]] | None:
    try:
        microscopy_files = find_nd2_files(input_dir, recursive=True)
    except Exception:
        return None

    for path in microscopy_files:
        try:
            detected = detect_channel_definitions_from_file(str(path))
        except Exception:
            continue
        if detected:
            return normalize_channel_definitions(detected)
    return None


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


def channel_definition_map(
    channels: Iterable[Dict[str, Any]] | None,
    channel_ids: Optional[Iterable[int]] = None,
) -> Dict[int, Dict[str, object]]:
    return {int(entry["channel"]): dict(entry) for entry in normalize_channel_definitions(channels, channel_ids=channel_ids)}


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
