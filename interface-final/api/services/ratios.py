"""Helpers for channel ratio definitions used across the interface."""

from __future__ import annotations

from typing import Iterable, List, Dict, Any

DEFAULT_RATIO_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "id": "channel_1_3_ratio",
        "label": "Channel 1 / Channel 3",
        "numerator_channel": 1,
        "denominator_channel": 3,
    },
    {
        "id": "channel_2_3_ratio",
        "label": "Channel 2 / Channel 3",
        "numerator_channel": 2,
        "denominator_channel": 3,
    },
]

VALID_CHANNELS = {1, 2, 3}
MAX_CUSTOM_RATIOS = 6


def normalize_ratio_definitions(ratios: Iterable[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    """Validate and standardize ratio definitions, falling back to defaults."""
    if not ratios:
        return list(DEFAULT_RATIO_DEFINITIONS)

    normalized: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()

    for index, entry in enumerate(ratios):
        if len(normalized) >= MAX_CUSTOM_RATIOS:
            break

        numerator = _coerce_channel(entry.get("numerator_channel") or entry.get("numerator"))
        denominator = _coerce_channel(entry.get("denominator_channel") or entry.get("denominator"))
        if numerator is None or denominator is None or numerator == denominator:
            continue

        ratio_id = str(entry.get("id") or f"channel_{numerator}_{denominator}_ratio")
        if ratio_id in seen_ids:
            continue
        seen_ids.add(ratio_id)

        label = entry.get("label") or f"Channel {numerator} / Channel {denominator}"

        normalized.append(
            {
                "id": ratio_id,
                "label": label,
                "numerator_channel": numerator,
                "denominator_channel": denominator,
            }
        )

    if not normalized:
        return list(DEFAULT_RATIO_DEFINITIONS)
    return normalized


def _coerce_channel(value: Any) -> int | None:
    if value is None:
        return None
    try:
        channel = int(value)
    except (TypeError, ValueError):
        return None
    if channel in VALID_CHANNELS:
        return channel
    return None
