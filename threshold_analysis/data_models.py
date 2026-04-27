"""Data models for interactive threshold analysis."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _normalize_channel_key(value: Any) -> int:
    channel = int(value)
    if channel <= 0:
        raise ValueError(f"Channel must be positive, got {value}")
    return channel


@dataclass
class ThresholdData:
    """Stores threshold analysis data for a single image."""

    mouse_id: str
    group: str
    filename: str
    channel_percentages: Dict[int, np.ndarray] = field(default_factory=dict)

    # Legacy fields kept for backward-compatible construction and reads.
    channel_1_percentages: Optional[np.ndarray] = None
    channel_2_percentages: Optional[np.ndarray] = None
    channel_3_percentages: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        normalized: Dict[int, np.ndarray] = {}

        for channel, values in (self.channel_percentages or {}).items():
            normalized[_normalize_channel_key(channel)] = np.asarray(values, dtype=np.float32)

        legacy_fields = {
            1: self.channel_1_percentages,
            2: self.channel_2_percentages,
            3: self.channel_3_percentages,
        }
        for channel, values in legacy_fields.items():
            if values is None:
                continue
            normalized.setdefault(channel, np.asarray(values, dtype=np.float32))

        self.channel_percentages = {channel: normalized[channel] for channel in sorted(normalized)}
        self.channel_1_percentages = self.channel_percentages.get(1)
        self.channel_2_percentages = self.channel_percentages.get(2)
        self.channel_3_percentages = self.channel_percentages.get(3)

    @property
    def channel_ids(self) -> List[int]:
        return sorted(self.channel_percentages)

    @property
    def max_threshold(self) -> int:
        if not self.channel_percentages:
            return 0
        return max(max(len(values) - 1, 0) for values in self.channel_percentages.values())

    def threshold_limit_for_channel(self, channel: int) -> int:
        values = self.channel_percentages.get(_normalize_channel_key(channel))
        if values is None:
            return 0
        return max(len(values) - 1, 0)

    def get_percentage_at_threshold(self, channel: int, threshold: int) -> float:
        """Get positive pixel percentage for a specific channel and threshold."""
        if threshold < 0:
            raise ValueError(f"Threshold must be non-negative, got {threshold}")

        values = self.channel_percentages.get(_normalize_channel_key(channel))
        if values is None:
            raise ValueError(f"Channel {channel} is not available for {self.filename}")
        if threshold >= len(values):
            return 0.0
        return float(values[threshold])


@dataclass
class ThresholdResults:
    """Container for all threshold analysis results from a study."""

    study_name: str
    image_data: List[ThresholdData]
    group_info: Dict[str, List[str]]  # group_name -> list of mouse_ids
    ratio_definitions: Optional[List[Dict[str, Any]]] = None
    channel_definitions: Optional[List[Dict[str, Any]]] = None
    pixel_size_um: Optional[float] = None

    @property
    def channel_ids(self) -> List[int]:
        if self.channel_definitions:
            defined = sorted(
                {
                    _normalize_channel_key(entry.get("channel"))
                    for entry in self.channel_definitions
                    if entry is not None and entry.get("channel") is not None
                }
            )
            if defined:
                return defined

        seen = {channel for entry in self.image_data for channel in entry.channel_ids}
        return sorted(seen) if seen else [1, 2, 3]

    @property
    def channel_limits(self) -> Dict[int, int]:
        limits: Dict[int, int] = {}
        for entry in self.image_data:
            for channel in entry.channel_ids:
                limits[channel] = max(limits.get(channel, 0), entry.threshold_limit_for_channel(channel))
        for channel in self.channel_ids:
            limits.setdefault(channel, 0)
        return dict(sorted(limits.items()))

    @property
    def max_threshold(self) -> int:
        limits = self.channel_limits
        return max(limits.values(), default=0)

    def get_mouse_averages(self, thresholds: Dict[str, int]) -> pd.DataFrame:
        """Calculate mouse averages for given thresholds."""
        results = []
        channel_ids = self.channel_ids

        for data in self.image_data:
            mouse_avg: Dict[str, Any] = {
                "Group": data.group,
                "MouseID": data.mouse_id,
            }
            for channel in channel_ids:
                threshold_key = f"channel_{channel}"
                threshold_value = int(thresholds.get(threshold_key, 0))
                mouse_avg[f"Channel_{channel}_area"] = data.get_percentage_at_threshold(channel, threshold_value)
            results.append(mouse_avg)

        df = pd.DataFrame(results)
        if df.empty:
            columns = ["Group", "MouseID"] + [f"Channel_{channel}_area" for channel in channel_ids]
            return pd.DataFrame(columns=columns)

        # Average by mouse (in case multiple images per mouse)
        return df.groupby(["Group", "MouseID"]).mean(numeric_only=True).reset_index()
