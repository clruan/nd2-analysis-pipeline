"""Tests for channel range normalization used by preview rendering."""

from __future__ import annotations

from pathlib import Path
import sys

INTERFACE_ROOT = Path(__file__).resolve().parents[2]
if str(INTERFACE_ROOT) not in sys.path:
    sys.path.append(str(INTERFACE_ROOT))

from api.schemas import ChannelRange  # noqa: E402
from api.services.studies import _normalize_channel_ranges  # noqa: E402


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
