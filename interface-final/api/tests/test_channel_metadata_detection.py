"""Tests for metadata-derived channel labels and channel loading."""

from __future__ import annotations

from pathlib import Path
import sys
import types

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from image_processing import detect_channel_definitions, load_nd2_file  # noqa: E402


class _FakeOmeMetadata:
    def to_xml(self) -> str:
        return (
            '<OME><Image><Pixels>'
            '<Channel Id="Channel:0" Name="TL Brightfield" />'
            '<Channel Id="Channel:1" Name="DAPI" />'
            '<Channel Id="Channel:2" Name="Cy3" />'
            '<Channel Id="Channel:3" Name="Cy5" />'
            "</Pixels></Image></OME>"
        )


class _FakeAICSImage:
    def __init__(self, _filepath: str):
        self.ome_metadata = _FakeOmeMetadata()

    def get_image_data(self, _dims: str, T: int = 0) -> np.ndarray:
        del T
        return np.array(
            [
                [[[10, 10], [10, 10]]],
                [[[20, 20], [20, 20]]],
                [[[30, 30], [30, 30]]],
                [[[40, 40], [40, 40]]],
            ],
            dtype=np.uint16,
        )


def test_detect_channel_definitions_includes_all_metadata_channels_for_czi(monkeypatch) -> None:
    fake_module = types.ModuleType("aicsimageio")
    fake_module.AICSImage = _FakeAICSImage  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "aicsimageio", fake_module)

    definitions = detect_channel_definitions("/tmp/sample.czi")

    assert definitions is not None
    assert [entry["label"] for entry in definitions] == ["TL Brightfield", "DAPI", "Cy3", "Cy5"]
    assert [entry["channel"] for entry in definitions] == [1, 2, 3, 4]


def test_load_nd2_file_returns_first_three_source_channels(monkeypatch) -> None:
    fake_module = types.ModuleType("aicsimageio")
    fake_module.AICSImage = _FakeAICSImage  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "aicsimageio", fake_module)

    channel_1, channel_2, channel_3 = load_nd2_file("/tmp/sample.czi", is_3d=False)

    assert np.all(channel_1 == 10)
    assert np.all(channel_2 == 20)
    assert np.all(channel_3 == 30)
