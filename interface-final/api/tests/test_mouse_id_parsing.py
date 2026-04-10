"""Regression tests for filename-to-subject matching in threshold generation."""

from __future__ import annotations

import sys
import types
from pathlib import Path

if "pyclesperanto" not in sys.modules:
    sys.modules["pyclesperanto"] = types.ModuleType("pyclesperanto")

if "nd2reader" not in sys.modules:
    nd2_stub = types.ModuleType("nd2reader")
    nd2_stub.ND2Reader = object  # type: ignore[attr-defined]
    nd2_stub.Nd2 = object  # type: ignore[attr-defined]
    sys.modules["nd2reader"] = nd2_stub

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from image_processing import parse_mouse_id  # noqa: E402


def test_parse_mouse_id_prefers_longer_exact_known_ids() -> None:
    mouse_id = parse_mouse_id(
        "/tmp/study/Picture WT C12 sample.nd2",
        mouse_ids=["C1", "C2", "C11", "C12"],
    )

    assert mouse_id == "C12"


def test_parse_mouse_id_accepts_unpadded_filename_for_zero_padded_subject() -> None:
    mouse_id = parse_mouse_id(
        "/tmp/study/Picture WT C3 sample.nd2",
        mouse_ids=["C1", "C03", "C04"],
    )

    assert mouse_id == "C03"


def test_parse_mouse_id_uses_most_specific_token_in_dense_filename() -> None:
    mouse_id = parse_mouse_id(
        "/tmp/study/A1_A17_rep1.nd2",
        mouse_ids=["A1", "A17"],
    )

    assert mouse_id == "A17"
