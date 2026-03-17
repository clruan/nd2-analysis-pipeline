"""Tests for persisting study annotations into result artifacts."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from threshold_analysis.data_models import ThresholdResults

if "joblib" not in sys.modules:
    joblib_stub = types.ModuleType("joblib")
    joblib_stub.Parallel = lambda *args, **kwargs: None  # type: ignore[assignment]
    joblib_stub.delayed = lambda func, *args, **kwargs: func  # type: ignore[assignment]
    sys.modules["joblib"] = joblib_stub

if "pyclesperanto" not in sys.modules:
    sys.modules["pyclesperanto"] = types.ModuleType("pyclesperanto")

if "nd2reader" not in sys.modules:
    nd2_stub = types.ModuleType("nd2reader")
    nd2_stub.ND2Reader = object  # type: ignore[attr-defined]
    nd2_stub.Nd2 = object  # type: ignore[attr-defined]
    sys.modules["nd2reader"] = nd2_stub

INTERFACE_ROOT = Path(__file__).resolve().parents[2]
if str(INTERFACE_ROOT) not in sys.path:
    sys.path.append(str(INTERFACE_ROOT))

from threshold_analysis.batch_processor import load_threshold_results, save_threshold_results  # noqa: E402
from api.services.studies import _persist_study_annotations  # noqa: E402
from api.state import StudyRecord  # noqa: E402


def test_persist_study_annotations_updates_result_and_meta_files(tmp_path: Path) -> None:
    result_path = tmp_path / "threshold_results_demo.json"
    result_path.write_text(
        json.dumps(
            {
                "study_name": "demo",
                "group_info": {"Control": ["A1"]},
                "image_data": [],
            }
        ),
        encoding="utf-8",
    )

    ratios = [{"id": "ratio_1_3", "label": "Channel 1 / Channel 3", "numerator_channel": 1, "denominator_channel": 3}]
    channels = [
        {"channel": 1, "label": "DNA", "color": "#3366ff"},
        {"channel": 2, "label": "Marker A", "color": "#ff0000"},
        {"channel": 3, "label": "Marker B", "color": "#00ff00"},
    ]

    record = StudyRecord(
        study_id="demo",
        results=ThresholdResults(study_name="demo", image_data=[], group_info={}),
        source_path=result_path,
        input_dir=tmp_path,
        ratio_definitions=ratios,
        channel_definitions=channels,
        pixel_size_um=0.42,
    )

    _persist_study_annotations(record)

    saved_result = json.loads(result_path.read_text(encoding="utf-8"))
    saved_meta = json.loads(Path(f"{result_path}.meta.json").read_text(encoding="utf-8"))

    assert saved_result["ratio_definitions"] == ratios
    assert saved_result["channel_definitions"] == channels
    assert saved_result["pixel_size_um"] == 0.42
    assert saved_meta["ratio_definitions"] == ratios
    assert saved_meta["channel_definitions"] == channels
    assert saved_meta["pixel_size_um"] == 0.42


def test_threshold_results_round_trip_preserves_metadata(tmp_path: Path) -> None:
    result_path = tmp_path / "threshold_results_roundtrip.json"
    ratios = [{"id": "ratio_2_3", "label": "Channel 2 / Channel 3", "numerator_channel": 2, "denominator_channel": 3}]
    channels = [
        {"channel": 1, "label": "DNA", "color": "#3366ff"},
        {"channel": 2, "label": "Marker A", "color": "#ff0000"},
        {"channel": 3, "label": "Marker B", "color": "#00ff00"},
    ]
    original = ThresholdResults(
        study_name="roundtrip",
        image_data=[],
        group_info={"Control": ["A1"]},
        ratio_definitions=ratios,
        channel_definitions=channels,
        pixel_size_um=0.31,
    )

    save_threshold_results(original, str(result_path))
    loaded = load_threshold_results(str(result_path))

    assert loaded.ratio_definitions == ratios
    assert loaded.channel_definitions == channels
    assert loaded.pixel_size_um == 0.31
