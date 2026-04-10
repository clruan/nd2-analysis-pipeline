"""Tests for fast threshold-analysis aggregation."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

INTERFACE_ROOT = Path(__file__).resolve().parents[2]
if str(INTERFACE_ROOT) not in sys.path:
    sys.path.append(str(INTERFACE_ROOT))

from api.runtime import RuntimeSettings  # noqa: E402
from api.schemas import AnalyzeRequest  # noqa: E402
from api.services.studies import analyze_study  # noqa: E402
from api.state import STATE, StudyRecord  # noqa: E402
from threshold_analysis.data_models import ThresholdData, ThresholdResults  # noqa: E402


def _threshold_series(value: float) -> np.ndarray:
    return np.full(4096, value, dtype=np.float32)


def test_analyze_study_averages_replicates_per_mouse(tmp_path: Path) -> None:
    STATE.configure(
        RuntimeSettings(
            state_db=(tmp_path / "runtime" / "state.sqlite3").resolve(),
            project_id="test-project",
            session_id="analysis-service",
            worker_heartbeat_ttl_seconds=15,
        )
    )

    record = StudyRecord(
        study_id="demo-study",
        results=ThresholdResults(
            study_name="demo-study",
            group_info={"Control": ["A1"], "Treatment": ["B1"]},
            image_data=[
                ThresholdData(
                    mouse_id="A1",
                    group="Control",
                    filename="a1_rep1.nd2",
                    channel_1_percentages=_threshold_series(10.0),
                    channel_2_percentages=_threshold_series(20.0),
                    channel_3_percentages=_threshold_series(40.0),
                ),
                ThresholdData(
                    mouse_id="A1",
                    group="Control",
                    filename="a1_rep2.nd2",
                    channel_1_percentages=_threshold_series(30.0),
                    channel_2_percentages=_threshold_series(40.0),
                    channel_3_percentages=_threshold_series(80.0),
                ),
                ThresholdData(
                    mouse_id="B1",
                    group="Treatment",
                    filename="b1_rep1.nd2",
                    channel_1_percentages=_threshold_series(50.0),
                    channel_2_percentages=_threshold_series(60.0),
                    channel_3_percentages=_threshold_series(100.0),
                ),
            ],
        ),
        source_path=tmp_path / "threshold_results_demo.json",
        ratio_definitions=[
            {"id": "ratio_1_3", "label": "Channel 1 / Channel 3", "numerator_channel": 1, "denominator_channel": 3}
        ],
    )
    STATE.add_study(record)

    response = analyze_study(
        "demo-study",
        AnalyzeRequest(thresholds={"channel_1": 100, "channel_2": 200, "channel_3": 300}),
    )

    assert len(response.mouse_averages) == 2
    control = next(item for item in response.mouse_averages if item.MouseID == "A1")
    assert control.Channel_1_area == 20.0
    assert control.Channel_2_area == 30.0
    assert control.Channel_3_area == 60.0
    assert control.ratios["ratio_1_3"] == pytest.approx(20.0 / (60.0 + 1e-3))

    assert [item.replicate_index for item in response.individual_images if item.mouse_id == "A1"] == [1, 2]
