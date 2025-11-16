"""Tests for subject tokenization during study loading."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np

from threshold_analysis.data_models import ThresholdData, ThresholdResults

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

from api.services.studies import _retokenize_mouse_ids  # noqa: E402


def _make_threshold_entry(mouse_id: str, group: str, filename: str) -> ThresholdData:
    zeros = np.zeros(4096, dtype=np.float32)
    return ThresholdData(
        mouse_id=mouse_id,
        group=group,
        filename=filename,
        channel_1_percentages=zeros.copy(),
        channel_2_percentages=zeros.copy(),
        channel_3_percentages=zeros.copy(),
    )


def test_retokenize_mouse_ids_prefers_specific_tokens() -> None:
    results = ThresholdResults(
        study_name="tokenizer",
        group_info={
            "Heterozygous +TNF": ["A1", "A12", "A18", "A9"],
            "Contol C57 No Treatment": ["C01"],
        },
        image_data=[
            _make_threshold_entry("A1", "Heterozygous +TNF", "Picture A12 sample.nd2"),
            _make_threshold_entry("A1", "Heterozygous +TNF", "Picture A18 sample.nd2"),
            _make_threshold_entry("A1", "Heterozygous +TNF", "Picture A1 sample.nd2"),
            _make_threshold_entry("c1", "Contol C57 No Treatment", "Picture c01 sample.nd2"),
        ],
    )

    _retokenize_mouse_ids(results)

    subjects = [entry.mouse_id for entry in results.image_data]
    assert subjects[0] == "A12"
    assert subjects[1] == "A18"
    assert subjects[2] == "A1"
    assert subjects[3] == "C01"


def test_retokenize_mouse_ids_restores_missing_mice_from_sample_study() -> None:
    sample_path = (
        INTERFACE_ROOT
        / "api"
        / "services"
        / "generated_results"
        / "threshold_results_lung-thrombosis-joan-beckman-09052025-20x2-vimentin-g-axl-r-dapi-b.json"
    )
    if not sample_path.exists():  # pragma: no cover - optional data file
        raise RuntimeError(f"Sample study not found: {sample_path}")

    results = _load_results_from_json(sample_path)
    _retokenize_mouse_ids(results)

    observed = {}
    for entry in results.image_data:
        bucket = observed.setdefault(entry.group, set())
        bucket.add(entry.mouse_id)

    het_tnf_subjects = set(results.group_info.get("Heterozygous +TNF", []))
    het_no_tnf_subjects = set(results.group_info.get("Heterozygous No TNF", []))

    assert het_tnf_subjects.issubset(observed.get("Heterozygous +TNF", set()))
    assert het_no_tnf_subjects.issubset(observed.get("Heterozygous No TNF", set()))


def test_zero_padded_subjects_do_not_merge_groups() -> None:
    results = ThresholdResults(
        study_name="zero-pad",
        group_info={
            "Contol C57 No Treatment": ["C01", "C02"],
            "Control WT +TNF": ["C1", "C2"],
        },
        image_data=[
            _make_threshold_entry("C1", "Control WT +TNF", "Picture WT C1 sample.nd2"),
            _make_threshold_entry("C02", "Contol C57 No Treatment", "Picture C02 sample.nd2"),
        ],
    )

    _retokenize_mouse_ids(results)

    groups = {entry.mouse_id: entry.group for entry in results.image_data}
    assert groups["C1"] == "Control WT +TNF"
    assert groups["C02"] == "Contol C57 No Treatment"


def _load_results_from_json(path: Path) -> ThresholdResults:
    raw = json.loads(path.read_text())
    image_entries = []
    for entry in raw.get("image_data", []):
        image_entries.append(
            ThresholdData(
                mouse_id=entry["mouse_id"],
                group=entry["group"],
                filename=entry["filename"],
                channel_1_percentages=np.array(entry["channel_1_percentages"], dtype=np.float32),
                channel_2_percentages=np.array(entry["channel_2_percentages"], dtype=np.float32),
                channel_3_percentages=np.array(entry["channel_3_percentages"], dtype=np.float32),
            )
        )
    return ThresholdResults(
        study_name=raw.get("study_name", "sample"),
        image_data=image_entries,
        group_info=raw.get("group_info", {}),
    )
