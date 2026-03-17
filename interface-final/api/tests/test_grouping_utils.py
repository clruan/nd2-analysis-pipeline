"""Tests for scan-time subject inference."""

from __future__ import annotations

from pathlib import Path
import sys

INTERFACE_ROOT = Path(__file__).resolve().parents[2]
if str(INTERFACE_ROOT) not in sys.path:
    sys.path.append(str(INTERFACE_ROOT))

from api.schemas import ConfigScanRequest  # noqa: E402
from api.utils import assign_subject_ids, infer_subject_ids, normalize_path  # noqa: E402


def test_infer_subject_ids_prefers_varying_prefix_over_constant_marker() -> None:
    files = [
        Path("/tmp/study/HbAA/Set#1 20X4 HbAA Media Histrulline H3.nd2"),
        Path("/tmp/study/HbAA/Set#2 20X4 HbAA Media Histrulline H3.nd2"),
        Path("/tmp/study/HbAA/Set#3 20X4 HbAA Media Histrulline H3.nd2"),
        Path("/tmp/study/HbAA/Set#4 20X4 HbAA Media Histrulline H3.nd2"),
        Path("/tmp/study/HbSS/Set#1 20X4 HbSS PMA Histrulline H3.nd2"),
        Path("/tmp/study/HbSS/Set#2 20X4 HbSS PMA Histrulline H3.nd2"),
        Path("/tmp/study/HbSS/Set#3 20X4 HbSS PMA Histrulline H3.nd2"),
        Path("/tmp/study/HbSS/Set#4 20X4 HbSS PMA Histrulline H3.nd2"),
    ]

    inferred = infer_subject_ids(files)
    assert sorted(set(inferred.values())) == ["Set1", "Set2", "Set3", "Set4"]


def test_assign_subject_ids_defaults_to_per_file() -> None:
    files = [
        Path("/tmp/study/HbAA/Set#1 20X4 HbAA Media Histrulline H3.nd2"),
        Path("/tmp/study/HbAA/Set#2 20X4 HbAA Media Histrulline H3.nd2"),
    ]

    inferred = assign_subject_ids(files)
    assert len(set(inferred.values())) == len(files)


def test_infer_subject_ids_falls_back_to_unique_per_file_when_ambiguous() -> None:
    files = [
        Path("/tmp/study/groupA/Field A marker H3.nd2"),
        Path("/tmp/study/groupA/Field B marker H3.nd2"),
        Path("/tmp/study/groupB/Field C marker H3.nd2"),
    ]

    inferred = infer_subject_ids(files)
    assert len(set(inferred.values())) == len(files)


def test_normalize_path_handles_unbalanced_pasted_quotes() -> None:
    raw = "'/Volumes/user/User Data Goes Here/Julia Nguyen/BM Neutrophil  Cytospin/H3-Citrulline+ MPO+Dapi"
    normalized = normalize_path(raw)
    assert str(normalized) == "/Volumes/user/User Data Goes Here/Julia Nguyen/BM Neutrophil  Cytospin/H3-Citrulline+ MPO+Dapi"


def test_config_scan_request_defaults_to_per_file_strategy() -> None:
    request = ConfigScanRequest(input_dir="/tmp/study")
    assert request.subject_strategy == "per_file"
