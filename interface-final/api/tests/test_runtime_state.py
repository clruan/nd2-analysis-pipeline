"""Tests for durable backend state, worker execution, and readiness endpoints."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException

from threshold_analysis.batch_processor import save_threshold_results
from threshold_analysis.data_models import ThresholdData, ThresholdResults

from api import main as main_module
from api.runtime import RuntimeSettings
from api.schemas import AnalyzeRequest, LoadStudyRequest
from api.services import threshold_runner
from api.services.studies import analyze_study
from api.state import RunRecord, STATE, utc_now
from api.worker import ThresholdWorker


def _make_settings(tmp_path: Path, session_id: str) -> RuntimeSettings:
    return RuntimeSettings(
        state_db=(tmp_path / "runtime" / "state.sqlite3").resolve(),
        project_id="test-project",
        session_id=session_id,
        worker_heartbeat_ttl_seconds=15,
    )


def _configure_state(tmp_path: Path, session_id: str = "session-a") -> RuntimeSettings:
    settings = _make_settings(tmp_path, session_id)
    STATE.configure(settings)
    return settings


def _make_threshold_entry(mouse_id: str = "A1", group: str = "Control", filename: str = "sample.nd2") -> ThresholdData:
    values = np.zeros(4096, dtype=np.float32)
    values[0] = 12.5
    return ThresholdData(
        mouse_id=mouse_id,
        group=group,
        filename=filename,
        channel_1_percentages=values.copy(),
        channel_2_percentages=values.copy(),
        channel_3_percentages=values.copy(),
    )


def _write_results_file(path: Path) -> Path:
    results = ThresholdResults(
        study_name="demo-study",
        image_data=[_make_threshold_entry()],
        group_info={"Control": ["A1"]},
    )
    save_threshold_results(results, str(path))
    return path


def _call_route(app, path: str, method: str = "GET", **kwargs):
    route = next(
        route
        for route in app.routes
        if getattr(route, "path", None) == path and method.upper() in getattr(route, "methods", set())
    )
    return asyncio.run(route.endpoint(**kwargs))


def test_run_rows_survive_state_reconfigure(tmp_path: Path) -> None:
    settings = _configure_state(tmp_path)
    base_fields = {
        "project_id": settings.project_id,
        "session_id": settings.session_id,
        "input_dir": tmp_path,
        "config_path": tmp_path / "config.json",
        "output_path": tmp_path / "output.json",
        "study_name": "demo",
        "metadata_path": tmp_path / "output.json.meta.json",
    }
    records = [
        RunRecord(job_id="queued-job", state="queued", message="queued", **base_fields),
        RunRecord(job_id="running-job", state="running", message="running", **base_fields),
        RunRecord(job_id="succeeded-job", state="succeeded", message="done", completed_at=utc_now(), **base_fields),
        RunRecord(job_id="failed-job", state="failed", message="boom", completed_at=utc_now(), **base_fields),
    ]
    for record in records:
        STATE.record_run(record)

    STATE.configure(settings.with_session("session-b"))

    assert STATE.get_run("queued-job").state == "queued"
    assert STATE.get_run("running-job").state == "running"
    assert STATE.get_run("succeeded-job").state == "succeeded"
    assert STATE.get_run("failed-job").state == "failed"


def test_launch_threshold_run_reuse_existing_persists_succeeded_run(tmp_path: Path, monkeypatch) -> None:
    settings = _configure_state(tmp_path)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    nd2_path = input_dir / "sample.nd2"
    nd2_path.write_bytes(b"nd2")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"groups": {"Control": ["A1"]}}), encoding="utf-8")
    output_path = tmp_path / "threshold_results_demo.json"
    output_path.write_text("{}", encoding="utf-8")
    meta_path = Path(f"{output_path}.meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "source_hash": "test-hash",
                "latest_source_mtime": 1.0,
                "ratio_definitions": [],
                "channel_definitions": [
                    {"channel": 1, "label": "Channel 1", "color": "#00ff00"},
                    {"channel": 2, "label": "Channel 2", "color": "#ff0000"},
                    {"channel": 3, "label": "Channel 3", "color": "#0000ff"},
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(threshold_runner, "_fingerprint_sources", lambda *_args: (1.0, "test-hash", []))
    monkeypatch.setattr(threshold_runner, "find_nd2_files", lambda *_args: [nd2_path])

    status = threshold_runner.launch_threshold_run(
        threshold_runner.ThresholdRunRequest(
            input_dir=str(input_dir),
            config_path=str(config_path),
            output_path=str(output_path),
            reuse_existing=True,
        )
    )

    persisted = STATE.get_run(status.job_id)
    assert status.state == "succeeded"
    assert persisted.state == "succeeded"
    assert persisted.project_id == settings.project_id
    assert persisted.session_id == settings.session_id


def test_launch_threshold_run_rejects_missing_non_nd2_reader_dependency(tmp_path: Path, monkeypatch) -> None:
    _configure_state(tmp_path)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    oib_path = input_dir / "sample.oib"
    oib_path.write_bytes(b"oib")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"groups": {"Control": ["A1"]}}), encoding="utf-8")

    monkeypatch.setattr(threshold_runner, "find_nd2_files", lambda *_args: [oib_path])
    monkeypatch.setattr(
        threshold_runner,
        "ensure_microscopy_reader_dependencies",
        lambda _paths: (_ for _ in ()).throw(
            ImportError(
                "Found microscopy files with extensions .oib, but 'aicsimageio' is not installed "
                "in the active Python environment. Install it to process .czi/.oib/.oif files."
            )
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        threshold_runner.launch_threshold_run(
            threshold_runner.ThresholdRunRequest(
                input_dir=str(input_dir),
                config_path=str(config_path),
                reuse_existing=False,
            )
        )

    assert exc_info.value.status_code == 400
    assert "aicsimageio" in exc_info.value.detail
    assert ".oib" in exc_info.value.detail


def test_worker_executes_queued_runs_and_persists_success(tmp_path: Path, monkeypatch) -> None:
    settings = _configure_state(tmp_path)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    record = RunRecord(
        job_id="job-success",
        project_id=settings.project_id,
        session_id=settings.session_id,
        state="queued",
        input_dir=input_dir,
        config_path=config_path,
        output_path=tmp_path / "result.json",
        study_name="demo-study",
        metadata_path=tmp_path / "result.json.meta.json",
        preview_root=tmp_path / "previews" / "job-success",
    )
    STATE.record_run(record)

    def _fake_process_directory_all_thresholds(**kwargs):
        callback = kwargs["progress_callback"]
        callback(1, 2, "file-a")
        callback(2, 2, "file-b")
        return ThresholdResults(
            study_name="demo-study",
            image_data=[_make_threshold_entry()],
            group_info={"Control": ["A1"]},
        )

    monkeypatch.setattr("api.worker.process_directory_all_thresholds", _fake_process_directory_all_thresholds)
    monkeypatch.setattr("api.worker.cache_run_previews", lambda *_args: record.preview_root)
    monkeypatch.setattr("api.worker.fingerprint_run_sources", lambda *_args: (5.0, "final-hash", []))
    monkeypatch.setattr("api.worker.write_run_metadata", lambda *_args: None)

    worker = ThresholdWorker(state=STATE, worker_id="worker-test")
    completed = worker.run_next_job()

    assert completed is not None
    assert completed.state == "succeeded"
    assert completed.progress_completed == 2
    assert completed.progress_total == 2
    assert STATE.get_run("job-success").state == "succeeded"


def test_worker_persists_failures_and_recovers_stale_runs(tmp_path: Path, monkeypatch) -> None:
    settings = _configure_state(tmp_path)
    queued = RunRecord(
        job_id="job-fail",
        project_id=settings.project_id,
        session_id=settings.session_id,
        state="queued",
        input_dir=tmp_path,
        config_path=tmp_path / "config.json",
        output_path=tmp_path / "result.json",
        study_name="demo-study",
        metadata_path=tmp_path / "result.json.meta.json",
        preview_root=tmp_path / "previews" / "job-fail",
    )
    stale = RunRecord(
        job_id="job-stale",
        project_id=settings.project_id,
        session_id=settings.session_id,
        state="running",
        input_dir=tmp_path,
        config_path=tmp_path / "config.json",
        output_path=tmp_path / "stale.json",
        study_name="demo-study",
        metadata_path=tmp_path / "stale.json.meta.json",
    )
    STATE.record_run(queued)
    STATE.record_run(stale)

    monkeypatch.setattr("api.worker.process_directory_all_thresholds", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    worker = ThresholdWorker(state=STATE, worker_id="worker-test")
    recovered = worker.recover_stale_runs()
    failed = worker.run_next_job()

    assert recovered == 1
    assert STATE.get_run("job-stale").state == "failed"
    assert failed is not None
    assert failed.state == "failed"
    assert "boom" in (failed.message or "")


def test_loaded_study_survives_restart_and_can_be_analyzed_via_api(tmp_path: Path) -> None:
    settings = _configure_state(tmp_path)
    results_path = _write_results_file(tmp_path / "threshold_results_demo-study.json")

    app = main_module.create_app()
    payload = _call_route(
        app,
        "/studies/load",
        method="POST",
        request=LoadStudyRequest(file_path=str(results_path)),
    )
    study_id = payload["study_id"]

    listed = _call_route(app, "/studies")
    assert study_id in listed

    STATE.configure(settings.with_session("session-b"))

    restarted_app = main_module.create_app()
    listed_after_restart = _call_route(restarted_app, "/studies")
    assert study_id in listed_after_restart

    analysis_payload = analyze_study(
        study_id,
        AnalyzeRequest(
            thresholds={"channel_1": 0, "channel_2": 0, "channel_3": 0},
            analysis_mode="positive_area_percent",
        ),
    )
    assert analysis_payload.study_id == study_id
    assert len(analysis_payload.mouse_averages) == 1


def test_status_healthz_and_readyz_behavior(tmp_path: Path) -> None:
    _configure_state(tmp_path)
    app = main_module.create_app()

    status_response = _call_route(app, "/status")
    health_response = _call_route(app, "/healthz")
    ready_before = _call_route(app, "/readyz")

    assert status_response == {"status": "ok"}
    assert health_response.status_code == 200
    assert json.loads(health_response.body)["database"] == "ok"
    assert ready_before.status_code == 503
    assert json.loads(ready_before.body)["ready"] is False

    STATE.record_worker_heartbeat("worker-ready", status="idle", details={"mode": "test"})
    ready_after = _call_route(app, "/readyz")

    assert ready_after.status_code == 200
    assert json.loads(ready_after.body)["ready"] is True
