"""Local queue worker for durable threshold-generation runs."""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from threshold_analysis.batch_processor import process_directory_all_thresholds

from .logging_utils import configure_logging, log_event
from .services.threshold_runner import (
    cache_run_previews,
    fingerprint_run_sources,
    prepare_run_preview_root,
    write_run_metadata,
)
from .state import GlobalState, RunRecord, STATE, utc_now


LOGGER = logging.getLogger(__name__)


class ThresholdWorker:
    def __init__(
        self,
        state: GlobalState = STATE,
        worker_id: Optional[str] = None,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        self.state = state
        self.worker_id = worker_id or f"local-worker-{os.getpid()}"
        self.poll_interval_seconds = poll_interval_seconds
        self._heartbeat_interval_seconds = max(1.0, state.settings.worker_heartbeat_ttl_seconds / 3.0)
        self._stop_event = threading.Event()
        self._status_lock = threading.Lock()
        self._status = "idle"
        self._details: Dict[str, object] = {}
        self._heartbeat_thread: Optional[threading.Thread] = None

    def _set_status(self, status: str, **details: object) -> None:
        with self._status_lock:
            self._status = status
            self._details = details

    def _snapshot_status(self) -> Tuple[str, Dict[str, object]]:
        with self._status_lock:
            return self._status, dict(self._details)

    def _emit_heartbeat(self) -> None:
        status, details = self._snapshot_status()
        self.state.record_worker_heartbeat(self.worker_id, status=status, details=details)

    def _start_heartbeat(self) -> None:
        if self._heartbeat_thread is not None:
            return

        def _run() -> None:
            self._emit_heartbeat()
            while not self._stop_event.wait(self._heartbeat_interval_seconds):
                self._emit_heartbeat()

        self._heartbeat_thread = threading.Thread(target=_run, name="interface-worker-heartbeat", daemon=True)
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        self._stop_event.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=self._heartbeat_interval_seconds + 1.0)
            self._heartbeat_thread = None

    def recover_stale_runs(self) -> int:
        recovered = self.state.mark_stale_running_runs_failed("Worker restarted before job completion")
        if recovered:
            log_event(
                LOGGER,
                "worker_recovered_stale_runs",
                worker_id=self.worker_id,
                project_id=self.state.settings.project_id,
                session_id=self.state.settings.session_id,
                recovered_runs=recovered,
            )
        return recovered

    def run_next_job(self) -> Optional[RunRecord]:
        record = self.state.claim_next_queued_run()
        if record is None:
            return None

        self._set_status("processing", job_id=record.job_id)
        self._emit_heartbeat()
        log_event(
            LOGGER,
            "run_claimed",
            worker_id=self.worker_id,
            job_id=record.job_id,
            study_id=record.study_name,
            project_id=record.project_id,
            session_id=self.state.settings.session_id,
            state=record.state,
        )

        preview_root = prepare_run_preview_root(record)
        record.preview_root = preview_root
        record = self.state.update_run(
            record.job_id,
            preview_root=preview_root,
            progress_completed=0,
            progress_total=record.progress_total,
            message="Processing microscopy files",
        )
        progress_total = record.progress_total

        def _on_progress(completed: int, total: int, label: str) -> None:
            nonlocal progress_total
            progress_total = total
            noun = "file" if total == 1 else "files"
            message = f"Processed {completed}/{total} {noun}"
            if label:
                message = f"{message}: {label}"
            self.state.update_run(
                record.job_id,
                state="running",
                message=message,
                progress_completed=completed,
                progress_total=total,
            )
            log_event(
                LOGGER,
                "run_progress",
                worker_id=self.worker_id,
                job_id=record.job_id,
                study_id=record.study_name,
                project_id=record.project_id,
                session_id=self.state.settings.session_id,
                state="running",
                progress_completed=completed,
                progress_total=total,
                label=label,
            )

        try:
            results = process_directory_all_thresholds(
                input_dir=str(record.input_dir),
                config_path=str(record.config_path),
                output_file=str(record.output_path),
                is_3d=record.is_3d,
                marker=record.marker,
                n_jobs=record.n_jobs,
                max_threshold=record.max_threshold,
                save_intermediate=True,
                progress_callback=_on_progress,
            )
        except Exception as exc:  # pragma: no cover - exercised by dedicated tests
            failed = self.state.update_run(
                record.job_id,
                state="failed",
                message=str(exc),
                completed_at=utc_now(),
            )
            log_event(
                LOGGER,
                "run_failed",
                level=logging.ERROR,
                worker_id=self.worker_id,
                job_id=failed.job_id,
                study_id=failed.study_name,
                project_id=failed.project_id,
                session_id=self.state.settings.session_id,
                state=failed.state,
                error=str(exc),
            )
            self._set_status("idle")
            self._emit_heartbeat()
            return failed

        cached_preview_root = cache_run_previews(record, results)
        if cached_preview_root is not None:
            record.preview_root = cached_preview_root
            self.state.update_run(record.job_id, preview_root=record.preview_root)

        final_mtime, final_hash, final_sources = fingerprint_run_sources(record.input_dir, record.config_path)
        write_run_metadata(record.metadata_path, record, final_mtime, final_hash, final_sources)
        completed = self.state.update_run(
            record.job_id,
            state="succeeded",
            message="Threshold results generated",
            completed_at=utc_now(),
            sources_latest_mtime=final_mtime,
            source_hash=final_hash,
            preview_root=record.preview_root,
            progress_completed=progress_total or 0,
            progress_total=progress_total,
        )
        log_event(
            LOGGER,
            "run_completed",
            worker_id=self.worker_id,
            job_id=completed.job_id,
            study_id=completed.study_name,
            project_id=completed.project_id,
            session_id=self.state.settings.session_id,
            state=completed.state,
            output_path=str(completed.output_path) if completed.output_path else None,
        )
        self._set_status("idle")
        self._emit_heartbeat()
        return completed

    def run_forever(self) -> None:
        self.state.initialize()
        self.recover_stale_runs()
        self._set_status("idle")
        self._start_heartbeat()
        log_event(
            LOGGER,
            "worker_started",
            worker_id=self.worker_id,
            project_id=self.state.settings.project_id,
            session_id=self.state.settings.session_id,
        )
        try:
            while True:
                record = self.run_next_job()
                if record is None:
                    time.sleep(self.poll_interval_seconds)
        except KeyboardInterrupt:  # pragma: no cover
            self._set_status("stopped", reason="keyboard_interrupt")
            self._emit_heartbeat()
        finally:
            log_event(
                LOGGER,
                "worker_stopped",
                worker_id=self.worker_id,
                project_id=self.state.settings.project_id,
                session_id=self.state.settings.session_id,
            )
            self._stop_heartbeat()
            self.state.record_worker_heartbeat(
                self.worker_id,
                status="stopped",
                details={"reason": "worker_shutdown"},
            )


def main() -> int:
    configure_logging()
    STATE.initialize()
    worker = ThresholdWorker()
    worker.run_forever()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
