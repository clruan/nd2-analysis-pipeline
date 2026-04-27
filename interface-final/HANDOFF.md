# Interface-Final Handoff

Updated on 2026-03-18.

## Current Status

- Phase 1, Phase 2, and Phase 3 from `interface-final/PHASE_PLAN.md` are implemented.
- Backend state is now durable through a local SQLite store:
  - threshold runs persist across API restart
  - loaded-study metadata persists across API restart
  - `/studies` is sourced from durable state and study records are rehydrated on demand
- Threshold generation no longer depends on FastAPI `BackgroundTasks`:
  - `POST /runs/threshold` queues work durably
  - `python -m api.worker` claims queued jobs and persists progress/state
  - stale `running` jobs are marked failed when the worker starts
- Structured JSON logging plus `/healthz` and `/readyz` are in place.
- `interface-final/start.sh` now launches backend, worker, and frontend together.
- The backend study service layer is now split into focused modules:
  - `study_common.py`
  - `study_loader.py`
  - `analysis_service.py`
  - `statistics_service.py`
  - `preview_rendering.py`
  - `preview_service.py`
  - `study_metadata_service.py`
  - `download_service.py`
  - `studies.py` remains as the compatibility facade for existing imports
- Backend API tests currently pass with:
  - `interface-final/.venv/bin/python -m pytest interface-final/api/tests -q`
  - latest result: `25 passed`

## Next Recommended Task

Start Phase 4 from `interface-final/PHASE_PLAN.md`:

- split `interface-final/web/src/components/LeftPanel.tsx` into smaller workflow sections
- split `interface-final/web/src/components/AnalysisBoard.tsx` into smaller rendering/helpers
- keep the current UI workflow and polling behavior unchanged while reducing component size

## Recommended Files To Read First

- `interface-final/HANDOFF.md`
- `interface-final/PHASE_PLAN.md`
- `interface-final/unresolved_issues.md`
- `interface-final/development_log.md`
- `interface-final/api/services/studies.py`
- `interface-final/web/src/components/LeftPanel.tsx`
- `interface-final/web/src/components/AnalysisBoard.tsx`

## Notes For The Next Codex Session

- Do not start Phase 5 or later phases yet.
- Preserve the existing frontend polling contract for run status.
- Keep the SQLite-backed runtime state, worker flow, and `api.services.studies`
  compatibility facade intact unless Phase 4 needs only a small import cleanup.
- Do not add agent or literature-search features yet.
