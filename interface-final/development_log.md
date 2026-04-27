# Development Log

## 2026-03-27 - Preview group retention + subject ID parsing fix
- Fixed filename subject matching so threshold runs prefer the most specific known ID (`C12` no longer collapses into `C1`) and can still resolve zero-padded config IDs from unpadded filename tokens (`C3` -> `C03`) (`image_processing.py:198`, `interface-final/api/tests/test_mouse_id_parsing.py:1`).
- Updated the live preview pane to keep the loaded study's full group list visible, rendering an explicit `No preview` placeholder instead of silently dropping groups with no generated tiles (`interface-final/web/src/components/PreviewPane.tsx:178`).

## 2026-03-18 - Phase 3 backend monolith breakup
- Split the backend study monolith into focused service modules for shared helpers, study loading/rehydration, analysis, statistics, preview rendering, preview orchestration, study metadata persistence, and downloads, while keeping the existing FastAPI routes and durable runtime model unchanged (`interface-final/api/services/study_common.py`, `interface-final/api/services/study_loader.py`, `interface-final/api/services/analysis_service.py`, `interface-final/api/services/statistics_service.py`, `interface-final/api/services/preview_rendering.py`, `interface-final/api/services/preview_service.py`, `interface-final/api/services/study_metadata_service.py`, `interface-final/api/services/download_service.py`).
- Replaced `interface-final/api/services/studies.py` with a compatibility facade that preserves the import surface used by `api.main`, `threshold_runner`, and existing backend tests, including the private helpers that were already imported directly.
- Added an explicit backend facade-compatibility test so the route-facing `api.services.studies` entrypoints stay stable across future refactors (`interface-final/api/tests/test_studies_facade.py`).
- Current backend suite result: `25 passed` via `interface-final/.venv/bin/python -m pytest interface-final/api/tests -q`.

## 2026-03-17 - Phase 1 + 2 durable state and worker
- Replaced the in-memory backend singleton with a SQLite-backed durable state layer for runs, loaded-study metadata, and worker heartbeats, while keeping preview/analysis caches ephemeral and lazily rebuilt after restart (`interface-final/api/state.py`, `interface-final/api/runtime.py`).
- Removed FastAPI `BackgroundTasks` from threshold generation and added a local queue worker entrypoint that claims queued runs, persists progress and terminal state, recovers stale `running` jobs as failed on startup, and writes worker heartbeats for readiness checks (`interface-final/api/services/threshold_runner.py`, `interface-final/api/worker.py`).
- Added shared structured JSON logging plus `/healthz` and `/readyz` endpoints, and updated the startup script to launch the backend, worker, and frontend together with a shared Python path (`interface-final/api/logging_utils.py`, `interface-final/api/main.py`, `interface-final/start.sh`).
- Persisted loaded-study metadata so `/studies` survives API restart and study records can be rehydrated on demand from saved threshold-results files without changing the existing analysis or preview API contracts (`interface-final/api/services/studies.py`).
- Added backend coverage for run persistence, reuse-existing success, worker success/failure paths, stale-run recovery, durable study reload, and health/readiness behavior. Current backend suite result: `24 passed` via `interface-final/.venv/bin/python -m pytest interface-final/api/tests -q`.

## 2026-03-17 - Phase planning baseline
- Audited `interface-final/unresolved_issues.md` against the current codebase and converted it from a stale backlog into a status tracker with `resolved`, `partially resolved`, `still open`, and `deferred` sections.
- Added `interface-final/PHASE_PLAN.md` so future Codex sessions can execute one phase at a time without relying on prior chat context.
- Established the immediate priority order for productization work: durable state, queue/worker execution, backend monolith breakup, frontend monolith breakup, then product hardening.
- Extended the plan so advanced analysis features (colocalization, denoise, future modules) have their own phase, followed by a separate Napari-inspired viewer expansion phase.

## 2026-03-12 – Reverted preview compaction experiment
- Prototyped a compact preview-header / drag-drop panel-order pass, binary mask preview rendering, and loading-bar-first status treatment across previews, charts, and threshold generation.
- The UI pass was reverted after review, but the experiment is being kept in the log as a recorded iteration rather than an accepted interface change.

## 2026-03-10 – Threshold decoupling + visible progress
- Decoupled preview window edits from analysis invalidation so `Min` / `Max` slider commits now update preview bounds only, while the threshold thumb remains the only slider action that refreshes histogram/statistics data (`interface-final/web/src/components/LeftPanel.tsx:1690`, `interface-final/web/src/state/useAppStore.ts:136`).
- Added visible run progress for threshold generation plus incremental preview readiness feedback in the UI; users now see file-level progress while a batch is running and per-tile loading status as previews refresh (`interface-final/api/services/threshold_runner.py:83`, `interface-final/api/schemas.py:108`, `interface-final/web/src/components/LeftPanel.tsx:1898`, `interface-final/web/src/components/PreviewPane.tsx:719`).
- Made each left-panel workflow module individually collapsible so users can hide sections they are not actively editing without losing the stage-guided flow (`interface-final/web/src/components/LeftPanel.tsx:562`).
- Replaced the per-threshold brute-force loop in threshold generation with a histogram-based pass, which keeps the output identical but removes the `4096 x 3` full-image comparisons previously done for every file (`threshold_analysis/generator.py:12`).
- Simplified the analysis chart back to a static rendering path: no threshold pulse, no autoscale highlighting, and no chart animation. The chart now stays fast and uses only hover-triggered grey replica dots plus a small hover region to keep replica clusters stable while the cursor moves between points (`interface-final/web/src/components/AnalysisBoard.tsx:540`).

## 2026-03-09 – Threshold interaction + preview cache narrowing
- Narrowed preview cache dependencies so unrelated panels now reuse prior PNGs across threshold/range edits; raw, mask, and overlay invalidation is now keyed only by the channels each panel actually depends on (`interface-final/api/services/studies.py:1198`).
- Added per-image cache tokens to the preview API so the frontend only reloads images whose files actually changed instead of busting the entire preview grid on every response (`interface-final/api/schemas.py:200`, `interface-final/web/src/components/PreviewPane.tsx:719`).
- Reworked Min / Threshold / Max editing to use local drafts with commit-on-blur / Enter / slider release, which avoids expensive intermediate requests while typing and allows thresholds to temporarily exceed the current display window before the final commit normalizes it (`interface-final/web/src/components/LeftPanel.tsx:94`).
- Threshold result JSON files now carry `ratio_definitions`, `channel_definitions`, and `pixel_size_um`, and study-side edits write back to both the result file and `.meta.json` so reloading remains stable even after files are moved (`threshold_analysis/data_models.py:35`, `threshold_analysis/batch_processor.py:169`, `interface-final/api/services/studies.py:1657`).

## 2025-11-04 – Preview pipeline refresh
- Reworked preview generation caching so raw images persist across threshold changes while masks and overlays regenerate per metric (`interface-final/api/services/studies.py:554`).
- Added metric and channel metadata to preview API contracts to drive richer layouts (`interface-final/api/schemas.py:140`, `interface-final/web/src/api/types.ts:86`).
- Updated the preview pane UI to stack rows per metric, constrain preview widths, and align group cards side by side with horizontal scrolling (`interface-final/web/src/components/PreviewPane.tsx:9`).

## 2025-11-04 – UI workflow improvements
- Preview API now serves all channel metrics at once with metric identifiers and consolidated caching (`interface-final/api/services/studies.py:554`).
- Live preview grid renders compact rows with shared raw/mask/overlay labels, subject details moved to tooltips, and all channel/ratio views visible simultaneously (`interface-final/web/src/components/PreviewPane.tsx:1`).
- Added file upload endpoints and UI so configs and threshold results can be selected via system dialogs instead of manual paths (`interface-final/api/main.py:74`, `interface-final/web/src/components/LeftPanel.tsx:1`).
- Introduced a collapsible sidebar toggle to reclaim horizontal space when adjusting previews (`interface-final/web/src/App.tsx:1`).

## 2026-03-07 – Grouping strategy + channel metadata
- Added scan-time `subject_strategy` with `per_file` default and optional `auto` inference to prevent accidental subject collapsing in replicate-heavy studies (`interface-final/api/schemas.py:18`, `interface-final/api/services/configurator.py:33`).
- Introduced normalized channel definitions (`channel`, `label`, `color`) in configs, run metadata, and loaded-study state so channel semantics survive reloads (`data_models.py:10`, `interface-final/api/services/channels.py:1`, `interface-final/api/services/threshold_runner.py:39`).
- Added per-study channel update endpoints and UI controls to rename channels / change pseudocolors after loading results; updates persist to `.meta.json` and refresh preview rendering (`interface-final/api/main.py:173`, `interface-final/api/services/studies.py:1652`, `interface-final/web/src/components/LeftPanel.tsx:2070`).
- Updated preview and export labeling to use channel metadata, including pseudocolor-aware composite rendering for raw/mask/overlay previews and downloaded panels (`interface-final/api/services/studies.py:2115`, `visualization.py:63`, `interface-final/web/src/components/AnalysisBoard.tsx:274`).
