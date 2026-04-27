# Interface-Final Phase Plan

Updated on 2026-03-17.

This plan is designed so each phase can be handed to Codex as a separate task.
Do not start agent features before Phases 1-7 are complete.

## Execution Rules

- Run one implementation phase at a time on the same branch.
- Do not run overlapping refactor phases simultaneously.
- At the end of every phase:
  - update `interface-final/development_log.md`
  - update `interface-final/unresolved_issues.md`
  - record the next step in a short `interface-final/HANDOFF.md`
- Preserve behavior first. Refactor second. Expand scope only after tests pass.

## Phase 0 - Planning Baseline

Status: completed in the current turn.

Outputs:
- corrected `interface-final/unresolved_issues.md`
- created `interface-final/PHASE_PLAN.md`

## Phase 1 - Durable State And Service Boundaries

Goal:
- remove the most fragile workstation assumptions without changing the user-facing workflow

Scope:
- replace in-memory run/study state with durable persistence
- add structured logging
- add health/readiness endpoints
- introduce explicit project/session boundaries in the backend model

Primary files:
- `interface-final/api/state.py`
- `interface-final/api/main.py`
- `interface-final/api/services/threshold_runner.py`
- new persistence modules under `interface-final/api/services/` or `interface-final/api/`

Acceptance criteria:
- run status survives API restart
- loaded study metadata survives API restart
- logs are structured enough to trace a run lifecycle
- health endpoint reports app readiness, not just process liveness

Recommended Codex settings:
- model: newest non-mini Codex model available
- effort: high
- plan: on

## Phase 2 - Queue / Worker For Long Jobs

Goal:
- move threshold generation out of FastAPI process memory and request lifecycle

Scope:
- replace `BackgroundTasks` for threshold generation
- add queue-backed run state transitions
- preserve current UI polling behavior initially

Primary files:
- `interface-final/api/services/threshold_runner.py`
- `interface-final/api/main.py`
- queue/worker modules to be added

Acceptance criteria:
- threshold runs continue if the API process restarts
- failed jobs are persisted as failed, not lost
- current frontend can still poll job status

Recommended Codex settings:
- model: newest non-mini Codex model available
- effort: high
- plan: on

## Phase 3 - Backend Monolith Breakup

Goal:
- reduce `studies.py` into maintainable service modules without changing behavior

Split target:
- `study_loader.py`
- `analysis_service.py`
- `statistics_service.py`
- `preview_service.py`
- `download_service.py`
- `study_metadata_service.py`
- optional low-level `preview_rendering.py`

Primary source:
- `interface-final/api/services/studies.py`

Acceptance criteria:
- public API behavior stays the same
- tests still pass
- file sizes are materially smaller and responsibilities are clearer

Recommended Codex settings:
- model: newest non-mini Codex model available
- effort: high
- plan: on

## Phase 4 - Frontend Monolith Breakup

Goal:
- split oversized React components by workflow section and rendering concern

Split targets:
- `LeftPanel.tsx` into:
  - `ScanSection`
  - `ConfigBuilderSection`
  - `AnalysisControlsSection`
  - `VisualizationSettingsSection`
  - `ThresholdGenerationSection`
  - `StudyLoaderSection`
- `AnalysisBoard.tsx` into:
  - `MetricCard`
  - `SignificanceOverlay`
  - `figureExport.ts`
  - `useMetricDescriptors.ts`
  - `usePlotData.ts`

Primary files:
- `interface-final/web/src/components/LeftPanel.tsx`
- `interface-final/web/src/components/AnalysisBoard.tsx`

Acceptance criteria:
- no workflow regression
- component responsibilities are clearer
- repeated logic moves into hooks/helpers instead of staying inline

Recommended Codex settings:
- model: newest non-mini Codex model available
- effort: high
- plan: on

## Phase 5 - Product Hardening

Goal:
- make the app easier to operate as a real product rather than a local station

Scope:
- auth/session scaffold
- project isolation
- CI test command and documentation cleanup
- deployment notes
- observability polish

Primary files:
- `interface-final/api/main.py`
- `interface-final/start.sh`
- new docs/config files as needed

Acceptance criteria:
- there is a documented deployment path
- there is a standard backend test command
- there is a standard frontend verification command
- there is a clear boundary for projects/sessions

Recommended Codex settings:
- model: newest non-mini Codex model available
- effort: medium
- plan: on

## Phase 6 - Analysis Module Expansion

Goal:
- expose and productize higher-value image-analysis modules before attempting any broad agent work

Scope:
- expose existing colocalization analysis paths through the API and UI
- expose denoise controls as explicit preprocessing settings
- add an analysis-mode selector instead of hardcoding a single mode
- persist module settings with study metadata and export provenance
- define a repeatable module pattern for future analysis additions

Examples for this phase:
- coloc intensity
- coloc overlap
- optional denoise before metric calculation
- future morphology / segmentation-oriented analysis modules

Primary files:
- `interface-final/api/schemas.py`
- `interface-final/api/services/studies.py`
- `interface-final/web/src/api/hooks.ts`
- `interface-final/web/src/components/LeftPanel.tsx`
- `interface-final/web/src/components/AnalysisBoard.tsx`

Acceptance criteria:
- users can select among supported analysis modes from the UI
- denoise settings are explicit and persisted
- colocalization metrics are visible, testable, and exportable
- the implementation pattern is reusable for future modules

Recommended Codex settings:
- model: newest non-mini Codex model available
- effort: medium
- plan: on

## Phase 7 - Deferred Performance / UX Cleanup

Goal:
- clean up the non-blocking but still open performance and UX issues

Scope:
- shrink preview payloads by visible scope
- add `safe` vs `fast` execution presets
- improve parallel progress reporting
- add preview cache integrity markers
- continue reducing three-channel assumptions where safe

Recommended Codex settings:
- model: newest non-mini Codex model available
- effort: medium
- plan: on

## Phase 8 - Napari-Inspired Viewer Expansion

Goal:
- bring the product closer to the highest-value day-to-day Napari workflows without chasing full desktop parity

Scope:
- layer list and per-layer visibility
- per-layer opacity, blending, contrast limits, and colormap controls
- z-slice and stack navigation
- richer overlay and annotation tools
- measurements, ROI helpers, and keyboard shortcuts
- lazy loading / tiled rendering for larger image volumes

Non-goals:
- full Napari plugin compatibility
- arbitrary Python plugin execution inside the browser
- exact feature parity with the desktop app

Acceptance criteria:
- the viewer supports the most common review workflows directly in the browser
- layer state is explicit and testable
- large-image interaction is still responsive
- architecture leaves room for more viewer tools later

Recommended Codex settings:
- model: newest non-mini Codex model available
- effort: high
- plan: on

## After Phase 8

Only then consider:
- study copilot / agent design
- literature assistant
- dynamic channel-count redesign

## Suggested Prompt Footer For Every Future Codex Phase

Use this at the end of every implementation prompt:

```text
Before finishing:
- update interface-final/development_log.md
- update interface-final/unresolved_issues.md if issue status changed
- create or update interface-final/HANDOFF.md with:
  - summary of changes
  - files touched
  - tests run
  - remaining blockers
  - exact next recommended task
Run relevant tests and report the result.
```
