# Interface-Final Issue Status

Status audit updated on 2026-03-18.

This file is no longer just a raw backlog. It now tracks what is already fixed,
what is partially fixed, and what still belongs in upcoming phases.

## Resolved

- [x] **Offline ND2 warning is explicit**: the preview pane now shows a warning
  banner when the original image directory is offline and instructs the user to
  remount it before refreshing previews.
- [x] **Manual preview regeneration exists**: the UI now exposes a `Refresh previews`
  action and a `Flush cache` action so users can rebuild preview assets without
  reloading the full study.
- [x] **Export ergonomics improved**: preview panels can be downloaded directly,
  and analysis plots now expose an `Export PNG` action.
- [x] **Statistical controls are no longer fixed**: users can choose test type,
  significance display, and comparison mode from the left panel.
- [x] **Troubleshooting coverage exists**: missing ND2 mount behavior and preview
  recovery steps are documented in `interface-final/troubleshooting_notes.md`.
- [x] **Run and study state are durable across API restart**: run rows, loaded
  study metadata, and study listings now persist in a local SQLite state store,
  and study records are rehydrated from saved threshold-results files on demand.
- [x] **Long-running threshold jobs no longer depend on FastAPI `BackgroundTasks`**:
  threshold generation now runs through a local queue/worker path with persisted
  `queued` / `running` / `succeeded` / `failed` transitions.
- [x] **Health and readiness endpoints exist**: the API now exposes `/healthz`
  and `/readyz`, with readiness tied to database access, queue writability, and
  worker heartbeats instead of process liveness alone.
- [x] **Backend study service monolith is split**: `interface-final/api/services/studies.py`
  is now a compatibility facade over smaller backend service modules for study
  loading, analysis, statistics, previews, metadata persistence, and downloads.

## Partially Resolved

- [~] **Documentation for statistical controls is partly updated**: the user guide
  includes a statistics section, but some option names and behavior descriptions
  no longer exactly match the current UI.
- [~] **Publication-style visualization controls exist, but are incomplete**:
  there are palette presets, manual color overrides, and font size controls, but
  there is still no full publication-grade typography system or broader figure
  styling surface.
- [~] **Preview regeneration after drive recovery is only manual**: users can
  refresh previews after remounting the ND2 source, but there is still no
  automatic remount detection or automatic rebuild.
- [~] **Observability is improved but still incomplete**: structured JSON logs
  and health/readiness checks exist now, but there is still no error tracking,
  no broader deployment/CI product checks, and no deeper operational dashboards.

## Still Open

### Performance

- [ ] **Preview payload is still oversized**: the client still requests the full
  preview matrix for active thresholds rather than fetching by visible metric or
  visible group column.
- [ ] **Threshold generation still defaults to conservative parallelism**:
  `n_jobs` still defaults to `1`, with no hardware-aware default and no `safe`
  vs `fast` execution preset.
- [ ] **Preview render path is still whole-image only**: large 3D studies would
  still benefit from downsampled tiles or pre-rendered mask atlases.
- [ ] **Parallel-run progress is still coarse**: detailed file-by-file progress
  updates exist only in sequential mode; parallel mode still reports only a
  terminal completion message.

### Preview Cache Integrity

- [ ] **Placeholder/corruption detection is missing**: cached preview reuse is
  metadata-token based, but there is still no checksum or explicit corruption
  detection for broken PNGs.

### UI / Data Model

- [ ] **Three-channel assumptions still dominate the app**: threshold sliders,
  preview controls, metrics, and several store types still assume exactly
  `channel_1`, `channel_2`, and `channel_3`.

### Product Readiness

- [ ] **Auth and project isolation are missing**: the current app still behaves
  like a single-user workstation rather than a productized multi-project system.
- [ ] **Frontend monoliths are still too large**: `LeftPanel.tsx` and
  `AnalysisBoard.tsx` still need to be split into smaller modules as the next
  productization phase.

## Deferred Until After Productization Phases

- [ ] **Study copilot / agent features**
- [ ] **Publication search / literature assistant**
- [ ] **Automatic remount detection**
- [ ] **Dynamic channel-count support across the whole product**

## Next Productization Priority Order

1. Split `interface-final/web/src/components/LeftPanel.tsx`
2. Split `interface-final/web/src/components/AnalysisBoard.tsx`
3. Product hardening: auth/session scaffold, deployment docs, CI/docs cleanup
4. Analysis module expansion
5. Deferred performance and viewer expansion phases
