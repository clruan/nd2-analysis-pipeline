# Outstanding Issues for Next Iteration

These are the follow-ups that remain after the latest preview/statistics refresh. Tackling them next will round out the workflow.

## 1. ND2 Availability Handling
- [x] **Offline hints are still passive** → PreviewPane now surfaces a warning banner with the expected ND2 path plus guidance to remount and refresh.
- [x] **No automatic regeneration** → Added a “Refresh previews” action that clears the active threshold cache and refetches overlays without reloading the study.

## 2. Preview Cache Lifecycle
- [x] **Stale threshold folders**: A rolling retention now keeps the latest threshold folders (and prunes week-old sets) whenever previews regenerate, preventing unbounded growth under `generated_previews/<thresholds>`.
- [x] **Placeholder detection**: Preview PNGs now emit metadata sentinels; cached assets are only reused if their metadata matches the current metric/variant, so corrupted or placeholder files are automatically regenerated.

## 3. Statistical UX Polish
- [x] **Method disclosure**: Analysis cards now surface the exact statistical test (e.g., ordinary ANOVA, Kruskal–Wallis, or pairwise t-tests) in the summary chip once results arrive, mirroring the `testType` selection.

## 4. Preview & Histogram Enhancements
- [x] **Additional samples per group**: Scientists can now select 1–4 subjects per group via the preview pane slider; the API respects the cap.
- [x] **Export ergonomics**: Preview tiles expose download icons and each Plotly card ships an Export PNG action powered by `Plotly.downloadImage`.
- [x] **Configurable ratios**: Channel ratios can now be authored during config creation or updated after loading a study, and the new metrics flow through histograms, statistics, downloads, and previews.

## 5. Build & Performance
- [x] **Plotly bundle size**: Plotly is now lazy-loaded via a factory component, so the heavy bundle only downloads when AnalysisBoard mounts, keeping the default Vite build under the warning threshold.
- [x] **Preview batching**: Raw previews are now cached and reused across threshold changes while mask/overlay regenerate from cached channel stacks, eliminating redundant ND2 reads for large studies.

## 6. Documentation & Onboarding
- [x] Update the user guide to cover the new statistical controls (test mode, significance style, custom pairs).
- [x] Add a short troubleshooting section describing the new preview highlight cues and how to resolve missing ND2 mounts.

## 7. Pending UI Polish
- [ ] Huse a publication-grade palette/fonts controller for scientists.
- [ ] Threshold sliders still assume three channels; detect study channel count and render the correct number of controls and labels.
- **Plan of attack**
  - [ ] Add preset palettes/font stacks plus manual overrides inside the Plot section of the left panel, with live preview swatches.
  - [ ] Surface study channel metadata (or infer from ND2) and dynamically render N sliders with per-channel labels + colors.
  - [ ] Document the new palette + slider behavior in the user guide so collaborators know how to customize figures.

Completing these items will bring the interface in line with the publishing workflow we’re targeting.
