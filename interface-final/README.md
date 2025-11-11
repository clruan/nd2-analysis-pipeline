# Interface-Final Station

This directory contains the all-in-one interactive threshold analysis station.

- `api/` hosts the FastAPI backend that will orchestrate study discovery,
  threshold generation, statistical analysis, and live preview rendering.
- `web/` is a Vite + React TypeScript client inspired by Napari that will
  provide the full mouse-driven workflow.

## Research Motivation & Contributions

| Theme | Contribution |
| --- | --- |
| **Interactive threshold science** | Pre-compute 4k threshold curves for every replicate, then expose them through synchronized sliders, live boxplots, and ND2 overlays so scientists can reason about stability rather than single static values. |
| **Preview resilience** | Hybrid caching (NPY planes + PNG fallbacks) plus the new ND2 availability watcher and refresh action keep previews trustworthy when shared drives are unplugged or remounted mid-session. |
| **Statistical transparency** | Analysts can now disclose the exact test (Welch t-test, Kruskal–Wallis, etc.) alongside significance bands, export the figures directly to PNG, and document the configuration alongside regenerated Excel summaries. |
| **Sample-aware exploration** | A publications-grade workflow needs to contrast multiple biological replicates. The interface now lets reviewers pull 1–4 mice per group into the preview wall while highlighting jittered replicates in the plots. |

These pillars align with the types of arguments IEEE VIS/CHI favor: problem framing around domain pain points, novel interaction techniques (threshold sweeps + live previews), and instrumentation for rigorous evaluation (exportable artifacts, transparent statistics).

## Quick start

```bash
# Backend environment
python3 -m venv interface-final/api/venv
source interface-final/api/venv/bin/activate
pip install -r interface-final/api/requirements.txt
# run from the repository root
# Launch from the repository root so the interface package resolves correctly
uvicorn --app-dir interface-final api.main:app --reload --port 8100

# Frontend environment
npm --prefix interface-final/web install
npm --prefix interface-final/web run dev
```

Open the frontend at http://localhost:5173/ and point it at the running API
(`http://localhost:8100`).

## Workflow

1. **Scan input directory** – automatically enumerate groups, subjects and
   replicas from an ND2 dataset, then tweak the generated group mapping in the
   JSON editor.
2. **Create or load configs** – save a new configuration (including optional
   pixel size) or load an existing config to continue editing.
3. **Launch threshold generation** – run the original batch processor and track
   progress directly in the UI.
4. **Load threshold results** – switch to interactive mode with live boxplots,
   statistical overlays, multi-subject previews (raw/mask/overlay), and per-channel/ratio exports.
5. **Download artefacts** – export regenerated Excel summaries that reflect the
   current slider-controlled thresholds and save publication-ready PNGs from either the plot cards or the preview grid.

### Interaction Details

- **Samples per group selector** – pull up to four subjects per treatment into the preview wall when you need to cross-check outliers or 3D stacks.
- **ND2 root override on load** – when reopening historical `threshold_results_*.json` files, supply the ND2 Input Directory (even if the images live in nested folders) and the backend will recursively index every `.nd2` beneath that path for previews.
- **ND2 availability banner** – if the source drive disappears, the interface surfaces a warning with the expected mount point and offers a one-click “Refresh previews” action that rebuilds overlays as soon as the drive returns.
- **Preview refresh & flush controls** – refresh the active threshold folder or wipe the entire cache without leaving the browser; both operations automatically retrigger the preview API so you see the regenerated PNGs.
- **Per-group preview overrides** – keep a global subject cap while optionally pinning individual groups to show more (or fewer) mice directly from the preview pane.
- **Custom ratio editor** – define arbitrary channel ratios when creating configs or after loading a study; the new metrics cascade through charts, stats, downloads, and previews automatically.
- **One-click exports** – every preview tile now includes a download icon, and each Plotly-based analysis card exposes an `Export PNG` button that invokes `Plotly.downloadImage` with consistent sizing.

### Study Workflow (Detailed)

1. **Scan & curate groups**  
   Use the Study Pipeline panel to point at any ND2 directory (local disk or mounted share). The FastAPI service enumerates groups and replicas, populates the JSON editor, and lets you reconcile mis-labeled mice before a single threshold is computed.
2. **Persist configs & metadata**  
   Save the curated grouping, optional pixel size, and any known ratios into a generated config JSON. Uploading a prior config is also supported for longitudinal studies.
3. **Launch batch thresholds**  
   Kick off threshold generation directly from the interface; every job is tracked (status, source hash, output path) so you can rehydrate context after restarts.
4. **Load results & explore**  
   Once a `threshold_results_*.json` is ready, the analysis view hydrates mouse-level tables, subject-level previews, and statistical overlays. The preview controls now support multi-subject sampling and force-refresh so imaging changes are reflected immediately.
5. **Communicate & export**  
   Download synchronized Excel workbooks, save PNGs for figures (plots or overlays), and copy the documented test details to your manuscript. This closes the loop for VIS/CHI reproducibility packages.
