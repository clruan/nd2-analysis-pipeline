# Development Log

## 2025-11-04 – Preview pipeline refresh
- Reworked preview generation caching so raw images persist across threshold changes while masks and overlays regenerate per metric (`interface-final/api/services/studies.py:554`).
- Added metric and channel metadata to preview API contracts to drive richer layouts (`interface-final/api/schemas.py:140`, `interface-final/web/src/api/types.ts:86`).
- Updated the preview pane UI to stack rows per metric, constrain preview widths, and align group cards side by side with horizontal scrolling (`interface-final/web/src/components/PreviewPane.tsx:9`).

## 2025-11-04 – UI workflow improvements
- Preview API now serves all channel metrics at once with metric identifiers and consolidated caching (`interface-final/api/services/studies.py:554`).
- Live preview grid renders compact rows with shared raw/mask/overlay labels, subject details moved to tooltips, and all channel/ratio views visible simultaneously (`interface-final/web/src/components/PreviewPane.tsx:1`).
- Added file upload endpoints and UI so configs and threshold results can be selected via system dialogs instead of manual paths (`interface-final/api/main.py:74`, `interface-final/web/src/components/LeftPanel.tsx:1`).
- Introduced a collapsible sidebar toggle to reclaim horizontal space when adjusting previews (`interface-final/web/src/App.tsx:1`).
