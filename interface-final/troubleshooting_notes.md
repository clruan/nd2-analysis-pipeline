# Preview & Statistics Troubleshooting Notes

This log captures the issues uncovered during the most recent debugging session so future optimizations can avoid repeating them.

---

## 1. Study Loading
- **Missing `image_data` in results JSON**  
  Loading a partial or intermediate JSON triggered `Threshold results file is missing required field 'image_data'`. Always use the full `threshold_results_*.json` produced by `process_directory_all_thresholds` (e.g., via `threshold_analysis/run_batch.py` or the Interface-Final runner).

- **Wrong file selected**  
  Selecting the config JSON instead of the results file yields the same error. Verify the filename pattern and double-check the `.meta.json` sidecar to confirm the path.

## 2. FastAPI Boot Failures
- **SyntaxError in `load_study`**  
  Passing keyword arguments inside a tuple when raising `HTTPException` broke Uvicorn (`Maybe you meant '==' or ':='`). Never wrap keyword arguments in parentheses; construct the `HTTPException` directly.

- **NameError for `input_dir`**  
  After refactoring previews we referenced `input_dir` without defining it in `generate_previews`. Keep state on the `StudyRecord` (`record.input_dir`) and pass that through.

## 3. Preview Generation Pitfalls
- **ND2 volume unavailable**  
  When `/Volumes/...` was offline the preview generator silently failed. Fixes implemented:
  - Recover ND2 root from the `.meta.json` sidecar or nearby folders.
  - Reuse cached PNGs before touching ND2 files.
  - Generate placeholder PNGs (highlighted in the UI) if the ND2 files are unreachable.
  - Important: mount/copy the ND2 directory at least once to populate the cache with real previews.

- **No root `index.html` for Vite**  
  The app returned a 404 until `interface-final/web/index.html` was copied from `public/index.html`. Ensure the root HTML file exists.

- **Proxy returned 404 for `/config/scan`**  
  Vite forwarded `/api/config/scan` to FastAPI without rewriting; the API routes lacked the `/api` prefix. Added `rewrite: path => path.replace(/^\/api/, "")` to the dev server config.

## 4. Statistical Analysis
- **Pydantic float parsing errors**  
  Returning dicts with string keys & values forced Pydantic to coerce strings into floats, producing dozens of `float_parsing` errors. Introduced typed records (`MouseAverageRecord`, `IndividualImageRecord`) and explicit casts.

- **Small-sample warnings (`SmallSampleWarning`)**  
  SciPy emitted warnings when groups contained one sample. The new implementation filters finite values, skips comparisons lacking sufficient samples, and returns a descriptive `note` explaining why results are missing.

- **Static test settings**  
  The UI previously hardcoded `test_type="auto"` and `comparison_mode="all_vs_one"`. Users can now select parametric vs non-parametric tests, reference vs pairwise comparisons, and significance display (`stars` vs exact p-values).

- **TypeScript build failures**  
  Switched React Query mutation state checks (`isLoading`) to the v5 API (`isPending`) and added missing TypeScript types for responses and the Plotly module.

## 5. Output Paths & Tooling
- **Hard to regenerate canonical results**  
  Added `threshold_analysis/run_batch.py` to wrap the classic pipeline and guarantee a full results JSON.

- **`.meta.json` not consulted**  
  Earlier versions ignored the metadata file storing the ND2 input path. The loader now uses it to rebuild replicate lookups after restarts.

## 7. React Hook Order Regression
- **White screen after loading a study**  
  The analysis dashboard returned early before calling several hooks. After a study loaded, those hooks executed, breaking React’s hook order and triggering a blank screen with “Rendered more hooks than during the previous render.” Reordered the hook declarations in `AnalysisBoard.tsx` so they run on every render regardless of data readiness.

## 8. Browser Autofill Extensions
- **`content_script.js` cannot read property 'control'**  
  Chrome autofill/completion extensions inject their own scripts and can throw in the console when they inspect MUI inputs. These messages do not come from the app and can be ignored or silenced by disabling the offending extension while testing.

## 6. Miscellaneous
- **`preview_restructure_avoid.md` replaced**  
  Consolidated all encountered errors into this single document (`troubleshooting_notes.md`) for easier reference in future restructuring efforts.

## 9. Latest Session Notes
- **Histogram navigation mismatch**  
  Attempted to solve long x-axis labels by wrapping charts in a horizontal scroller. Users preferred Plotly’s native zoom/pan so the change was reverted and replaced with scroll-wheel zoom plus pan/zoom buttons.
- **Subject ID heuristics**  
  `guess_subject_id` returned the first alphanumeric token in a filename (e.g., `A1_A17_rep1` → `A1`), causing downstream grouping errors where `A17` appeared under `A1`. The helper now ranks all matches by digit length and position, picking the most specific token.

---

### Quick Checklist Before Future Work
1. Use a full `threshold_results_*.json` with `image_data`.
2. Make sure the ND2 directory is mounted or copied locally before loading a study.
3. Verify cached PNGs exist (or allow placeholders) before relying on previews offline.
4. Keep FastAPI exception syntax simple—no tuples containing keyword args.
5. Test statistical modes with small sample counts; ensure the UI communicates when results are skipped.
6. Re-run `npm --prefix interface-final/web run build` after TypeScript changes to catch typing regressions.

Keeping these constraints in mind will make future preview and analysis optimization loops much smoother.
