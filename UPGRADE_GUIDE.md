# ND2 Pipeline Future Upgrade Guide

**Date:** June 17, 2025  
**Role:** Senior Data Scientist  
**Environment:** Windows + PowerShell venv

---

## 1. Local Development Setup

1. Open PowerShell and navigate to project root:
   ```powershell
   cd "c:\Users\Duke\Documents\ND2_Analysis_Pipeline"
   ```
2. Activate venv:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## 2. Highlighting Representative Images in Excel

- `excel_output.py` now includes a `highlight_fill` style using the `highlight_color` in `config.EXCEL_SETTINGS`.
- The raw data sheet (`Raw_Data`) receives an extra parameter, `representative_images`, which is a `Dict[str, List[str]]` mapping each group to its top filenames.
- Rows where the `Filename` matches a representative image are filled in the highlight color (yellow by default).

Usage example in Python:
```python
from excel_output import ExcelReporter
from data_models import ProcessingResults

# Assuming you have a ProcessingResults instance:
reporter = ExcelReporter()
reporter.create_simple_report(results, 'analysis_results.xlsx')
```

## 3. Simplified Project Structure

```
ND2_Analysis_Pipeline/
├── config.py                # Thresholds, color maps, Excel settings
├── data_models.py           # ProcessingResults, VisualizationConfig, type defs
├── image_processing.py      # Core image metrics and mouse-level aggregation
├── excel_output.py          # Simplified Excel reporting + highlights
├── visualization.py         # Image plotting and comparison visualizations
├── processing_pipeline.py   # Workflow orchestration, I/O, JSON/Excel outputs
├── main.py                  # CLI entry point for full pipeline
├── visualize.py             # Standalone visualization CLI tool
├── example_usage.ipynb      # Jupyter tutorial example
├── requirements.txt         # pip dependencies
├── sample_config.json       # Example group config template
├── UPGRADE_GUIDE.md         # (This file) future upgrades and tasks
├── README.md                # User guide and quick start
└── venv/                    # Local virtual environment
```

## 4. Future Enhancement Roadmap

- **Packaging**: Convert to pip package (`setup.py` or `pyproject.toml`) for easier installs.
- **CI/CD**: Add GitHub Actions for linting, testing (pytest), and packaging.
- **Testing**: Create `tests/` directory with unit tests for each module; use `pytest`.
- **Reports**: Extend `excel_output.py` for charts and statistical tests (ANOVA, p-values).
- **Formats**: Add HTML/PDF report generation via Jinja2 or ReportLab.
- **Plugins**: Design plugin architecture for custom analyses and outputs.
- **Multi-format**: Support other microscopy formats via Bio-Formats or AICSImageIO.
- **Dashboard**: Integrate with Dash or Streamlit for interactive dashboards.
- **GUI**: Develop PyQt or Tkinter GUI to wrap CLI options.
- **Documentation**: Maintain `CHANGELOG.md` and enforce semantic versioning.

---

**Maintainer:** Senior Data Science Team
