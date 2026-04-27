"""Regression tests for preview scale-bar rendering."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from data_models import VisualizationConfig  # noqa: E402
from visualization import ND2Visualizer  # noqa: E402


def _text_bounds_in_data_coords(ax, text):
    figure = ax.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    bbox = text.get_window_extent(renderer=renderer)
    (x0, y0), (x1, y1) = ax.transData.inverted().transform([[bbox.x0, bbox.y0], [bbox.x1, bbox.y1]])
    return x0, y0, x1, y1


def test_scale_bar_label_stays_inside_panel_bounds() -> None:
    visualizer = ND2Visualizer(
        VisualizationConfig(scale_bar_um=150, scale_bar_font_size=24),
        pixel_size_um=1.0,
    )
    fig, ax = plt.subplots(figsize=(2, 2), dpi=100)
    ax.imshow(np.zeros((80, 80)))
    ax.axis("off")

    visualizer.add_scale_bar(ax, (80, 80))

    assert ax.texts
    label = ax.texts[-1]
    x0, _, x1, _ = _text_bounds_in_data_coords(ax, label)

    assert label.get_ha() == "right"
    assert x0 >= -1
    assert x1 <= 81
    plt.close(fig)


def test_scale_bar_font_size_is_clamped_for_narrow_exports() -> None:
    requested_font_size = 32
    visualizer = ND2Visualizer(
        VisualizationConfig(scale_bar_um=200, scale_bar_font_size=requested_font_size),
        pixel_size_um=1.0,
    )
    fig, ax = plt.subplots(figsize=(1.8, 1.8), dpi=100)
    ax.imshow(np.zeros((72, 72)))
    ax.axis("off")

    visualizer.add_scale_bar(ax, (72, 72))

    assert ax.texts
    label = ax.texts[-1]
    assert 6 <= label.get_fontsize() < requested_font_size
    plt.close(fig)
