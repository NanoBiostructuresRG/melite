# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for mosaic.plot_metrics."""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for tests

import pytest
from pathlib import Path
from mosaic.plot_metrics import plot_cv_distributions

F1  = [0.76, 0.90, 0.82]
ACC = [0.77, 0.90, 0.82]
AUC = [0.83, 0.95, 0.89]


def test_saves_png_to_existing_directory(tmp_path):
    save_to = tmp_path / "figures" / "test_plot.png"
    save_to.parent.mkdir(parents=True)
    plot_cv_distributions(F1, ACC, AUC, "SVC", "{}", save_to=save_to)
    assert save_to.exists()


def test_creates_nested_directories_automatically(tmp_path):
    save_to = tmp_path / "a" / "b" / "c" / "test_plot.png"
    assert not save_to.parent.exists()
    plot_cv_distributions(F1, ACC, AUC, "SVC", "{}", save_to=save_to)
    assert save_to.exists()


def test_saved_png_is_non_empty(tmp_path):
    save_to = tmp_path / "test_plot.png"
    plot_cv_distributions(F1, ACC, AUC, "SVC", "{}", save_to=save_to)
    assert save_to.stat().st_size > 0


def test_save_to_none_does_not_raise():
    """When save_to=None the function shows the plot — should not raise."""
    import matplotlib.pyplot as plt
    plot_cv_distributions(F1, ACC, AUC, "SVC", "{}", save_to=None)
    plt.close("all")


def test_auc_none_does_not_raise(tmp_path):
    """AUC panel is hidden when auc=None."""
    save_to = tmp_path / "test_no_auc.png"
    plot_cv_distributions(F1, ACC, None, "SVC", "{}", save_to=save_to)
    assert save_to.exists()
