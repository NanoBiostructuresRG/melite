# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for melite.plot_metrics."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pytest

from melite.plot_metrics import plot_f1_macro_evidence


FAMILY_SCORES = {
    "SVC": [0.76, 0.82, 0.90],
    "RandomForestClassifier": [0.78, 0.84, 0.87],
    "XGBClassifier": [0.80, 0.86, 0.89],
}


def test_returns_figure_without_showing(monkeypatch):
    def fail_show():
        raise AssertionError("plot_f1_macro_evidence must not call plt.show()")

    monkeypatch.setattr(plt, "show", fail_show)

    fig = plot_f1_macro_evidence(
        FAMILY_SCORES,
        selected_family="XGBClassifier",
        dataset_id="example",
    )

    assert isinstance(fig, Figure)
    plt.close(fig)


def test_saves_png_and_creates_nested_directories(tmp_path):
    save_to = tmp_path / "a" / "b" / "evaluation_f1_macro_example.png"

    fig = plot_f1_macro_evidence(
        FAMILY_SCORES,
        selected_family="XGBClassifier",
        dataset_id="example",
        save_to=save_to,
    )

    assert save_to.exists()
    assert save_to.stat().st_size > 0
    plt.close(fig)


def test_shows_all_model_families_and_f1_macro_axis():
    fig = plot_f1_macro_evidence(
        FAMILY_SCORES,
        selected_family="XGBClassifier",
        dataset_id="example",
    )

    ax = fig.axes[0]
    labels = [tick.get_text() for tick in ax.get_xticklabels()]

    assert labels == list(FAMILY_SCORES)
    assert ax.get_ylabel() == "Outer-CV F1-macro"
    assert ax.get_ylim() == pytest.approx((0.0, 1.0))
    plt.close(fig)


def test_marks_selected_family():
    fig = plot_f1_macro_evidence(
        FAMILY_SCORES,
        selected_family="XGBClassifier",
        dataset_id="example",
    )

    texts = [text.get_text() for text in fig.axes[0].texts]

    assert "Selected" in texts
    plt.close(fig)


def test_identifies_dataset_in_title():
    fig = plot_f1_macro_evidence(
        FAMILY_SCORES,
        selected_family="XGBClassifier",
        dataset_id="morgan_r2_2048",
    )

    assert "morgan_r2_2048" in fig.axes[0].get_title()
    plt.close(fig)


def test_smoke_mode_is_explicitly_marked():
    fig = plot_f1_macro_evidence(
        FAMILY_SCORES,
        selected_family="XGBClassifier",
        dataset_id="example",
        smoke=True,
    )

    figure_text = " ".join(text.get_text() for text in fig.texts)

    assert "SMOKE MODE" in figure_text
    assert "not for final model selection" in figure_text
    plt.close(fig)


def test_rejects_unknown_selected_family():
    with pytest.raises(ValueError, match="selected_family"):
        plot_f1_macro_evidence(
            FAMILY_SCORES,
            selected_family="StackingClassifier",
            dataset_id="example",
        )
