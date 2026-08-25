# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for melite.plot_metrics."""

import doctest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pytest

import melite.plot_metrics as plot_metrics_module
from melite.plot_metrics import plot_f1_macro_evidence


CLASSIFIER_SCORES = {
    "SVC": [0.76, 0.82, 0.90],
    "RandomForestClassifier": [0.78, 0.84, 0.87],
    "XGBClassifier": [0.80, 0.86, 0.89],
}


def test_returns_figure_without_showing(monkeypatch):
    def fail_show():
        raise AssertionError("plot_f1_macro_evidence must not call plt.show()")

    monkeypatch.setattr(plt, "show", fail_show)

    fig = plot_f1_macro_evidence(
        CLASSIFIER_SCORES,
        selected_classifier="XGBClassifier",
        dataset_id="example",
    )

    assert isinstance(fig, Figure)
    plt.close(fig)


def test_saves_png_and_creates_nested_directories(tmp_path):
    save_to = tmp_path / "a" / "b" / "evaluation_f1_macro_example.png"

    fig = plot_f1_macro_evidence(
        CLASSIFIER_SCORES,
        selected_classifier="XGBClassifier",
        dataset_id="example",
        save_to=save_to,
    )

    assert save_to.exists()
    assert save_to.stat().st_size > 0
    plt.close(fig)


def test_shows_all_classifiers_and_f1_macro_axis():
    fig = plot_f1_macro_evidence(
        CLASSIFIER_SCORES,
        selected_classifier="XGBClassifier",
        dataset_id="example",
    )

    ax = fig.axes[0]
    labels = [tick.get_text() for tick in ax.get_xticklabels()]

    assert labels == list(CLASSIFIER_SCORES)
    assert ax.get_ylabel() == "Outer-CV F1-macro"
    assert ax.get_ylim() == pytest.approx((0.0, 1.0))
    plt.close(fig)


def test_marks_selected_classifier():
    fig = plot_f1_macro_evidence(
        CLASSIFIER_SCORES,
        selected_classifier="XGBClassifier",
        dataset_id="example",
    )

    texts = [text.get_text() for text in fig.axes[0].texts]

    assert "Selected" in texts
    plt.close(fig)


def test_identifies_dataset_in_title():
    fig = plot_f1_macro_evidence(
        CLASSIFIER_SCORES,
        selected_classifier="XGBClassifier",
        dataset_id="morgan_r2_2048",
    )

    assert fig.axes[0].get_title() == "Classifier evaluation — morgan_r2_2048"
    plt.close(fig)


def test_smoke_mode_is_explicitly_marked():
    fig = plot_f1_macro_evidence(
        CLASSIFIER_SCORES,
        selected_classifier="XGBClassifier",
        dataset_id="example",
        smoke=True,
    )

    figure_text = " ".join(text.get_text() for text in fig.texts)

    assert "SMOKE MODE" in figure_text
    assert "not for final classifier selection" in figure_text
    plt.close(fig)


def test_rejects_unknown_selected_classifier():
    with pytest.raises(ValueError, match="selected_classifier"):
        plot_f1_macro_evidence(
            CLASSIFIER_SCORES,
            selected_classifier="StackingClassifier",
            dataset_id="example",
        )


def test_rejects_empty_classifier_scores():
    with pytest.raises(ValueError, match="at least one classifier"):
        plot_f1_macro_evidence(
            {},
            selected_classifier="SVC",
            dataset_id="example",
        )


def test_rejects_empty_score_sequence_with_classifier_name():
    with pytest.raises(ValueError, match="SVC.*must not be empty"):
        plot_f1_macro_evidence(
            {"SVC": []},
            selected_classifier="SVC",
            dataset_id="example",
        )


def test_rejects_non_1d_scores_with_classifier_name_and_shape():
    with pytest.raises(
        ValueError,
        match=r"SVC.*one-dimensional.*shape \(2, 2\)",
    ):
        plot_f1_macro_evidence(
            {"SVC": [[0.70, 0.71], [0.72, 0.73]]},
            selected_classifier="SVC",
            dataset_id="example",
        )


def test_wraps_score_conversion_error_with_classifier_name():
    with pytest.raises(ValueError, match="BrokenClassifier.*converted"):
        plot_f1_macro_evidence(
            {
                "SVC": [0.80],
                "BrokenClassifier": ["not-a-score"],
            },
            selected_classifier="SVC",
            dataset_id="example",
        )


@pytest.mark.parametrize(
    "invalid_score",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
    ],
)
def test_rejects_non_finite_scores_before_range_validation(invalid_score):
    with pytest.raises(ValueError, match="SVC.*finite") as exc_info:
        plot_f1_macro_evidence(
            {"SVC": [invalid_score]},
            selected_classifier="SVC",
            dataset_id="example",
        )

    assert "within [0, 1]" not in str(exc_info.value)


@pytest.mark.parametrize(
    "invalid_score",
    [pytest.param(-0.01, id="below-zero"), pytest.param(1.01, id="above-one")],
)
def test_rejects_scores_outside_f1_macro_range(invalid_score):
    with pytest.raises(ValueError, match=r"SVC.*within \[0, 1\]"):
        plot_f1_macro_evidence(
            {"SVC": [invalid_score]},
            selected_classifier="SVC",
            dataset_id="example",
        )


def test_one_score_sequence_is_valid_with_zero_population_sd():
    fig = plot_f1_macro_evidence(
        {"SVC": [0.73]},
        selected_classifier="SVC",
        dataset_id="example",
    )

    errorbar = fig.axes[0].containers[0]
    mean_line, _, bar_collections = errorbar.lines
    segment = bar_collections[0].get_segments()[0]

    assert mean_line.get_ydata() == pytest.approx([0.73])
    assert segment[:, 1] == pytest.approx([0.73, 0.73])
    plt.close(fig)


def test_shows_mean_and_population_standard_deviation():
    fig = plot_f1_macro_evidence(
        {"SVC": [0.20, 0.80]},
        selected_classifier="SVC",
        dataset_id="example",
    )

    errorbar = fig.axes[0].containers[0]
    mean_line, _, bar_collections = errorbar.lines
    segment = bar_collections[0].get_segments()[0]

    assert mean_line.get_ydata() == pytest.approx([0.50])
    assert segment[:, 1] == pytest.approx([0.20, 0.80])
    plt.close(fig)


def test_does_not_recompute_selection():
    scores = {
        "SVC": [0.90, 0.91],
        "RandomForestClassifier": [0.70, 0.71],
    }

    fig = plot_f1_macro_evidence(
        scores,
        selected_classifier="RandomForestClassifier",
        dataset_id="example",
    )

    ax = fig.axes[0]
    selected_text = next(text for text in ax.texts if text.get_text() == "Selected")
    markers = [container.lines[0].get_marker() for container in ax.containers]

    assert selected_text.get_position()[0] == 1
    assert markers == ["o", "*"]
    plt.close(fig)


def test_accepts_str_save_path_and_infers_format(tmp_path):
    save_to = tmp_path / "nested" / "evaluation_f1_macro_example.svg"

    fig = plot_f1_macro_evidence(
        CLASSIFIER_SCORES,
        selected_classifier="XGBClassifier",
        dataset_id="example",
        save_to=str(save_to),
    )

    assert save_to.exists()
    assert save_to.stat().st_size > 0
    plt.close(fig)


def test_plot_metrics_module_doctest_executes_public_example():
    result = doctest.testmod(plot_metrics_module)

    assert result.attempted > 0
    assert result.failed == 0
