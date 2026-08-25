# SPDX-License-Identifier: LGPL-3.0-or-later
"""Visualize preserved outer-CV evidence for MELITE.

This module plots supplied F1-macro evidence and marks the classifier
previously selected by the evaluation workflow.
"""

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

__all__ = ["plot_f1_macro_evidence"]


def plot_f1_macro_evidence(
    classifier_scores: Mapping[str, Sequence[float]],
    selected_classifier: str,
    dataset_id: str,
    save_to: Path | str | None = None,
    smoke: bool = False,
) -> Figure:
    """Visualize preserved outer-CV F1-macro evidence.

    The supplied scores are plotted directly, and the classifier previously
    selected by the evaluation workflow is explicitly marked.

    Parameters
    ----------
    classifier_scores : Mapping[str, Sequence[float]]
        Mapping from classifier name to supplied outer-CV F1-macro scores.
        Each score sequence must be numeric, one-dimensional, non-empty,
        finite, and within ``[0, 1]``.
    selected_classifier : str
        Classifier previously selected by the evaluation workflow. This
        function marks that classifier but does not recompute selection.
    dataset_id : str
        User-defined dataset identifier shown in the figure title.
    save_to : pathlib.Path or str or None, optional
        Optional destination path. Parent directories are created
        automatically, and Matplotlib infers the output format from the path
        or extension. If ``None``, the figure is not saved.
    smoke : bool, optional
        Whether to annotate the figure as smoke-mode evidence. This affects
        presentation only.

    Returns
    -------
    matplotlib.figure.Figure
        Newly created figure. The function does not show or close it; the
        caller owns the returned figure and is responsible for displaying or
        closing it when appropriate.

    Raises
    ------
    ValueError
        If ``classifier_scores`` is empty; if ``selected_classifier`` is not
        present; or if any classifier's scores cannot be converted to floats,
        are not one-dimensional, are empty, contain non-finite values, or fall
        outside ``[0, 1]``.

    Notes
    -----
    This function performs no fitting, hyperparameter tuning, cross-validation,
    or classifier selection. Supplied scores are visualized directly. The mean
    and population standard deviation (``ddof=0``) are shown. Classifiers
    follow the iteration order of the supplied mapping. Horizontal jitter uses
    a deterministic local random-number generator and affects display position
    only; supplied F1-macro values are not modified.

    Saved raster output uses 300 dpi; vector formats follow Matplotlib's
    behavior. Filesystem and Matplotlib saving errors propagate to the caller.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from melite import plot_f1_macro_evidence
    >>> scores = {
    ...     "SVC": [0.78, 0.82, 0.80],
    ...     "RandomForestClassifier": [0.81, 0.84, 0.83],
    ... }
    >>> fig = plot_f1_macro_evidence(
    ...     scores,
    ...     selected_classifier="RandomForestClassifier",
    ...     dataset_id="sample_tabular",
    ... )
    >>> fig.axes[0].get_ylabel()
    'Outer-CV F1-macro'
    >>> plt.close(fig)
    """
    if not classifier_scores:
        raise ValueError("classifier_scores must contain at least one classifier.")

    if selected_classifier not in classifier_scores:
        raise ValueError(
            f"selected_classifier '{selected_classifier}' is not present in "
            "classifier_scores."
        )

    classifiers = list(classifier_scores)
    scores_by_classifier = []
    for classifier in classifiers:
        try:
            scores = np.asarray(classifier_scores[classifier], dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Scores for classifier '{classifier}' could not be converted "
                "to a float array."
            ) from exc
        if scores.ndim != 1:
            raise ValueError(
                f"Scores for classifier '{classifier}' must be one-dimensional; "
                f"got shape {scores.shape}."
            )
        if scores.size == 0:
            raise ValueError(f"Scores for classifier '{classifier}' must not be empty.")
        if not np.all(np.isfinite(scores)):
            raise ValueError(
                f"Scores for classifier '{classifier}' must contain only finite values."
            )
        if np.any((scores < 0.0) | (scores > 1.0)):
            raise ValueError(
                f"Scores for classifier '{classifier}' must be within [0, 1]."
            )
        scores_by_classifier.append(scores)

    fig, ax = plt.subplots(figsize=(8, 5))
    rng = np.random.default_rng(42)

    for index, (classifier, scores) in enumerate(
        zip(classifiers, scores_by_classifier)
    ):
        jitter = rng.normal(0.0, 0.04, size=len(scores))
        ax.scatter(
            index + jitter,
            scores,
            s=28,
            alpha=0.75,
            zorder=2,
        )

        mean = float(np.mean(scores))
        std = float(np.std(scores))

        marker = "*" if classifier == selected_classifier else "o"
        markersize = 11 if classifier == selected_classifier else 7

        ax.errorbar(
            index,
            mean,
            yerr=std,
            fmt=marker,
            markersize=markersize,
            capsize=5,
            linewidth=1.5,
            zorder=3,
        )

        if classifier == selected_classifier:
            ax.text(
                index,
                0.03,
                "Selected",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    ax.set_xticks(range(len(classifiers)))
    ax.set_xticklabels(classifiers)
    ax.set_ylabel("Outer-CV F1-macro")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Classifier evaluation — {dataset_id}")
    ax.grid(axis="y", alpha=0.2)

    if smoke:
        fig.text(
            0.5,
            0.01,
            "SMOKE MODE — not for final classifier selection",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    fig.tight_layout(rect=(0, 0.04 if smoke else 0, 1, 1))

    if save_to is not None:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=300, bbox_inches="tight")

    return fig
