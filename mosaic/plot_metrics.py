# SPDX-License-Identifier: LGPL-3.0-or-later
"""Cross-validation metric distribution plots for MOSAIC.

This module provides :func:`plot_cv_distributions`, which generates a
three-panel figure showing the distribution of F1, Accuracy, and AUC-ROC
scores across cross-validation folds. Each panel combines a box plot with
jittered scatter points for individual fold scores.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Iterable, Optional

__all__ = ["plot_cv_distributions"]


def _scatter_with_jitter(ax, data, color="black", s=20, jitter_scale=0.04):
    np.random.seed(42)
    x_vals = 1 + np.random.normal(0, jitter_scale, len(data))
    ax.scatter(x_vals, data, color=color, s=s, zorder=3)


def plot_cv_distributions(
    f1: Iterable[float],
    acc: Iterable[float],
    auc: Optional[Iterable[float]],
    model_name: str,
    params: str,
    save_to: Optional[Path] = None,
) -> None:
    """Generate and optionally save a three-panel CV metric distribution plot.

    Creates a figure with one panel per metric (F1, Accuracy, AUC-ROC). Each
    panel shows a box plot overlaid with jittered scatter points representing
    individual cross-validation fold scores. If *auc* is ``None``, the
    AUC-ROC panel is hidden.

    Parameters
    ----------
    f1 : iterable of float
        F1-macro scores from each cross-validation fold.
    acc : iterable of float
        Accuracy scores from each cross-validation fold.
    auc : iterable of float or None
        AUC-ROC scores from each cross-validation fold. Pass ``None`` to hide
        the AUC-ROC panel (e.g. for binary classifiers without probability
        support).
    model_name : str
        Model name shown in the figure title (e.g. ``"SVC"``).
    params : str
        Serialised hyperparameter string shown in the figure subtitle.
    save_to : pathlib.Path or None, optional
        Destination path for the PNG file. Parent directories are created
        automatically if they do not exist. If ``None``, the figure is
        displayed interactively via :func:`matplotlib.pyplot.show`. Default
        is ``None``.

    Notes
    -----
    When *save_to* is provided, the figure is saved at 300 DPI with
    ``bbox_inches="tight"`` and the directory tree is created automatically.
    The function does not close the figure after saving; callers are
    responsible for calling :func:`matplotlib.pyplot.close` if needed.

    Examples
    --------
    Save a plot for an SVC model to a nested directory:

    >>> from pathlib import Path
    >>> from mosaic import plot_cv_distributions
    >>> f1  = [0.76, 0.90, 0.82]
    >>> acc = [0.77, 0.90, 0.82]
    >>> auc = [0.83, 0.95, 0.89]
    >>> plot_cv_distributions(
    ...     f1, acc, auc,
    ...     model_name="SVC",
    ...     params="{'kernel': 'linear', 'C': 1}",
    ...     save_to=Path("output/figures/SVC_PCA70.png"),
    ... )
    """
    metrics = [("F1 Score", f1), ("Accuracy", acc), ("AUC-ROC", auc)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)

    for ax, (title, data) in zip(axes, metrics):
        if data is None:
            ax.set_visible(False)
            continue

        ax.boxplot(
            data,
            patch_artist=True,
            boxprops={"facecolor": "#5DA5DA", "alpha": 0.6},
        )
        _scatter_with_jitter(ax, data)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_ylabel(title)

    fig.suptitle(
        f"CV Metrics Distribution - {model_name}\nHyperparameters: {params}",
        fontsize=13,
    )
    fig.tight_layout()

    if save_to:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=300, bbox_inches="tight")
    else:
        plt.show()
