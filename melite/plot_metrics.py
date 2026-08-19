# SPDX-License-Identifier: LGPL-3.0-or-later
"""Outer cross-validation evidence plots for MELITE.

This module provides :func:`plot_f1_macro_evidence`, which visualizes the
outer-CV F1-macro scores for every evaluated model family in one dataset and
identifies the family selected by mean outer-CV F1-macro.
"""

from pathlib import Path
from typing import Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

__all__ = ["plot_f1_macro_evidence"]


def plot_f1_macro_evidence(
    family_scores: Mapping[str, Sequence[float]],
    selected_family: str,
    dataset_id: str,
    save_to: Optional[Path] = None,
    smoke: bool = False,
) -> Figure:
    """Plot outer-CV F1-macro evidence for all evaluated model families.

    Each model family is represented by its individual outer-CV F1-macro
    scores together with the mean and standard deviation. The selected family
    is explicitly marked.

    Parameters
    ----------
    family_scores : mapping of str to sequence of float
        Outer-CV F1-macro scores keyed by model-family name.
    selected_family : str
        Family selected by mean outer-CV F1-macro.
    dataset_id : str
        Dataset identifier shown in the figure title.
    save_to : pathlib.Path or None, optional
        Destination path for the PNG file. Parent directories are created
        automatically. If ``None``, the figure is returned without being
        displayed or saved.
    smoke : bool, optional
        Whether the evidence comes from a smoke-mode run.

    Returns
    -------
    matplotlib.figure.Figure
        Generated figure.

    Raises
    ------
    ValueError
        If *family_scores* is empty or *selected_family* is not present.
    """
    if not family_scores:
        raise ValueError("family_scores must contain at least one model family.")

    if selected_family not in family_scores:
        raise ValueError(
            f"selected_family '{selected_family}' is not present in family_scores."
        )

    families = list(family_scores)
    scores_by_family = [
        np.asarray(family_scores[family], dtype=float)
        for family in families
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    rng = np.random.default_rng(42)

    for index, (family, scores) in enumerate(zip(families, scores_by_family)):
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

        marker = "*" if family == selected_family else "o"
        markersize = 11 if family == selected_family else 7

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

        if family == selected_family:
            ax.text(
                index,
                0.03,
                "Selected",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(families)
    ax.set_ylabel("Outer-CV F1-macro")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Model-family evaluation — {dataset_id}")
    ax.grid(axis="y", alpha=0.2)

    if smoke:
        fig.text(
            0.5,
            0.01,
            "SMOKE MODE — not for final model selection",
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
