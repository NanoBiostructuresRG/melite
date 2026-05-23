# SPDX-License-Identifier: LGPL-3.0-or-later
__all__ = ["plot_cv_distributions"]
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Iterable, Optional


def _scatter_with_jitter(ax, data, color="black", s=20, jitter_scale=0.04):
    np.random.seed(42)                 # reproducibilidad
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

    metrics = [("F1 Score", f1), ("Accuracy", acc), ("AUC-ROC", auc)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)

    for ax, (title, data) in zip(axes, metrics):
        if data is None:      
            ax.set_visible(False)
            continue

        #ax.boxplot(data, patch_artist=True)
        #ax.scatter([1] * len(data), data, s=18, color="black", zorder=3)
        
        bp = ax.boxplot(
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