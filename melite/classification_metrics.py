# SPDX-License-Identifier: LGPL-3.0-or-later
"""Classification target identity and evaluation-metric contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import column_or_1d

__all__ = [
    "ClassificationTargetInfo",
    "analyze_classification_target",
    "classification_metric_policy",
    "classification_scoring",
]


@dataclass(frozen=True)
class ClassificationTargetInfo:
    """Deterministically ordered class identities and observed counts."""

    classes: tuple[Any, ...]
    counts: tuple[int, ...]

    @property
    def is_multiclass(self) -> bool:
        """Whether the target contains three or more mutually exclusive classes."""
        return len(self.classes) >= 3


def analyze_classification_target(y) -> ClassificationTargetInfo:
    """Validate a classification target without changing its class identities."""
    y_array = column_or_1d(y, warn=True)
    check_classification_targets(y_array)
    encoder = LabelEncoder().fit(y_array)
    if len(encoder.classes_) < 2:
        raise ValueError(
            "Classification requires at least two distinct classes; "
            f"observed {len(encoder.classes_)}."
        )
    encoded = encoder.transform(y_array)
    counts = np.bincount(encoded, minlength=len(encoder.classes_))
    return ClassificationTargetInfo(
        classes=tuple(encoder.classes_),
        counts=tuple(int(count) for count in counts),
    )


def classification_scoring(target: ClassificationTargetInfo) -> dict[str, Any]:
    """Return MELITE's binary or multiclass outer-evaluation scorers."""
    auc_scorer: Any = "roc_auc"
    if target.is_multiclass:
        auc_scorer = make_scorer(
            roc_auc_score,
            response_method="predict_proba",
            multi_class="ovo",
            average="macro",
            labels=list(target.classes),
        )
    return {"f1": "f1_macro", "acc": "accuracy", "auc": auc_scorer}


def classification_metric_policy() -> dict[str, Any]:
    """Return the explicit classification metric policy for provenance."""
    return {
        "optimization_and_selection": {
            "metric": "f1_macro",
            "direction": "maximize",
        },
        "reported": {
            "f1_macro": {"average": "macro"},
            "accuracy": {},
            "roc_auc": {
                "binary": {
                    "scorer": "roc_auc",
                },
                "multiclass": {
                    "strategy": "ovo",
                    "average": "macro",
                    "response_method": "predict_proba",
                },
            },
        },
        "per_class": {
            "metrics": ["precision", "recall", "f1_score", "support"],
            "source": "held_out_outer_fold_predictions",
            "class_order": "original_target_label_order",
            "undefined_metric_convention": {
                "zero_division": 0,
                "description": (
                    "Undefined precision, recall, and F1-score values are "
                    "reported as 0."
                ),
            },
            "summary_aggregation": {
                "unit": "outer_fold",
                "mean": "arithmetic",
                "standard_deviation": {"ddof": 0},
                "support": "mean_and_standard_deviation_across_outer_folds",
            },
        },
    }
