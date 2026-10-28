# SPDX-License-Identifier: LGPL-3.0-or-later
"""Internal probability calibration for classifiers without native probabilities."""

from __future__ import annotations

from numbers import Integral

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

__all__ = ["ProbabilityCalibratedClassifier"]


def _estimator_has(attribute: str):
    def check(self) -> bool:
        estimator = getattr(self, "predictor_", self.estimator)
        return hasattr(estimator, attribute)

    return check


class ProbabilityCalibratedClassifier(ClassifierMixin, BaseEstimator):
    """Add calibrated probabilities without changing direct class predictions.

    The wrapped estimator is fitted twice: one clone supplies predictions and
    decision values, while an independent calibrated clone supplies class
    probabilities.

    Parameters
    ----------
    estimator : estimator
        Base classifier to fit directly and calibrate independently.
    max_calibration_cv : int, optional
        Maximum number of calibration folds. The effective fold count adapts
        to the smallest class count. Default is ``5``.
    """

    def __init__(self, estimator, max_calibration_cv=5):
        self.estimator = estimator
        self.max_calibration_cv = max_calibration_cv

    def fit(self, X, y):
        """Fit independent direct-prediction and probability-calibration paths."""
        if (
            isinstance(self.max_calibration_cv, bool)
            or not isinstance(self.max_calibration_cv, Integral)
            or self.max_calibration_cv < 2
        ):
            raise ValueError("max_calibration_cv must be an integer of at least 2.")

        X_valid, y_valid = validate_data(
            self,
            X,
            y,
            reset=True,
        )
        check_classification_targets(y_valid)
        unique_classes, class_counts = np.unique(y_valid, return_counts=True)
        if len(unique_classes) < 2:
            raise ValueError(
                "Probability calibration cannot fit data with one class; "
                "at least two classes are required."
            )
        minimum_class_count = int(class_counts.min()) if class_counts.size else 0
        if minimum_class_count < 2:
            raise ValueError(
                "Probability calibration requires at least two samples in every class."
            )

        self.calibration_cv_ = min(
            int(self.max_calibration_cv),
            minimum_class_count,
        )
        self.predictor_ = clone(self.estimator).fit(X_valid, y_valid)
        self.calibrator_ = CalibratedClassifierCV(
            estimator=clone(self.estimator),
            ensemble=False,
            cv=self.calibration_cv_,
        ).fit(X_valid, y_valid)

        if not np.array_equal(self.predictor_.classes_, self.calibrator_.classes_):
            raise RuntimeError(
                "Direct predictor and probability calibrator learned "
                "inconsistent class labels."
            )
        self.classes_ = self.predictor_.classes_
        return self

    def predict(self, X):
        """Predict classes through the directly fitted estimator."""
        check_is_fitted(self, "predictor_")
        X_valid = validate_data(self, X, reset=False)
        return self.predictor_.predict(X_valid)

    def predict_proba(self, X):
        """Predict calibrated class probabilities."""
        check_is_fitted(self, "calibrator_")
        X_valid = validate_data(self, X, reset=False)
        return self.calibrator_.predict_proba(X_valid)

    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X):
        """Return decision values from the directly fitted estimator."""
        check_is_fitted(self, "predictor_")
        X_valid = validate_data(self, X, reset=False)
        return self.predictor_.decision_function(X_valid)
