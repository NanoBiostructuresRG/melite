# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for MELITE's internal probability-calibration adapter."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.svm import SVC

from melite.probability_calibration import ProbabilityCalibratedClassifier


X, Y = make_classification(
    n_samples=40,
    n_features=5,
    n_informative=3,
    n_redundant=0,
    random_state=42,
)


def _adapter():
    return ProbabilityCalibratedClassifier(
        estimator=SVC(C=1.5, random_state=42),
        max_calibration_cv=5,
    )


def test_probability_adapter_is_sklearn_cloneable():
    original = _adapter()

    cloned = clone(original)

    assert isinstance(cloned, ProbabilityCalibratedClassifier)
    assert cloned is not original
    assert cloned.estimator is not original.estimator
    assert cloned.estimator.get_params() == original.estimator.get_params()
    assert cloned.max_calibration_cv == 5


def test_fit_creates_independent_direct_predictor_and_calibrator():
    original_estimator = SVC(C=1.5, random_state=42)
    adapter = ProbabilityCalibratedClassifier(original_estimator).fit(X, Y)

    assert adapter.predictor_ is not original_estimator
    assert isinstance(adapter.predictor_, SVC)
    assert hasattr(adapter.predictor_, "support_")
    assert isinstance(adapter.calibrator_, CalibratedClassifierCV)
    assert adapter.calibrator_.ensemble is False
    assert adapter.calibrator_.estimator is not original_estimator
    assert adapter.calibration_cv_ == 5
    assert np.array_equal(adapter.classes_, adapter.predictor_.classes_)
    assert np.array_equal(adapter.classes_, adapter.calibrator_.classes_)


def test_fit_records_validated_feature_count():
    adapter = _adapter().fit(X, Y)

    assert adapter.n_features_in_ == X.shape[1]


@pytest.mark.parametrize(
    "response_method",
    ["predict", "predict_proba", "decision_function"],
)
def test_response_methods_reject_the_wrong_number_of_features(response_method):
    adapter = _adapter().fit(X, Y)

    with pytest.raises(ValueError, match="X has 4 features"):
        getattr(adapter, response_method)(X[:, :-1])


def test_predict_and_predict_proba_delegate_to_independent_fitted_paths(monkeypatch):
    adapter = _adapter().fit(X, Y)
    expected_predictions = np.array([1, 0, 1])
    expected_probabilities = np.array(
        [
            [0.1, 0.9],
            [0.8, 0.2],
            [0.3, 0.7],
        ]
    )
    monkeypatch.setattr(
        adapter.predictor_,
        "predict",
        lambda _X: expected_predictions,
    )
    monkeypatch.setattr(
        adapter.calibrator_,
        "predict_proba",
        lambda _X: expected_probabilities,
    )

    assert adapter.predict(X[:3]) is expected_predictions
    assert adapter.predict_proba(X[:3]) is expected_probabilities


def test_svc_decision_function_is_preserved_from_direct_predictor():
    adapter = _adapter().fit(X, Y)

    assert hasattr(adapter, "decision_function")
    assert np.allclose(
        adapter.decision_function(X),
        adapter.predictor_.decision_function(X),
    )


def test_calibrated_probability_rows_sum_to_one():
    adapter = _adapter().fit(X, Y)

    probabilities = adapter.predict_proba(X)

    assert probabilities.shape == (40, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    ("minimum_class_count", "expected_calibration_cv"),
    [(2, 2), (3, 3), (4, 4), (5, 5), (6, 5)],
)
def test_calibration_cv_adapts_to_minimum_class_count(
    minimum_class_count,
    expected_calibration_cv,
):
    rng = np.random.default_rng(minimum_class_count)
    y = np.array(
        [0] * minimum_class_count + [1] * (minimum_class_count + 1),
        dtype=np.int64,
    )
    X_adaptive = rng.normal(size=(len(y), 3))
    X_adaptive[y == 1] += 1.0

    adapter = _adapter().fit(X_adaptive, y)

    assert adapter.calibration_cv_ == expected_calibration_cv


def test_calibration_rejects_one_class_target():
    X_one_class = np.arange(9, dtype=float).reshape(3, 3)
    y_one_class = np.zeros(3, dtype=np.int64)

    with pytest.raises(ValueError, match="one class"):
        _adapter().fit(X_one_class, y_one_class)


def test_calibration_rejects_multiclass_target_with_a_single_sample_class():
    X_small = np.arange(9, dtype=float).reshape(3, 3)
    y_small = np.array([0, 1, 1])

    with pytest.raises(
        ValueError,
        match="requires at least two samples in every class",
    ):
        _adapter().fit(X_small, y_small)


def test_continuous_target_uses_sklearn_classification_target_validation():
    y_continuous = np.linspace(0.0, 1.0, len(Y))

    with pytest.raises(ValueError, match="Unknown label type|continuous"):
        _adapter().fit(X, y_continuous)
