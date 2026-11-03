# SPDX-License-Identifier: LGPL-3.0-or-later
"""Focused tests for MELITE's regression Optuna contract."""

from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.datasets import make_regression

import melite.regression_optimization as regression_optimization
from melite.regression_optimization import (
    REGRESSION_OPTIMIZATION_DIRECTION,
    _run_regression_study,
)
from melite.regression_search_spaces import get_regression_search_space
from melite.regression_training import RegressionModelTrainer


def _config(active_regressors):
    return SimpleNamespace(
        ACTIVE_REGRESSORS=active_regressors,
        RANDOM_STATE=42,
        N_TRIALS=2,
        CV_CONFIG={"n_splits": 2, "n_repeats": 1, "inner_n_splits": 2},
    )


@pytest.mark.parametrize(
    ("regressor_key", "target"),
    [("ridge", "ridge__alpha"), ("lasso", "lasso__alpha")],
)
def test_regularized_regression_alpha_spaces_are_logarithmic(regressor_key, target):
    space = get_regression_search_space(regressor_key)
    parameter = space.parameters[0]

    assert parameter.name == "alpha"
    assert parameter.target == target
    assert parameter.domain.low == 1e-4
    assert parameter.domain.high == 1e2
    assert parameter.domain.log is True


def test_linear_and_polynomial_regression_have_no_search_space():
    assert get_regression_search_space("linear") is None
    assert get_regression_search_space("poly") is None


def test_regression_study_minimizes_positive_inner_rmse(monkeypatch):
    X, y = make_regression(n_samples=12, n_features=2, random_state=42)
    model = RegressionModelTrainer(_config(["ridge"])).model_builders["ridge"]()
    calls = []
    study_arguments = {}
    real_create_study = regression_optimization.optuna.create_study

    def fake_cross_val_score(*_args, **kwargs):
        calls.append(kwargs)
        return np.array([-2.0, -4.0])

    def recording_create_study(**kwargs):
        study_arguments.update(kwargs)
        return real_create_study(**kwargs)

    monkeypatch.setattr(
        regression_optimization, "cross_val_score", fake_cross_val_score
    )
    monkeypatch.setattr(
        regression_optimization.optuna, "create_study", recording_create_study
    )
    _, result = _run_regression_study(
        model,
        get_regression_search_space("ridge"),
        X,
        y,
        inner_n_splits=2,
        n_trials=1,
        random_state=42,
    )

    assert REGRESSION_OPTIMIZATION_DIRECTION == "minimize"
    assert study_arguments["direction"] == "minimize"
    assert calls[0]["scoring"] == "neg_root_mean_squared_error"
    assert calls[0]["n_jobs"] == 1
    assert calls[0]["error_score"] == "raise"
    assert result.best_inner_rmse == pytest.approx(3.0)
    assert 1e-4 <= result.best_params["ridge__alpha"] <= 1e2
