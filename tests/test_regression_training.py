# SPDX-License-Identifier: LGPL-3.0-or-later
"""Focused tests for MELITE's task-specific regression trainer."""

from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.datasets import load_diabetes, make_regression
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.utils.validation import check_is_fitted

import melite.regression_training as regression_training
from melite.optimization_policy import OPTIMIZATION_POLICY
from melite.regression_optimization import (
    OptunaSearchRegressor,
    RegressionOptimizationResult,
    build_regression_inner_cv,
)
from melite.regression_training import (
    MAX_POLYNOMIAL_FEATURES,
    RegressionModelTrainer,
    expected_polynomial_feature_count,
)


def _config(active_regressors, random_state=42, n_trials=2):
    return SimpleNamespace(
        ACTIVE_REGRESSORS=active_regressors,
        RANDOM_STATE=random_state,
        N_TRIALS=n_trials,
        CV_CONFIG={"n_splits": 2, "n_repeats": 1, "inner_n_splits": 2},
    )


def _evaluation(regressor_key, rmse):
    return {
        "regressor_key": regressor_key,
        "rmse": rmse,
        "rmse_std": 0.1,
        "mae": 0.8,
        "mae_std": 0.2,
        "r2": 0.7,
        "r2_std": 0.3,
        "outer_scores": [],
        "optimization_searches": [],
        "selected": False,
    }


@pytest.mark.parametrize("regressor_key", ["linear", "poly", "ridge", "lasso"])
def test_regression_builders_fit_and_produce_finite_prediction_vectors(regressor_key):
    X, y = make_regression(
        n_samples=30,
        n_features=4,
        noise=0.1,
        random_state=42,
    )
    model = RegressionModelTrainer(_config([regressor_key])).model_builders[
        regressor_key
    ]()

    model.fit(X, y)
    predictions = model.predict(X)

    assert predictions.shape == (30,)
    assert np.all(np.isfinite(predictions))


def test_regression_builders_preserve_approved_estimator_contracts():
    trainer = RegressionModelTrainer(_config(["linear", "poly", "ridge", "lasso"]))
    linear = trainer.model_builders["linear"]()
    poly = trainer.model_builders["poly"]()
    ridge = trainer.model_builders["ridge"]()
    lasso = trainer.model_builders["lasso"]()

    assert isinstance(linear, LinearRegression)
    assert isinstance(poly, SklearnPipeline)
    assert list(poly.named_steps) == ["scaler", "poly", "linear"]
    assert isinstance(poly.named_steps["scaler"], StandardScaler)
    assert isinstance(poly.named_steps["poly"], PolynomialFeatures)
    assert poly.named_steps["poly"].degree == 2
    assert poly.named_steps["poly"].include_bias is False
    assert isinstance(poly.named_steps["linear"], LinearRegression)
    assert isinstance(ridge.named_steps["scaler"], StandardScaler)
    assert isinstance(ridge.named_steps["ridge"], Ridge)
    assert isinstance(lasso.named_steps["scaler"], StandardScaler)
    assert isinstance(lasso.named_steps["lasso"], Lasso)
    assert lasso.named_steps["lasso"].max_iter == 10_000


def test_active_regressors_require_a_known_nonempty_ordered_subset():
    trainer = RegressionModelTrainer(_config(["lasso", "linear"]))
    assert trainer.active_regressors == ["lasso", "linear"]

    with pytest.raises(ValueError, match="at least one"):
        RegressionModelTrainer(_config([]))
    with pytest.raises(ValueError, match="Unknown active regressor.*unknown"):
        RegressionModelTrainer(_config(["linear", "unknown"]))


def test_outer_and_inner_cv_use_regression_specific_strategies_and_seed():
    outer = RegressionModelTrainer(
        _config(["linear"], random_state=17)
    )._build_outer_cv()
    inner = build_regression_inner_cv(inner_n_splits=3, random_state=17)

    assert isinstance(outer, RepeatedKFold)
    assert outer.cvargs["n_splits"] == 2
    assert outer.n_repeats == 1
    assert outer.random_state == 17
    assert isinstance(inner, KFold)
    assert inner.n_splits == 3
    assert inner.shuffle is True
    assert inner.random_state == 17


def test_metric_aggregation_restores_positive_rmse_and_mae(monkeypatch):
    trainer = RegressionModelTrainer(_config(["linear"]))
    captured = {}

    def fake_cross_validate(*_args, **kwargs):
        captured.update(kwargs)
        return {
            "test_rmse": np.array([-2.0, -4.0]),
            "test_mae": np.array([-1.0, -3.0]),
            "test_r2": np.array([0.5, -0.1]),
        }

    monkeypatch.setattr(regression_training, "cross_validate", fake_cross_validate)
    result = trainer._cross_validate_model_with_scores(
        object(), np.ones((6, 2)), np.arange(6, dtype=float)
    )

    assert captured["scoring"] == {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }
    assert captured["error_score"] == "raise"
    assert result["rmse"] == pytest.approx(3.0)
    assert result["rmse_std"] == pytest.approx(1.0)
    assert result["mae"] == pytest.approx(2.0)
    assert result["mae_std"] == pytest.approx(1.0)
    assert result["r2"] == pytest.approx(0.2)
    assert result["r2_std"] == pytest.approx(0.3)
    assert result["outer_scores"] == [
        {
            "outer_split": 0,
            "outer_repeat": 0,
            "outer_fold": 0,
            "rmse": 2.0,
            "mae": 1.0,
            "r2": 0.5,
        },
        {
            "outer_split": 1,
            "outer_repeat": 0,
            "outer_fold": 1,
            "rmse": 4.0,
            "mae": 3.0,
            "r2": -0.1,
        },
    ]


def test_tunable_outer_folds_preserve_optimization_summaries(monkeypatch):
    trainer = RegressionModelTrainer(_config(["ridge"]))
    wrapper = OptunaSearchRegressor(
        trainer.model_builders["ridge"](),
        regression_training.get_regression_search_space("ridge"),
        2,
        2,
        42,
    )
    first = RegressionOptimizationResult({"ridge__alpha": 0.1}, 1.1, 2, 2, 0)
    second = RegressionOptimizationResult({"ridge__alpha": 1.0}, 1.2, 2, 2, 0)

    def fake_cross_validate(*_args, **kwargs):
        assert kwargs["return_estimator"] is True
        return {
            "test_rmse": np.array([-1.0, -1.2]),
            "test_mae": np.array([-0.8, -0.9]),
            "test_r2": np.array([0.7, 0.6]),
            "estimator": [
                SimpleNamespace(optimization_result_=first),
                SimpleNamespace(optimization_result_=second),
            ],
        }

    monkeypatch.setattr(regression_training, "cross_validate", fake_cross_validate)
    result = trainer._cross_validate_model_with_scores(
        wrapper, np.ones((6, 2)), np.arange(6, dtype=float)
    )

    assert [item["best_params"] for item in result["optimization_searches"]] == [
        {"ridge__alpha": 0.1},
        {"ridge__alpha": 1.0},
    ]
    assert [item["outer_fold"] for item in result["optimization_searches"]] == [0, 1]


def test_load_diabetes_evaluation_produces_regression_evidence_and_final_model():
    X, y = load_diabetes(return_X_y=True)
    trainer = RegressionModelTrainer(_config(["linear"]))

    selected, evaluations = trainer.evaluate_and_select_models(X, y)

    assert len(selected) == 8
    assert selected[0].predict(X[:5]).shape == (5,)
    assert selected[1] == {}
    assert selected[2] >= 0.0
    assert selected[4] >= 0.0
    assert len(evaluations) == 1
    assert len(evaluations[0]["outer_scores"]) == 2
    assert evaluations[0]["selected"] is True
    assert evaluations[0]["final_optimization_search"] is None


@pytest.mark.parametrize(
    ("regressor_key", "alpha_parameter"),
    [("ridge", "ridge__alpha"), ("lasso", "lasso__alpha")],
)
def test_regularized_regression_real_nested_search_and_final_refit(
    regressor_key, alpha_parameter
):
    X, y = make_regression(
        n_samples=24,
        n_features=3,
        noise=0.1,
        random_state=42,
    )
    trainer = RegressionModelTrainer(_config([regressor_key], n_trials=1))

    selected, evaluations = trainer.evaluate_and_select_models(X, y)

    evaluation = evaluations[0]
    assert len(evaluation["outer_scores"]) == 2
    for outer_score in evaluation["outer_scores"]:
        assert np.all(
            np.isfinite(
                [outer_score["rmse"], outer_score["mae"], outer_score["r2"]]
            )
        )

    outer_searches = evaluation["optimization_searches"]
    assert [search["outer_split"] for search in outer_searches] == [0, 1]
    assert [
        (search["outer_repeat"], search["outer_fold"])
        for search in outer_searches
    ] == [(0, 0), (0, 1)]
    assert len(outer_searches) == 2
    for search in outer_searches:
        assert search["n_trials_requested"] == 1
        assert search["n_trials_complete"] == 1
        assert np.isfinite(search["best_inner_rmse"])
        assert 1e-4 <= search["best_params"][alpha_parameter] <= 1e2

    final_search = evaluation["final_optimization_search"]
    assert final_search is not None
    assert final_search.n_trials_requested == 1
    assert final_search.n_trials_complete == 1
    assert np.isfinite(final_search.best_inner_rmse)
    assert final_search.best_params == selected[1]
    assert 1e-4 <= selected[1][alpha_parameter] <= 1e2

    final_model = selected[0]
    check_is_fitted(final_model.named_steps[regressor_key])
    assert final_model.get_params()[alpha_parameter] == selected[1][alpha_parameter]
    predictions = final_model.predict(X)
    assert predictions.shape == (24,)
    assert np.all(np.isfinite(predictions))
    assert np.all(np.isfinite(selected[2:]))


@pytest.mark.parametrize(
    ("scores", "expected_key"),
    [
        ({"linear": 2.0, "poly": 1.0}, "poly"),
        ({"linear": 1.0, "poly": 1.0}, "linear"),
    ],
)
def test_selection_minimizes_rmse_and_preserves_order_on_exact_ties(
    monkeypatch, scores, expected_key
):
    X, y = make_regression(n_samples=12, n_features=2, random_state=42)
    trainer = RegressionModelTrainer(_config(["linear", "poly"]))
    monkeypatch.setattr(
        trainer,
        "_evaluate_regressor",
        lambda key, *_args: _evaluation(key, scores[key]),
    )

    selected, evaluations = trainer.evaluate_and_select_models(X, y)

    assert [item["regressor_key"] for item in evaluations] == ["linear", "poly"]
    assert [item["selected"] for item in evaluations] == [
        expected_key == "linear",
        expected_key == "poly",
    ]
    if expected_key == "linear":
        assert isinstance(selected[0], LinearRegression)
    else:
        assert isinstance(selected[0], SklearnPipeline)


def test_tunable_winner_runs_fresh_full_data_optimization(monkeypatch):
    X, y = make_regression(n_samples=12, n_features=2, random_state=42)
    trainer = RegressionModelTrainer(_config(["ridge"], n_trials=3))
    monkeypatch.setattr(
        trainer,
        "_evaluate_regressor",
        lambda key, *_args: _evaluation(key, 1.0),
    )
    calls = []
    fitted = trainer.model_builders["ridge"]().fit(X, y)
    result = RegressionOptimizationResult({"ridge__alpha": 0.25}, 0.9, 3, 3, 0)

    def fake_optimize(model, space, X_train, y_train, **kwargs):
        calls.append((model, space, X_train, y_train, kwargs))
        return fitted, result

    monkeypatch.setattr(
        regression_training, "optimize_and_refit_regressor", fake_optimize
    )
    selected, evaluations = trainer.evaluate_and_select_models(X, y)

    assert len(calls) == 1
    assert calls[0][2] is X
    assert calls[0][3] is y
    assert calls[0][4] == {
        "inner_n_splits": 2,
        "n_trials": 3,
        "random_state": 42,
    }
    assert selected[0] is fitted
    assert selected[1] == {"ridge__alpha": 0.25}
    assert evaluations[0]["final_optimization_search"] is result


def test_non_tunable_winner_fits_directly_without_optuna(monkeypatch):
    X, y = make_regression(n_samples=12, n_features=2, random_state=42)
    trainer = RegressionModelTrainer(_config(["linear"]))
    monkeypatch.setattr(
        trainer,
        "_evaluate_regressor",
        lambda key, *_args: _evaluation(key, 1.0),
    )
    monkeypatch.setattr(
        regression_training,
        "optimize_and_refit_regressor",
        lambda *_args, **_kwargs: pytest.fail("linear regression must not search"),
    )

    selected, evaluations = trainer.evaluate_and_select_models(X, y)

    assert hasattr(selected[0], "coef_")
    assert selected[1] == {}
    assert evaluations[0]["final_optimization_search"] is None


def test_polynomial_feature_count_matches_fitted_transformer():
    X, y = make_regression(n_samples=20, n_features=4, random_state=42)
    model = RegressionModelTrainer(_config(["poly"])).model_builders["poly"]()

    model.fit(X, y)

    assert expected_polynomial_feature_count(4) == 14
    assert model.named_steps["poly"].n_output_features_ == 14


def test_polynomial_expansion_limit_is_rejected_before_fitting(monkeypatch):
    X = np.ones((4, 140))
    y = np.arange(4, dtype=float)
    trainer = RegressionModelTrainer(_config(["poly"]))
    monkeypatch.setattr(
        trainer,
        "_evaluate_regressor",
        lambda *_args: pytest.fail("oversized polynomial model must not be fitted"),
    )

    with pytest.raises(
        ValueError,
        match="original dimensions=140, expanded dimensions=10010",
    ):
        trainer.evaluate_and_select_models(X, y)

    assert MAX_POLYNOMIAL_FEATURES == 10_000


@pytest.mark.parametrize(
    ("X", "y", "message"),
    [
        (np.ones(4), np.ones(4), "two-dimensional"),
        (np.ones((4, 2)), np.ones((4, 1)), "one-dimensional"),
        (np.ones((4, 2)), np.array([1.0, 2.0, np.nan, 4.0]), "finite"),
    ],
)
def test_regression_input_contract_rejects_invalid_shapes_and_targets(X, y, message):
    trainer = RegressionModelTrainer(_config(["linear"]))
    with pytest.raises(ValueError, match=message):
        trainer.evaluate_and_select_models(X, y)


def test_classification_optimization_contract_remains_f1_maximization():
    assert OPTIMIZATION_POLICY.direction == "maximize"
    assert OPTIMIZATION_POLICY.objective == "f1_macro"
