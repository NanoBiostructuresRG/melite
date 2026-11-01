# SPDX-License-Identifier: LGPL-3.0-or-later
"""Task-specific training and evaluation for MELITE regression models."""

from dataclasses import asdict
from math import comb

import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from melite.regression_optimization import (
    OptunaSearchRegressor,
    optimize_and_refit_regressor,
)
from melite.regression_search_spaces import get_regression_search_space

__all__ = [
    "MAX_POLYNOMIAL_FEATURES",
    "RegressionModelTrainer",
    "expected_polynomial_feature_count",
]

MAX_POLYNOMIAL_FEATURES = 10_000
_POLYNOMIAL_DEGREE = 2


def expected_polynomial_feature_count(
    n_features: int, degree: int = _POLYNOMIAL_DEGREE
) -> int:
    """Return the number of polynomial terms through ``degree``, sans bias."""
    if n_features < 1:
        raise ValueError("n_features must be at least 1.")
    if degree < 1:
        raise ValueError("degree must be at least 1.")
    return comb(n_features + degree, degree) - 1


class RegressionModelTrainer:
    """Evaluate and select configured regressors with nested regression CV."""

    def __init__(self, config):
        self.config = config
        self.model_builders = {
            "linear": lambda: LinearRegression(),
            "poly": lambda: SklearnPipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "poly",
                        PolynomialFeatures(
                            degree=_POLYNOMIAL_DEGREE,
                            include_bias=False,
                        ),
                    ),
                    ("linear", LinearRegression()),
                ]
            ),
            "ridge": lambda: SklearnPipeline(
                [
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge()),
                ]
            ),
            "lasso": lambda: SklearnPipeline(
                [
                    ("scaler", StandardScaler()),
                    ("lasso", Lasso(max_iter=10_000)),
                ]
            ),
        }
        self.active_regressors = self._validate_active_regressors()

    def _validate_active_regressors(self) -> list[str]:
        active_regressors = self.config.ACTIVE_REGRESSORS
        if not active_regressors:
            raise ValueError(
                "ACTIVE_REGRESSORS must contain at least one regressor key."
            )

        unknown = [
            regressor
            for regressor in active_regressors
            if regressor not in self.model_builders
        ]
        if unknown:
            unknown_regressors = ", ".join(unknown)
            valid_regressors = ", ".join(self.model_builders)
            raise ValueError(
                f"Unknown active regressor(s): {unknown_regressors}. "
                f"Valid regressor keys are: {valid_regressors}."
            )
        return list(active_regressors)

    def _build_outer_cv(self) -> RepeatedKFold:
        cv_cfg = self.config.CV_CONFIG
        return RepeatedKFold(
            n_splits=cv_cfg["n_splits"],
            n_repeats=cv_cfg["n_repeats"],
            random_state=self.config.RANDOM_STATE,
        )

    def _build_cv_strategy(self) -> RepeatedKFold:
        """Return the outer regression CV strategy."""
        return self._build_outer_cv()

    @staticmethod
    def _validate_inputs(X, y) -> tuple[np.ndarray, np.ndarray]:
        X_array = np.asarray(X)
        y_array = np.asarray(y)
        if X_array.ndim != 2:
            raise ValueError("Regression features must be a two-dimensional matrix.")
        if y_array.ndim != 1:
            raise ValueError("Regression targets must be a one-dimensional vector.")
        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError(
                "Regression features and targets must have equal row counts."
            )
        if X_array.shape[1] < 1:
            raise ValueError("Regression features must contain at least one column.")
        for name, values in (("features", X_array), ("targets", y_array)):
            if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
                values.dtype, np.complexfloating
            ):
                raise ValueError(f"Regression {name} must be real numeric values.")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Regression {name} must contain only finite values.")
        return X_array, y_array

    def _validate_polynomial_expansion(self, n_features: int) -> int:
        expanded_features = expected_polynomial_feature_count(n_features)
        if expanded_features > MAX_POLYNOMIAL_FEATURES:
            raise ValueError(
                "PolynomialRegression expansion exceeds 10,000 features: "
                f"original dimensions={n_features}, "
                f"expanded dimensions={expanded_features}."
            )
        return expanded_features

    def cross_validate_model(self, model, X_train, y_train):
        """Return RMSE, MAE, and R-squared aggregate metrics and deviations."""
        evaluation = self._cross_validate_model_with_scores(model, X_train, y_train)
        return (
            evaluation["rmse"],
            evaluation["rmse_std"],
            evaluation["mae"],
            evaluation["mae_std"],
            evaluation["r2"],
            evaluation["r2_std"],
        )

    def _cross_validate_model_with_scores(self, model, X_train, y_train) -> dict:
        cv_cfg = self.config.CV_CONFIG
        tunable = isinstance(model, OptunaSearchRegressor)
        cross_validate_kwargs = {
            "scoring": {
                "rmse": "neg_root_mean_squared_error",
                "mae": "neg_mean_absolute_error",
                "r2": "r2",
            },
            "cv": self._build_outer_cv(),
            "n_jobs": 1,
            "return_train_score": False,
            "error_score": "raise",
        }
        if tunable:
            cross_validate_kwargs["return_estimator"] = True
        scores = cross_validate(model, X_train, y_train, **cross_validate_kwargs)

        rmse_values = -np.asarray(scores["test_rmse"], dtype=float)
        mae_values = -np.asarray(scores["test_mae"], dtype=float)
        r2_values = np.asarray(scores["test_r2"], dtype=float)
        n_splits = cv_cfg["n_splits"]
        outer_scores = [
            {
                "outer_split": outer_split,
                "outer_repeat": outer_split // n_splits,
                "outer_fold": outer_split % n_splits,
                "rmse": rmse_values[outer_split],
                "mae": mae_values[outer_split],
                "r2": r2_values[outer_split],
            }
            for outer_split in range(len(rmse_values))
        ]

        optimization_searches = []
        if tunable:
            fitted_wrappers = scores.pop("estimator")
            optimization_searches = [
                {
                    "outer_split": outer_split,
                    "outer_repeat": outer_split // n_splits,
                    "outer_fold": outer_split % n_splits,
                    **asdict(wrapper.optimization_result_),
                }
                for outer_split, wrapper in enumerate(fitted_wrappers)
            ]

        return {
            "rmse": float(rmse_values.mean()),
            "rmse_std": float(rmse_values.std()),
            "mae": float(mae_values.mean()),
            "mae_std": float(mae_values.std()),
            "r2": float(r2_values.mean()),
            "r2_std": float(r2_values.std()),
            "outer_scores": outer_scores,
            "optimization_searches": optimization_searches,
        }

    def _evaluate_regressor(self, regressor_key, X_train, y_train) -> dict:
        model = self.model_builders[regressor_key]()
        search_space = get_regression_search_space(regressor_key)
        evaluation_estimator = (
            model
            if search_space is None
            else OptunaSearchRegressor(
                estimator=model,
                search_space=search_space,
                inner_n_splits=self.config.CV_CONFIG["inner_n_splits"],
                n_trials=self.config.N_TRIALS,
                random_state=self.config.RANDOM_STATE,
            )
        )
        evaluation = self._cross_validate_model_with_scores(
            evaluation_estimator, X_train, y_train
        )
        return {
            "regressor_key": regressor_key,
            **evaluation,
            "selected": False,
        }

    def evaluate_and_select_models(self, X_train, y_train):
        """Evaluate regressors, minimize mean RMSE, then fit the winner afresh."""
        X_array, y_array = self._validate_inputs(X_train, y_train)
        if "poly" in self.active_regressors:
            self._validate_polynomial_expansion(X_array.shape[1])

        evaluations = [
            self._evaluate_regressor(regressor_key, X_array, y_array)
            for regressor_key in self.active_regressors
        ]
        best = evaluations[0]
        for evaluation in evaluations[1:]:
            if evaluation["rmse"] < best["rmse"]:
                best = evaluation
        best["selected"] = True

        winning_model = self.model_builders[best["regressor_key"]]()
        winning_space = get_regression_search_space(best["regressor_key"])
        if winning_space is None:
            winning_model.fit(X_array, y_array)
            params = {}
            best["final_optimization_search"] = None
        else:
            winning_model, optimization_result = optimize_and_refit_regressor(
                winning_model,
                winning_space,
                X_array,
                y_array,
                inner_n_splits=self.config.CV_CONFIG["inner_n_splits"],
                n_trials=self.config.N_TRIALS,
                random_state=self.config.RANDOM_STATE,
            )
            params = optimization_result.best_params
            best["final_optimization_search"] = optimization_result

        selected_result = (
            winning_model,
            params,
            best["rmse"],
            best["rmse_std"],
            best["mae"],
            best["mae_std"],
            best["r2"],
            best["r2_std"],
        )
        return selected_result, evaluations

    def train_and_select_best_model(self, X_train, y_train):
        """Train all active regressors and return the selected result tuple."""
        selected_result, _ = self.evaluate_and_select_models(X_train, y_train)
        return selected_result
