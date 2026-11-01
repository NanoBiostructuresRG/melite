# SPDX-License-Identifier: LGPL-3.0-or-later
"""Optuna execution engine for MELITE regression optimization."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
from optuna.trial import TrialState
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score
from sklearn.utils.validation import check_is_fitted

from melite.optimization import OptimizationSearchError, TrialEvaluationError
from melite.optimization_policy import OPTIMIZATION_POLICY
from melite.regression_search_spaces import RegressionSearchSpace
from melite.search_spaces import FloatDomain

__all__ = [
    "REGRESSION_OPTIMIZATION_DIRECTION",
    "RegressionOptimizationResult",
    "OptunaSearchRegressor",
    "build_regression_inner_cv",
    "optimize_and_refit_regressor",
]

REGRESSION_OPTIMIZATION_DIRECTION = "minimize"


@dataclass(frozen=True)
class RegressionOptimizationResult:
    """Data-only summary of one regression optimization search."""

    best_params: dict[str, Any]
    best_inner_rmse: float
    n_trials_requested: int
    n_trials_complete: int
    n_trials_failed: int


def build_regression_inner_cv(inner_n_splits: int, random_state: int) -> KFold:
    """Build the shuffled inner regression folds."""
    return KFold(
        n_splits=inner_n_splits,
        shuffle=True,
        random_state=random_state,
    )


def _materialize_effective_params(
    search_space: RegressionSearchSpace,
    logical_params: dict[str, Any],
) -> dict[str, Any]:
    return {
        parameter.target: logical_params[parameter.name]
        for parameter in search_space.parameters
    }


def _suggest_effective_params(trial, search_space: RegressionSearchSpace) -> dict:
    logical_params = {}
    for parameter in search_space.parameters:
        domain = parameter.domain
        if not isinstance(domain, FloatDomain):
            raise TypeError(
                "Regression optimization currently supports FloatDomain only."
            )
        logical_params[parameter.name] = trial.suggest_float(
            parameter.name,
            domain.low,
            domain.high,
            log=domain.log,
        )
    return _materialize_effective_params(search_space, logical_params)


def _run_regression_study(
    estimator,
    search_space: RegressionSearchSpace,
    X,
    y,
    *,
    inner_n_splits: int,
    n_trials: int,
    random_state: int,
):
    """Run one seeded regression study and return its data-only result."""
    policy = OPTIMIZATION_POLICY
    inner_cv = build_regression_inner_cv(inner_n_splits, random_state)
    inner_splits = list(inner_cv.split(X, y))
    sampler = optuna.samplers.TPESampler(
        seed=random_state,
        n_startup_trials=policy.n_startup_trials,
        multivariate=policy.multivariate,
        group=policy.group,
        constant_liar=policy.constant_liar,
    )
    study = optuna.create_study(
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
        direction=REGRESSION_OPTIMIZATION_DIRECTION,
        storage=None,
    )

    def objective(trial) -> float:
        effective_params = _suggest_effective_params(trial, search_space)
        candidate = clone(estimator)
        candidate.set_params(**effective_params)
        try:
            scores = cross_val_score(
                candidate,
                X,
                y,
                scoring="neg_root_mean_squared_error",
                cv=inner_splits,
                n_jobs=1,
                error_score="raise",
            )
        except Exception as exc:  # noqa: BLE001 - candidate-evaluation boundary
            raise TrialEvaluationError("Candidate fitting or scoring failed.") from exc
        if not np.all(np.isfinite(scores)):
            raise TrialEvaluationError("Candidate produced non-finite inner-CV scores.")
        return float(-np.mean(scores))

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=policy.n_jobs,
        catch=(TrialEvaluationError,),
        show_progress_bar=False,
    )

    states = [trial.state for trial in study.trials]
    if TrialState.PRUNED in states:
        raise OptimizationSearchError(
            "A PRUNED trial violated MELITE's no-pruning optimization contract."
        )
    n_complete = states.count(TrialState.COMPLETE)
    n_failed = states.count(TrialState.FAIL)
    if n_complete == 0:
        raise OptimizationSearchError(
            "Regression optimization completed with zero successful candidate trials."
        )

    result = RegressionOptimizationResult(
        best_params=_materialize_effective_params(
            search_space, dict(study.best_trial.params)
        ),
        best_inner_rmse=float(study.best_value),
        n_trials_requested=n_trials,
        n_trials_complete=n_complete,
        n_trials_failed=n_failed,
    )
    return study, result


def optimize_and_refit_regressor(
    estimator,
    search_space: RegressionSearchSpace,
    X,
    y,
    *,
    inner_n_splits: int,
    n_trials: int,
    random_state: int,
) -> tuple[Any, RegressionOptimizationResult]:
    """Optimize a regressor and refit its best configuration on all input data."""
    _, result = _run_regression_study(
        estimator,
        search_space,
        X,
        y,
        inner_n_splits=inner_n_splits,
        n_trials=n_trials,
        random_state=random_state,
    )
    fitted_estimator = clone(estimator)
    fitted_estimator.set_params(**result.best_params)
    fitted_estimator.fit(X, y)
    return fitted_estimator, result


class OptunaSearchRegressor(RegressorMixin, BaseEstimator):
    """Scikit-learn regressor that optimizes and refits during ``fit``."""

    def __init__(
        self,
        estimator,
        search_space: RegressionSearchSpace,
        inner_n_splits: int,
        n_trials: int,
        random_state: int,
    ):
        self.estimator = estimator
        self.search_space = search_space
        self.inner_n_splits = inner_n_splits
        self.n_trials = n_trials
        self.random_state = random_state

    def fit(self, X, y):
        self.estimator_, self.optimization_result_ = optimize_and_refit_regressor(
            self.estimator,
            self.search_space,
            X,
            y,
            inner_n_splits=self.inner_n_splits,
            n_trials=self.n_trials,
            random_state=self.random_state,
        )
        if hasattr(self.estimator_, "n_features_in_"):
            self.n_features_in_ = self.estimator_.n_features_in_
        return self

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)
