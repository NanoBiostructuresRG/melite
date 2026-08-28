# SPDX-License-Identifier: LGPL-3.0-or-later
"""Optuna execution engine for MELITE classifier optimization."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
from optuna.trial import TrialState
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

from melite.optimization_policy import OPTIMIZATION_POLICY
from melite.search_spaces import (
    CategoricalDomain,
    ClassifierSearchSpace,
    FloatDomain,
    IntDomain,
    ParameterSpec,
)


@dataclass(frozen=True)
class OptimizationResult:
    """Data-only summary of one optimization search."""

    best_params: dict[str, Any]
    best_inner_f1_macro: float
    n_trials_requested: int
    n_trials_complete: int
    n_trials_failed: int


class TrialEvaluationError(RuntimeError):
    """A candidate failed while fitting or scoring on the fixed inner folds."""


class OptimizationSearchError(RuntimeError):
    """An optimization study violated MELITE's search-completion contract."""


def get_optimization_backend_info() -> dict[str, str]:
    """Identify the optimization backend for provenance metadata.

    This helper does not participate in optimization execution.
    """
    return {"name": "optuna", "version": optuna.__version__}


def _suggest_parameter(trial, parameter: ParameterSpec) -> Any:
    domain = parameter.domain
    if isinstance(domain, FloatDomain):
        return trial.suggest_float(
            parameter.name,
            domain.low,
            domain.high,
            log=domain.log,
        )
    if isinstance(domain, IntDomain):
        return trial.suggest_int(
            parameter.name,
            domain.low,
            domain.high,
            step=domain.step,
        )
    if isinstance(domain, CategoricalDomain):
        return trial.suggest_categorical(parameter.name, domain.choices)
    raise TypeError(f"Unsupported domain for parameter {parameter.name!r}: {domain!r}.")


def _suggest_logical_params(
    trial, search_space: ClassifierSearchSpace
) -> dict[str, Any]:
    logical_params = {
        parameter.name: _suggest_parameter(trial, parameter)
        for parameter in search_space.common_parameters
    }
    if search_space.selector is None:
        return logical_params

    selector_value = trial.suggest_categorical(
        search_space.selector.name,
        search_space.selector.choices,
    )
    logical_params[search_space.selector.name] = selector_value
    branch = search_space.branch_for(selector_value)
    for parameter in branch.parameters:
        logical_params[parameter.name] = _suggest_parameter(trial, parameter)
    return logical_params


def _materialize_effective_params(
    search_space: ClassifierSearchSpace,
    logical_params: dict[str, Any],
) -> dict[str, Any]:
    effective_params: dict[str, Any] = {}
    consumed: set[str] = set()

    def consume(parameter: ParameterSpec) -> None:
        if parameter.name not in logical_params:
            raise ValueError(f"Missing logical parameter {parameter.name!r}.")
        effective_params[parameter.target] = logical_params[parameter.name]
        consumed.add(parameter.name)

    for parameter in search_space.common_parameters:
        consume(parameter)

    if search_space.selector is not None:
        selector = search_space.selector
        if selector.name not in logical_params:
            raise ValueError(f"Missing logical selector {selector.name!r}.")
        selector_value = logical_params[selector.name]
        consumed.add(selector.name)
        branch = search_space.branch_for(selector_value)
        if selector.target is not None:
            effective_params[selector.target] = selector_value
        for fixed_parameter in branch.fixed_parameters:
            effective_params[fixed_parameter.target] = fixed_parameter.value
        for parameter in branch.parameters:
            consume(parameter)

    extra = sorted(set(logical_params) - consumed)
    if extra:
        raise ValueError(f"Unexpected logical parameter(s): {', '.join(extra)}.")
    return effective_params


def _run_study(
    estimator,
    search_space: ClassifierSearchSpace,
    X,
    y,
    *,
    inner_n_splits: int,
    n_trials: int,
    random_state: int,
):
    """Run one fresh study and return it with its data-only result."""
    policy = OPTIMIZATION_POLICY
    if policy.sampler != "tpe" or policy.storage != "in_memory" or policy.pruning:
        raise RuntimeError("Unsupported MELITE optimization policy configuration.")

    inner_cv = StratifiedKFold(
        n_splits=inner_n_splits,
        shuffle=True,
        random_state=random_state,
    )
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
        direction=policy.direction,
        storage=None,
    )

    def objective(trial) -> float:
        logical_params = _suggest_logical_params(trial, search_space)
        effective_params = _materialize_effective_params(search_space, logical_params)
        candidate = clone(estimator)
        candidate.set_params(**effective_params)
        try:
            scores = cross_val_score(
                candidate,
                X,
                y,
                scoring=policy.objective,
                cv=inner_splits,
                n_jobs=1,
                error_score="raise",
            )
        except Exception as exc:  # noqa: BLE001 - candidate-evaluation boundary
            raise TrialEvaluationError("Candidate fitting or scoring failed.") from exc
        if not np.all(np.isfinite(scores)):
            raise TrialEvaluationError("Candidate produced non-finite inner-CV scores.")
        return float(np.mean(scores))

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
            "Optimization completed with zero successful candidate trials."
        )

    result = OptimizationResult(
        best_params=_materialize_effective_params(
            search_space,
            dict(study.best_trial.params),
        ),
        best_inner_f1_macro=float(study.best_value),
        n_trials_requested=n_trials,
        n_trials_complete=n_complete,
        n_trials_failed=n_failed,
    )
    return study, result


def optimize_and_refit(
    estimator,
    search_space: ClassifierSearchSpace,
    X,
    y,
    *,
    inner_n_splits: int,
    n_trials: int,
    random_state: int,
) -> tuple[Any, OptimizationResult]:
    """Optimize an estimator and refit its best configuration on all input data."""
    _, result = _run_study(
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


def _estimator_has(attribute: str):
    def check(self) -> bool:
        estimator = getattr(self, "estimator_", self.estimator)
        return hasattr(estimator, attribute)

    return check


class OptunaSearchClassifier(ClassifierMixin, BaseEstimator):
    """Sklearn-compatible classifier that optimizes and refits on ``fit``."""

    def __init__(
        self,
        estimator,
        search_space: ClassifierSearchSpace,
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
        self.estimator_, self.optimization_result_ = optimize_and_refit(
            self.estimator,
            self.search_space,
            X,
            y,
            inner_n_splits=self.inner_n_splits,
            n_trials=self.n_trials,
            random_state=self.random_state,
        )
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)

    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.decision_function(X)

    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)


@contextmanager
def optuna_logging_scope(*, verbose: bool):
    """Set Optuna verbosity for one MELITE run and restore it afterward."""
    previous_verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(
        optuna.logging.INFO if verbose else optuna.logging.WARNING
    )
    try:
        yield
    finally:
        optuna.logging.set_verbosity(previous_verbosity)
