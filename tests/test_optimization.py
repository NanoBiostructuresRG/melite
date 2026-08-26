# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for MELITE's Optuna execution engine."""

import ast
import logging
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import optuna
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import melite.optimization as optimization_module
import melite
from melite.optimization import (
    OptimizationResult,
    OptimizationSearchError,
    OptunaSearchClassifier,
    TrialEvaluationError,
    _materialize_effective_params,
    _run_study,
    _suggest_logical_params,
    _suggest_parameter,
    optimize_and_refit,
    optuna_logging_scope,
)
from melite.optimization_policy import OPTIMIZATION_POLICY
from melite.search_spaces import (
    CategoricalDomain,
    ClassifierSearchSpace,
    FloatDomain,
    IntDomain,
    ParameterSpec,
    get_search_space,
)


X, Y = make_classification(
    n_samples=30,
    n_features=4,
    n_informative=3,
    n_redundant=0,
    random_state=42,
)
_REPOSITORY_ROOT = Path(__file__).parents[1]


def _tree_space():
    return ClassifierSearchSpace(
        "tree",
        common_parameters=(ParameterSpec("max_depth", "max_depth", IntDomain(1, 3)),),
    )


class RecordingTrial:
    def __init__(self, values):
        self.values = values
        self.calls = []

    def suggest_float(self, name, low, high, *, log):
        self.calls.append(("float", name, low, high, log))
        return self.values[name]

    def suggest_int(self, name, low, high, *, step):
        self.calls.append(("int", name, low, high, step))
        return self.values[name]

    def suggest_categorical(self, name, choices):
        self.calls.append(("categorical", name, choices))
        return self.values[name]


@pytest.mark.parametrize(
    ("parameter", "expected_call"),
    [
        (
            ParameterSpec("rate", "rate", FloatDomain(0.01, 1.0, log=True)),
            ("float", "rate", 0.01, 1.0, True),
        ),
        (
            ParameterSpec("depth", "depth", IntDomain(1, 5, step=2)),
            ("int", "depth", 1, 5, 2),
        ),
        (
            ParameterSpec("kind", "kind", CategoricalDomain(("a", "b"))),
            ("categorical", "kind", ("a", "b")),
        ),
    ],
)
def test_domain_translation_uses_the_matching_optuna_suggestion(
    parameter, expected_call
):
    trial = RecordingTrial(
        {
            parameter.name: parameter.domain.low
            if hasattr(parameter.domain, "low")
            else "a"
        }
    )

    _suggest_parameter(trial, parameter)

    assert trial.calls == [expected_call]


def test_selector_suggestion_order_and_effective_target_mapping():
    space = get_search_space("svc")
    trial = RecordingTrial({"C": 2.0, "kernel": "rbf", "gamma": 0.1})

    logical = _suggest_logical_params(trial, space)
    effective = _materialize_effective_params(space, logical)

    assert [call[1] for call in trial.calls] == ["C", "kernel", "gamma"]
    assert effective == {
        "svc__C": 2.0,
        "svc__kernel": "rbf",
        "svc__gamma": 0.1,
    }


@pytest.mark.parametrize(
    ("classifier_key", "logical", "expected"),
    [
        (
            "rf",
            {
                "n_estimators": 200,
                "max_features": "sqrt",
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "depth_mode": "unbounded",
            },
            {
                "n_estimators": 200,
                "max_features": "sqrt",
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_depth": None,
            },
        ),
        (
            "xgb",
            {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "max_depth": 4,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
                "gamma_mode": "zero",
            },
            {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "max_depth": 4,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
                "gamma": 0.0,
            },
        ),
    ],
)
def test_melite_only_selector_is_excluded_and_fixed_value_is_materialized(
    classifier_key, logical, expected
):
    assert (
        _materialize_effective_params(get_search_space(classifier_key), logical)
        == expected
    )


def test_materialization_rejects_missing_and_extra_logical_parameters():
    space = _tree_space()
    with pytest.raises(ValueError, match="Missing logical parameter"):
        _materialize_effective_params(space, {})
    with pytest.raises(ValueError, match="Unexpected logical parameter"):
        _materialize_effective_params(space, {"max_depth": 2, "extra": 1})


def test_wrapper_is_cloneable_and_compares_search_space_by_value():
    wrapper = OptunaSearchClassifier(
        DecisionTreeClassifier(random_state=42), _tree_space(), 2, 2, 42
    )

    cloned = clone(wrapper)

    assert cloned.search_space == wrapper.search_space
    assert cloned.estimator is not wrapper.estimator


def test_wrapper_fit_stores_result_classes_and_delegates_prediction():
    wrapper = OptunaSearchClassifier(
        DecisionTreeClassifier(random_state=42), _tree_space(), 2, 2, 42
    ).fit(X, Y)

    assert isinstance(wrapper.estimator_, DecisionTreeClassifier)
    assert isinstance(wrapper.optimization_result_, OptimizationResult)
    assert np.array_equal(wrapper.classes_, np.array([0, 1]))
    assert wrapper.predict(X).shape == (30,)


def test_wrapper_conditionally_exposes_classifier_response_methods():
    svc = OptunaSearchClassifier(
        Pipeline([("scaler", StandardScaler()), ("svc", SVC(probability=False))]),
        get_search_space("svc"),
        2,
        1,
        42,
    )
    rf = OptunaSearchClassifier(
        RandomForestClassifier(n_estimators=2, random_state=42),
        get_search_space("rf"),
        2,
        1,
        42,
    )

    assert hasattr(svc, "decision_function")
    assert not hasattr(svc, "predict_proba")
    assert not hasattr(rf, "decision_function")
    assert hasattr(rf, "predict_proba")


def test_all_trials_in_one_study_receive_the_same_materialized_splits(monkeypatch):
    candidate_calls = []

    def fake_cross_val_score(*_args, **kwargs):
        candidate_calls.append(kwargs)
        return np.array([0.7, 0.8])

    monkeypatch.setattr(optimization_module, "cross_val_score", fake_cross_val_score)

    _run_study(
        DecisionTreeClassifier(),
        _tree_space(),
        X,
        Y,
        inner_n_splits=2,
        n_trials=3,
        random_state=42,
    )

    assert len(candidate_calls) == 3
    assert all(call["cv"] is candidate_calls[0]["cv"] for call in candidate_calls)
    assert all(call["scoring"] == "f1_macro" for call in candidate_calls)
    assert all(call["n_jobs"] == 1 for call in candidate_calls)
    assert all(call["error_score"] == "raise" for call in candidate_calls)


def test_each_search_uses_fresh_explicitly_configured_sampler_and_study(monkeypatch):
    sampler_calls = []
    optimize_calls = []
    real_sampler = optuna.samplers.TPESampler
    real_optimize = optuna.study.Study.optimize

    def recording_sampler(**kwargs):
        sampler_calls.append(kwargs)
        return real_sampler(**kwargs)

    def recording_optimize(self, *args, **kwargs):
        optimize_calls.append(kwargs)
        return real_optimize(self, *args, **kwargs)

    monkeypatch.setattr(
        optimization_module.optuna.samplers, "TPESampler", recording_sampler
    )
    monkeypatch.setattr(optuna.study.Study, "optimize", recording_optimize)

    first, _ = _run_study(
        DecisionTreeClassifier(),
        _tree_space(),
        X,
        Y,
        inner_n_splits=2,
        n_trials=1,
        random_state=17,
    )
    second, _ = _run_study(
        DecisionTreeClassifier(),
        _tree_space(),
        X,
        Y,
        inner_n_splits=2,
        n_trials=1,
        random_state=17,
    )

    expected_sampler = {
        "seed": 17,
        "n_startup_trials": OPTIMIZATION_POLICY.n_startup_trials,
        "multivariate": False,
        "group": False,
        "constant_liar": False,
    }
    assert sampler_calls == [expected_sampler, expected_sampler]
    assert first is not second
    assert isinstance(first.pruner, optuna.pruners.NopPruner)
    assert all(call["n_jobs"] == 1 for call in optimize_calls)
    assert all(call["show_progress_bar"] is False for call in optimize_calls)
    assert all(call["catch"] == (TrialEvaluationError,) for call in optimize_calls)


def test_candidate_failure_becomes_fail_and_does_not_top_up(monkeypatch):
    calls = 0

    def sometimes_fails(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("candidate failure")
        return np.array([0.8, 0.8])

    monkeypatch.setattr(optimization_module, "cross_val_score", sometimes_fails)
    study, result = _run_study(
        DecisionTreeClassifier(),
        _tree_space(),
        X,
        Y,
        inner_n_splits=2,
        n_trials=3,
        random_state=42,
    )

    assert [trial.state for trial in study.trials].count(
        optuna.trial.TrialState.FAIL
    ) == 1
    assert len(study.trials) == result.n_trials_requested == 3
    assert result.n_trials_failed == 1
    assert result.n_trials_complete == 2


def test_zero_complete_trials_raise_search_error(monkeypatch):
    def always_fails(*_args, **_kwargs):
        raise RuntimeError("candidate failure")

    monkeypatch.setattr(optimization_module, "cross_val_score", always_fails)
    with pytest.raises(OptimizationSearchError, match="zero successful"):
        _run_study(
            DecisionTreeClassifier(),
            _tree_space(),
            X,
            Y,
            inner_n_splits=2,
            n_trials=2,
            random_state=42,
        )


def test_structural_set_params_error_propagates():
    bad_space = ClassifierSearchSpace(
        "tree",
        common_parameters=(ParameterSpec("depth", "not_a_parameter", IntDomain(1, 2)),),
    )
    with pytest.raises(ValueError, match="Invalid parameter"):
        _run_study(
            DecisionTreeClassifier(),
            bad_space,
            X,
            Y,
            inner_n_splits=2,
            n_trials=1,
            random_state=42,
        )


class RefitFailureClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, value=1):
        self.value = value

    def fit(self, X, y):
        raise RuntimeError("refit failure")

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


def test_best_configuration_refit_failure_propagates(monkeypatch):
    monkeypatch.setattr(
        optimization_module,
        "cross_val_score",
        lambda *_args, **_kwargs: np.array([0.8, 0.8]),
    )
    space = ClassifierSearchSpace(
        "failure",
        common_parameters=(ParameterSpec("value", "value", IntDomain(1, 2)),),
    )

    with pytest.raises(RuntimeError, match="refit failure"):
        optimize_and_refit(
            RefitFailureClassifier(),
            space,
            X,
            Y,
            inner_n_splits=2,
            n_trials=1,
            random_state=42,
        )


def test_nonfinite_candidate_scores_are_failed_trials(monkeypatch):
    monkeypatch.setattr(
        optimization_module,
        "cross_val_score",
        lambda *_args, **_kwargs: np.array([np.nan, 0.8]),
    )
    with pytest.raises(OptimizationSearchError, match="zero successful"):
        _run_study(
            DecisionTreeClassifier(),
            _tree_space(),
            X,
            Y,
            inner_n_splits=2,
            n_trials=1,
            random_state=42,
        )


def test_pruned_trial_is_an_explicit_contract_failure(monkeypatch):
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.NopPruner(),
        direction="maximize",
    )
    study.add_trial(optuna.trial.create_trial(state=optuna.trial.TrialState.PRUNED))
    monkeypatch.setattr(
        optimization_module.optuna, "create_study", lambda **_kwargs: study
    )

    with pytest.raises(OptimizationSearchError, match="PRUNED trial"):
        _run_study(
            DecisionTreeClassifier(),
            _tree_space(),
            X,
            Y,
            inner_n_splits=2,
            n_trials=1,
            random_state=42,
        )


def test_optuna_logging_scope_sets_and_restores_on_success_and_error():
    original = optuna.logging.get_verbosity()
    alternate = logging.DEBUG
    optuna.logging.set_verbosity(alternate)
    try:
        with optuna_logging_scope(verbose=False):
            assert optuna.logging.get_verbosity() == optuna.logging.WARNING
        assert optuna.logging.get_verbosity() == alternate

        with pytest.raises(RuntimeError):
            with optuna_logging_scope(verbose=True):
                assert optuna.logging.get_verbosity() == optuna.logging.INFO
                raise RuntimeError("stop")
        assert optuna.logging.get_verbosity() == alternate
    finally:
        optuna.logging.set_verbosity(original)


def test_seeded_sequential_search_is_repeatable_beyond_startup_phase():
    """Use a deterministic tree to isolate seeded Optuna from estimator randomness."""
    kwargs = {
        "estimator": DecisionTreeClassifier(random_state=42),
        "search_space": _tree_space(),
        "X": X,
        "y": Y,
        "inner_n_splits": 2,
        "n_trials": OPTIMIZATION_POLICY.n_startup_trials + 1,
        "random_state": 42,
    }

    first_study, first_result = _run_study(**kwargs)
    second_study, second_result = _run_study(**kwargs)

    first_trace = [
        (trial.params, trial.state, trial.value) for trial in first_study.trials
    ]
    second_trace = [
        (trial.params, trial.state, trial.value) for trial in second_study.trials
    ]
    assert first_trace == second_trace
    assert first_result.best_params == second_result.best_params
    assert first_result.best_inner_f1_macro == pytest.approx(
        second_result.best_inner_f1_macro
    )


def test_optimization_result_is_data_only_and_immutable():
    _, result = _run_study(
        DecisionTreeClassifier(),
        _tree_space(),
        X,
        Y,
        inner_n_splits=2,
        n_trials=1,
        random_state=42,
    )

    assert not hasattr(result, "study")
    assert not hasattr(result, "estimator")
    with pytest.raises(FrozenInstanceError):
        setattr(result, "n_trials_complete", 99)


def test_only_optimization_module_imports_optuna_and_grid_search_is_absent():
    optuna_importers = []
    grid_search_references = []
    for path in (_REPOSITORY_ROOT / "melite").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(
                    alias.name == "optuna" or alias.name.startswith("optuna.")
                    for alias in node.names
                ):
                    optuna_importers.append(path.name)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                if node.module == "optuna" or node.module.startswith("optuna."):
                    optuna_importers.append(path.name)
            if isinstance(node, ast.Name) and node.id == "GridSearchCV":
                grid_search_references.append(path.name)

    assert set(optuna_importers) == {"optimization.py"}
    assert grid_search_references == []


def test_export_has_no_optimization_engine_coupling():
    export_path = _REPOSITORY_ROOT / "melite" / "export_best_model.py"
    tree = ast.parse(export_path.read_text(encoding="utf-8"))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "melite.optimization" not in imported_modules
    assert "optuna" not in imported_modules
    assert called_names.isdisjoint({"optimize_and_refit", "create_study"})


def test_public_package_facade_remains_unchanged():
    assert melite.__all__ == [
        "Config",
        "load_datasets",
        "plot_f1_macro_evidence",
        "predict",
        "__version__",
    ]
