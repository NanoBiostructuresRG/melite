# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for MELITE nested classifier optimization and selection behavior."""

from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

import melite.model_training as model_training
from melite.model_training import MultiModelTrainer
from melite.optimization import OptimizationResult, OptunaSearchClassifier
from melite.search_spaces import get_search_space


X = np.ones((6, 2))
Y = np.array([0, 1, 0, 1, 0, 1])


def _config(active_classifiers, random_state=42, n_trials=5):
    return SimpleNamespace(
        ACTIVE_CLASSIFIERS=active_classifiers,
        RANDOM_STATE=random_state,
        N_TRIALS=n_trials,
        CV_CONFIG={"n_splits": 2, "n_repeats": 3, "inner_n_splits": 2},
    )


def _rich_evaluation(f1_macro, accuracy=0.8, auc_roc=0.9):
    return {
        "f1_macro": f1_macro,
        "f1_std": 0.01,
        "accuracy": accuracy,
        "acc_std": 0.02,
        "auc_roc": auc_roc,
        "auc_std": 0.03,
        "outer_scores": [],
        "optimization_searches": [],
    }


def _classifier_evaluation(classifier_key, f1_macro, accuracy=0.8, auc_roc=0.9):
    return {
        "classifier_key": classifier_key,
        **_rich_evaluation(f1_macro, accuracy, auc_roc),
        "selected": False,
    }


def _optimization_result(params=None):
    return OptimizationResult(
        best_params=params or {"n_estimators": 10},
        best_inner_f1_macro=0.75,
        n_trials_requested=5,
        n_trials_complete=5,
        n_trials_failed=0,
    )


def test_outer_cv_uses_configured_repeated_stratified_folds():
    cv = MultiModelTrainer(_config(["svc"]))._build_outer_cv()
    assert isinstance(cv, RepeatedStratifiedKFold)
    assert cv.cvargs["n_splits"] == 2
    assert cv.n_repeats == 3
    assert cv.random_state == 42


def test_canonical_seed_propagates_to_classifiers_and_stacking():
    trainer = MultiModelTrainer(_config(["svc", "stack"], random_state=17))
    stack = trainer.model_builders["stack"]()
    stack_estimators = dict(stack.estimators)
    assert trainer._build_outer_cv().random_state == 17
    assert trainer.model_builders["svc"]().named_steps["svc"].random_state == 17
    assert trainer.model_builders["rf"]().random_state == 17
    assert trainer.model_builders["xgb"]().random_state == 17
    assert stack.cv.random_state == 17
    assert stack_estimators["svc"].named_steps["svc"].random_state == 17
    assert stack_estimators["rf"].random_state == 17
    assert stack_estimators["xgb"].random_state == 17
    assert stack.final_estimator.random_state == 17


def test_tunable_classifier_is_wrapped_from_search_space_presence(monkeypatch):
    trainer = MultiModelTrainer(_config(["svc"], random_state=17, n_trials=9))
    captured = {}

    def fake_cross_validate(estimator, *_args):
        captured["estimator"] = estimator
        return _rich_evaluation(0.8)

    monkeypatch.setattr(
        trainer, "_cross_validate_model_with_scores", fake_cross_validate
    )
    trainer._evaluate_classifier("svc", X, Y)
    wrapper = captured["estimator"]
    assert isinstance(wrapper, OptunaSearchClassifier)
    assert wrapper.search_space == get_search_space("svc")
    assert wrapper.inner_n_splits == 2
    assert wrapper.n_trials == 9
    assert wrapper.random_state == 17


def test_non_tunable_classifier_is_evaluated_directly_without_optuna(monkeypatch):
    trainer = MultiModelTrainer(_config(["stack"]))
    captured = {}

    def fake_cross_validate(estimator, *_args):
        captured["estimator"] = estimator
        return _rich_evaluation(0.8)

    monkeypatch.setattr(
        trainer, "_cross_validate_model_with_scores", fake_cross_validate
    )
    trainer._evaluate_classifier("stack", X, Y)
    assert isinstance(captured["estimator"], StackingClassifier)
    assert not isinstance(captured["estimator"], OptunaSearchClassifier)


def test_search_decision_uses_space_presence_not_classifier_name(monkeypatch):
    trainer = MultiModelTrainer(_config(["svc"]))
    captured = {}
    monkeypatch.setattr(model_training, "get_search_space", lambda _name: None)

    def fake_cross_validate(estimator, *_args):
        captured["estimator"] = estimator
        return _rich_evaluation(0.8)

    monkeypatch.setattr(
        trainer, "_cross_validate_model_with_scores", fake_cross_validate
    )
    trainer._evaluate_classifier("svc", X, Y)
    assert isinstance(captured["estimator"], SklearnPipeline)


def test_direct_outer_cross_validate_preserves_historical_kwargs(monkeypatch):
    trainer = MultiModelTrainer(_config(["stack"]))
    captured = {}

    def fake_cross_validate(*args, **kwargs):
        captured.update(kwargs)
        return {
            "test_f1": np.array([0.7, 0.9]),
            "test_acc": np.array([0.6, 0.8]),
            "test_auc": np.array([0.75, 0.85]),
        }

    monkeypatch.setattr(model_training, "cross_validate", fake_cross_validate)
    result = trainer._cross_validate_model_with_scores(object(), X, Y)
    assert captured["scoring"] == {
        "f1": "f1_macro",
        "acc": "accuracy",
        "auc": "roc_auc",
    }
    assert captured["n_jobs"] == 1
    assert captured["return_train_score"] is False
    assert "error_score" not in captured
    assert "return_estimator" not in captured
    assert result["optimization_searches"] == []


def test_tunable_outer_cross_validate_raises_errors_and_extracts_level2(monkeypatch):
    trainer = MultiModelTrainer(_config(["svc"]))
    wrapper = OptunaSearchClassifier(
        trainer.model_builders["svc"](), get_search_space("svc"), 2, 5, 42
    )
    captured = {}
    fitted = [
        SimpleNamespace(optimization_result_=_optimization_result({"svc__C": value}))
        for value in (1.0, 2.0)
    ]

    def fake_cross_validate(*args, **kwargs):
        captured.update(kwargs)
        return {
            "test_f1": np.array([0.7, 0.9]),
            "test_acc": np.array([0.6, 0.8]),
            "test_auc": np.array([0.75, 0.85]),
            "estimator": fitted,
        }

    monkeypatch.setattr(model_training, "cross_validate", fake_cross_validate)
    result = trainer._cross_validate_model_with_scores(wrapper, X, Y)
    assert captured["error_score"] == "raise"
    assert captured["return_estimator"] is True
    assert [item["outer_split"] for item in result["optimization_searches"]] == [0, 1]
    assert [item["outer_fold"] for item in result["optimization_searches"]] == [0, 1]
    assert result["optimization_searches"][0]["best_params"] == {"svc__C": 1.0}
    assert all(not hasattr(item, "fit") for item in result["optimization_searches"])


def test_outer_metric_aggregation_and_split_indexing_are_unchanged(monkeypatch):
    trainer = MultiModelTrainer(_config(["stack"]))
    monkeypatch.setattr(
        model_training,
        "cross_validate",
        lambda *_args, **_kwargs: {
            "test_f1": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            "test_acc": np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            "test_auc": np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
        },
    )
    result = trainer._cross_validate_model_with_scores(object(), X, Y)
    assert result["f1_macro"] == np.mean([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert result["f1_std"] == np.std([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert result["accuracy"] == np.mean([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    assert result["auc_roc"] == np.mean([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    assert result["outer_scores"][5]["outer_split"] == 5
    assert result["outer_scores"][5]["outer_repeat"] == 2
    assert result["outer_scores"][5]["outer_fold"] == 1


def test_cross_validate_model_keeps_six_element_public_result(monkeypatch):
    trainer = MultiModelTrainer(_config(["stack"]))
    monkeypatch.setattr(
        trainer,
        "_cross_validate_model_with_scores",
        lambda *_args: _rich_evaluation(0.8, 0.7, 0.9),
    )
    assert trainer.cross_validate_model(object(), X, Y) == (
        0.8,
        0.01,
        0.7,
        0.02,
        0.9,
        0.03,
    )


def test_all_active_families_remain_ordered_and_exact_tie_selects_first(monkeypatch):
    trainer = MultiModelTrainer(_config(["svc", "rf"]))
    monkeypatch.setattr(
        trainer,
        "_evaluate_classifier",
        lambda name, *_args: _classifier_evaluation(name, 0.8),
    )
    monkeypatch.setattr(
        model_training,
        "optimize_and_refit",
        lambda model, *_args, **_kwargs: (
            model,
            _optimization_result({"svc__C": 1.0, "svc__kernel": "linear"}),
        ),
    )
    selected, evaluations = trainer.evaluate_and_select_models(X, Y, "dataset", None)
    assert [item["classifier_key"] for item in evaluations] == ["svc", "rf"]
    assert [item["selected"] for item in evaluations] == [True, False]
    assert isinstance(selected[0], SklearnPipeline)
    assert not isinstance(selected[0], OptunaSearchClassifier)
    assert len(selected) == 8


def test_final_tunable_winner_runs_one_fresh_search_and_uses_effective_params(
    monkeypatch,
):
    trainer = MultiModelTrainer(_config(["rf"]))
    monkeypatch.setattr(
        trainer,
        "_evaluate_classifier",
        lambda name, *_args: _classifier_evaluation(name, 0.8),
    )
    calls = []
    fitted = RandomForestClassifier(n_estimators=10)
    result = _optimization_result({"n_estimators": 10, "max_depth": None})

    def fake_optimize(model, space, X_train, y_train, **kwargs):
        calls.append((model, space, X_train, y_train, kwargs))
        return fitted, result

    monkeypatch.setattr(model_training, "optimize_and_refit", fake_optimize)
    selected, evaluations = trainer.evaluate_and_select_models(X, Y, "dataset", None)
    assert len(calls) == 1
    assert calls[0][1] == get_search_space("rf")
    assert calls[0][4] == {
        "inner_n_splits": 2,
        "n_trials": 5,
        "random_state": 42,
    }
    assert selected[0] is fitted
    assert selected[1] == {"n_estimators": 10, "max_depth": None}
    assert evaluations[0]["final_optimization_search"] is result
    assert evaluations[0]["final_optimization_search"].best_params == selected[1]


def test_final_non_tunable_winner_direct_fits_once_without_search(monkeypatch):
    trainer = MultiModelTrainer(_config(["stack"]))
    monkeypatch.setattr(
        trainer,
        "_evaluate_classifier",
        lambda name, *_args: _classifier_evaluation(name, 0.8),
    )
    instances = []

    class DummyStack:
        def __init__(self):
            self.fit_calls = 0

        def fit(self, X_train, y_train):
            self.fit_calls += 1
            return self

    def build_stack():
        instance = DummyStack()
        instances.append(instance)
        return instance

    trainer.model_builders["stack"] = build_stack
    monkeypatch.setattr(
        model_training,
        "optimize_and_refit",
        lambda *_args, **_kwargs: pytest.fail("non-tunable classifier must not search"),
    )
    selected, evaluations = trainer.evaluate_and_select_models(X, Y, "dataset", None)
    assert len(instances) == 1
    assert instances[0].fit_calls == 1
    assert selected[:2] == (instances[0], {})
    assert evaluations[0]["final_optimization_search"] is None


def test_train_and_select_best_model_still_returns_exactly_eight_elements(monkeypatch):
    trainer = MultiModelTrainer(_config(["stack"]))
    expected = (object(), {}, 0.8, 0.1, 0.7, 0.1, 0.9, 0.1)
    monkeypatch.setattr(
        trainer, "evaluate_and_select_models", lambda *_args: (expected, [])
    )
    assert trainer.train_and_select_best_model(X, Y, "dataset", None) is expected
    assert len(expected) == 8


def test_svc_builder_preserves_scaler_and_decision_function_auc_path():
    model = MultiModelTrainer(_config(["svc"])).model_builders["svc"]()
    assert isinstance(model, SklearnPipeline)
    assert list(model.named_steps) == ["scaler", "svc"]
    assert isinstance(model.named_steps["scaler"], StandardScaler)
    assert isinstance(model.named_steps["svc"], SVC)
    assert model.named_steps["svc"].probability is False
    assert hasattr(model, "decision_function")
    assert not hasattr(model, "predict_proba")


def test_real_tunable_svc_outer_auc_uses_decision_function():
    X_svc, y_svc = make_classification(
        n_samples=24,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )
    config = _config(["svc"], n_trials=1)
    config.CV_CONFIG["n_repeats"] = 1
    trainer = MultiModelTrainer(config)

    evaluation = trainer._evaluate_classifier("svc", X_svc, y_svc)

    assert np.isfinite(evaluation["auc_roc"])
    assert len(evaluation["optimization_searches"]) == 2


def test_rf_and_xgb_builders_remain_sequential_direct_estimators():
    trainer = MultiModelTrainer(_config(["rf", "xgb"]))
    rf = trainer.model_builders["rf"]()
    xgb = trainer.model_builders["xgb"]()
    assert isinstance(rf, RandomForestClassifier)
    assert isinstance(xgb, XGBClassifier)
    assert rf.n_jobs == 1
    assert xgb.n_jobs == 1


def test_stacking_builder_contract_remains_unchanged():
    model = MultiModelTrainer(_config(["stack"])).model_builders["stack"]()
    estimators = dict(model.estimators)
    assert isinstance(model, StackingClassifier)
    assert isinstance(model.cv, StratifiedKFold)
    assert model.cv.n_splits == 2
    assert model.cv.shuffle is True
    assert model.cv.random_state == 42
    assert model.stack_method == "predict_proba"
    assert model.passthrough is False
    assert model.n_jobs == -1
    assert isinstance(estimators["svc"], SklearnPipeline)
    assert estimators["svc"].named_steps["svc"].probability is True
    assert isinstance(estimators["rf"], RandomForestClassifier)
    assert isinstance(estimators["xgb"], XGBClassifier)


def test_real_stacking_builder_fits_predicts_and_predicts_probabilities():
    X_stack, y_stack = make_classification(
        n_samples=40,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        weights=[0.5, 0.5],
        random_state=42,
    )
    model = MultiModelTrainer(_config(["stack"])).model_builders["stack"]()
    model.set_params(rf__n_estimators=5, xgb__n_estimators=5)
    model.fit(X_stack, y_stack)
    assert model.predict(X_stack).shape == (40,)
    probabilities = model.predict_proba(X_stack)
    assert probabilities.shape == (40, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_legacy_grid_search_members_are_absent():
    trainer = MultiModelTrainer(_config(["svc"]))
    assert not hasattr(trainer, "_filter_param_grid")
    assert not hasattr(trainer, "_build_grid_search")
    assert not hasattr(trainer, "perform_grid_search")


def test_invalid_and_empty_active_classifier_lists_raise_clear_errors():
    with pytest.raises(ValueError, match="Unknown active classifier\\(s\\): knn"):
        MultiModelTrainer(_config(["svc", "knn"]))
    with pytest.raises(ValueError, match="must contain at least one"):
        MultiModelTrainer(_config([]))
