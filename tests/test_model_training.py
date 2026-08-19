# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for MELITE nested model-training and selection behavior."""

from types import SimpleNamespace

import numpy as np
import pytest

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

import melite.model_training as model_training
from melite.model_training import MultiModelTrainer


X = np.ones((6, 2))
Y = np.array([0, 1, 0, 1, 0, 1])


def _config(active_models):
    return SimpleNamespace(
        ACTIVE_MODELS=active_models,
        RANDOM_STATE=42,
        PARAM_GRID=[
            {"model": ["svc"], "svc__C": [1], "svc__kernel": ["linear"]},
            {"model": ["rf"], "n_estimators": [10]},
            {"model": ["xgb"], "n_estimators": [10]},
            {"model": ["stack"]},
        ],
        get_cv_config=lambda: {
            "n_splits": 2,
            "n_repeats": 3,
            "inner_n_splits": 2,
            "random_state": 42,
        },
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
    }


def _family_evaluation(model_key, f1_macro, accuracy=0.8, auc_roc=0.9):
    return {
        "model_key": model_key,
        **_rich_evaluation(f1_macro, accuracy, auc_roc),
        "selected": False,
    }


def test_outer_cv_uses_configured_repeated_stratified_folds():
    cv = MultiModelTrainer(_config(["svc"]))._build_outer_cv()

    assert isinstance(cv, RepeatedStratifiedKFold)
    assert cv.cvargs["n_splits"] == 2
    assert cv.n_repeats == 3
    assert cv.random_state == 42


def test_inner_cv_uses_configured_shuffled_stratified_folds():
    cv = MultiModelTrainer(_config(["svc"]))._build_inner_cv()

    assert isinstance(cv, StratifiedKFold)
    assert cv.n_splits == 2
    assert cv.shuffle is True
    assert cv.random_state == 42


def test_grid_search_uses_f1_macro_and_inner_cv():
    trainer = MultiModelTrainer(_config(["svc"]))

    grid = trainer._build_grid_search(
        trainer.model_builders["svc"](),
        trainer._filter_param_grid("svc"),
    )

    assert isinstance(grid, GridSearchCV)
    assert grid.scoring == "f1_macro"
    assert grid.n_jobs == -1
    assert isinstance(grid.cv, StratifiedKFold)
    assert grid.cv.n_splits == 2


@pytest.mark.parametrize("model_name", ["svc", "rf", "xgb"])
def test_tunable_family_evaluation_wraps_fresh_model_in_grid_search(
    monkeypatch, model_name
):
    trainer = MultiModelTrainer(_config([model_name]))
    captured = {}
    sentinel_grid = object()

    def fake_build_grid_search(model, param_grid):
        captured["model"] = model
        captured["param_grid"] = param_grid
        return sentinel_grid

    def fake_cross_validate(model, X_train, y_train):
        captured["evaluated"] = model
        return _rich_evaluation(0.7)

    monkeypatch.setattr(trainer, "_build_grid_search", fake_build_grid_search)
    monkeypatch.setattr(
        trainer, "_cross_validate_model_with_scores", fake_cross_validate
    )

    evaluation = trainer._evaluate_model_family(model_name, X, Y)

    assert captured["evaluated"] is sentinel_grid
    assert captured["param_grid"] == trainer._filter_param_grid(model_name)
    assert evaluation == _family_evaluation(model_name, 0.7)


def test_cross_validate_evaluates_search_with_outer_cv_and_single_job(monkeypatch):
    captured = {}
    sentinel_search = object()

    def fake_cross_validate(estimator, X_train, y_train, **kwargs):
        captured.update(estimator=estimator, kwargs=kwargs)
        return {
            "test_f1": np.array([0.6, 0.8]),
            "test_acc": np.array([0.7, 0.9]),
            "test_auc": np.array([0.8, 1.0]),
        }

    monkeypatch.setattr(model_training, "cross_validate", fake_cross_validate)
    trainer = MultiModelTrainer(_config(["svc"]))

    metrics = trainer.cross_validate_model(sentinel_search, X, Y)

    assert captured["estimator"] is sentinel_search
    assert isinstance(captured["kwargs"]["cv"], RepeatedStratifiedKFold)
    assert captured["kwargs"]["n_jobs"] == 1
    assert captured["kwargs"]["return_train_score"] is False
    assert captured["kwargs"]["scoring"] == {
        "f1": "f1_macro",
        "acc": "accuracy",
        "auc": "roc_auc",
    }
    assert metrics == pytest.approx((0.7, 0.1, 0.8, 0.1, 0.9, 0.1))


def test_rich_cross_validation_preserves_raw_outer_scores_and_indexing(monkeypatch):
    calls = []
    raw_scores = {
        "test_f1": np.array([0.60, 0.70, 0.80, 0.90, 0.65, 0.75]),
        "test_acc": np.array([0.61, 0.71, 0.81, 0.91, 0.66, 0.76]),
        "test_auc": np.array([0.62, 0.72, 0.82, 0.92, 0.67, 0.77]),
    }

    def fake_cross_validate(*args, **kwargs):
        calls.append((args, kwargs))
        return raw_scores

    monkeypatch.setattr(model_training, "cross_validate", fake_cross_validate)
    trainer = MultiModelTrainer(_config(["svc"]))

    evaluation = trainer._cross_validate_model_with_scores(object(), X, Y)

    assert len(calls) == 1
    assert evaluation["f1_macro"] == pytest.approx(raw_scores["test_f1"].mean())
    assert evaluation["f1_std"] == pytest.approx(raw_scores["test_f1"].std())
    assert evaluation["accuracy"] == pytest.approx(raw_scores["test_acc"].mean())
    assert evaluation["acc_std"] == pytest.approx(raw_scores["test_acc"].std())
    assert evaluation["auc_roc"] == pytest.approx(raw_scores["test_auc"].mean())
    assert evaluation["auc_std"] == pytest.approx(raw_scores["test_auc"].std())
    assert evaluation["outer_scores"] == [
        {
            "outer_split": split,
            "outer_repeat": split // 2,
            "outer_fold": split % 2,
            "f1_macro": raw_scores["test_f1"][split],
            "accuracy": raw_scores["test_acc"][split],
            "auc_roc": raw_scores["test_auc"][split],
        }
        for split in range(6)
    ]


def test_selection_uses_outer_f1_and_runs_one_final_search_for_winner(monkeypatch):
    trainer = MultiModelTrainer(_config(["svc", "rf", "xgb"]))
    trainer.model_builders = {
        name: (lambda name=name: f"{name}-fresh")
        for name in ("svc", "rf", "xgb", "stack")
    }
    family_evaluations = {
        "svc": _family_evaluation("svc", 0.70, 0.71, 0.72),
        "rf": _family_evaluation("rf", 0.85, 0.81, 0.82),
        "xgb": _family_evaluation("xgb", 0.75, 0.91, 0.92),
    }
    evaluated = []
    final_searches = []

    def fake_evaluate(model_name, X_train, y_train):
        evaluated.append(model_name)
        return family_evaluations[model_name]

    def fake_final_search(model, X_train, y_train, param_grid):
        final_searches.append((model, param_grid, X_train, y_train))
        return "rf-final-estimator", {"n_estimators": 10}

    monkeypatch.setattr(trainer, "_evaluate_model_family", fake_evaluate)
    monkeypatch.setattr(trainer, "perform_grid_search", fake_final_search)

    result, evaluations = trainer.evaluate_and_select_models(X, Y, "PCA", 70)

    assert evaluated == ["svc", "rf", "xgb"]
    assert [item["model_key"] for item in evaluations] == ["svc", "rf", "xgb"]
    assert [item["selected"] for item in evaluations] == [False, True, False]
    assert len(final_searches) == 1
    assert final_searches[0][0] == "rf-fresh"
    assert np.shares_memory(final_searches[0][2], X)
    assert np.shares_memory(final_searches[0][3], Y)
    assert result == (
        "rf-final-estimator",
        {"n_estimators": 10},
        0.85, 0.01, 0.81, 0.02, 0.82, 0.03,
    )
    assert len(result) == 8


def test_exact_outer_f1_tie_selects_first_active_model(monkeypatch):
    trainer = MultiModelTrainer(_config(["svc", "rf"]))
    monkeypatch.setattr(
        trainer,
        "_evaluate_model_family",
        lambda name, X_train, y_train: _family_evaluation(name, 0.8),
    )
    final_searches = []

    def fake_final_search(model, X_train, y_train, param_grid):
        final_searches.append(model)
        return "svc-final", {"svc__C": 1}

    monkeypatch.setattr(trainer, "perform_grid_search", fake_final_search)

    result, evaluations = trainer.evaluate_and_select_models(X, Y, "PCA", 70)

    assert result[:2] == ("svc-final", {"svc__C": 1})
    assert [item["selected"] for item in evaluations] == [True, False]
    assert len(final_searches) == 1
    assert isinstance(final_searches[0], SklearnPipeline)


def test_stack_is_evaluated_directly_without_grid_search(monkeypatch):
    trainer = MultiModelTrainer(_config(["stack"]))
    stack = object()
    captured = {}
    trainer.model_builders["stack"] = lambda: stack

    def forbidden_grid_search(*args, **kwargs):
        pytest.fail("stack must not be wrapped in GridSearchCV")

    def fake_cross_validate(model, X_train, y_train):
        captured["model"] = model
        return _rich_evaluation(0.7)

    monkeypatch.setattr(trainer, "_build_grid_search", forbidden_grid_search)
    monkeypatch.setattr(
        trainer, "_cross_validate_model_with_scores", fake_cross_validate
    )

    trainer._evaluate_model_family("stack", X, Y)

    assert captured["model"] is stack


def test_stack_internal_cv_uses_configured_inner_stratified_folds():
    model = MultiModelTrainer(_config(["stack"])).model_builders["stack"]()

    assert isinstance(model.cv, StratifiedKFold)
    assert model.cv.n_splits == 2
    assert model.cv.shuffle is True
    assert model.cv.random_state == 42


def test_stack_winner_fits_fresh_estimator_on_all_data(monkeypatch):
    trainer = MultiModelTrainer(_config(["stack"]))
    instances = []

    class DummyStack:
        def __init__(self):
            self.fit_args = None

        def fit(self, X_train, y_train):
            self.fit_args = (X_train, y_train)
            return self

    def build_stack():
        instance = DummyStack()
        instances.append(instance)
        return instance

    trainer.model_builders["stack"] = build_stack
    monkeypatch.setattr(
        trainer,
        "_cross_validate_model_with_scores",
        lambda model, X_train, y_train: _rich_evaluation(0.8, 0.7, 0.9),
    )

    result = trainer.train_and_select_best_model(X, Y, "PCA", 70)

    assert len(instances) == 2
    assert result[0] is instances[1]
    assert result[0] is not instances[0]
    assert result[0].fit_args == (X, Y)
    assert result[1] == {}
    assert result[2:] == (0.8, 0.01, 0.7, 0.02, 0.9, 0.03)
    assert len(result) == 8


def test_active_model_filtering_remains_intact(monkeypatch):
    trainer = MultiModelTrainer(_config(["rf"]))
    evaluated = []
    monkeypatch.setattr(
        trainer,
        "_evaluate_model_family",
        lambda name, X_train, y_train: (
            evaluated.append(name) or _family_evaluation(name, 0.8, 0.7, 0.9)
        ),
    )
    monkeypatch.setattr(
        trainer,
        "perform_grid_search",
        lambda model, X_train, y_train, grid: ("rf-final", {"best": True}),
    )

    result = trainer.train_and_select_best_model(X, Y, "PCA", 70)

    assert evaluated == ["rf"]
    assert result[:2] == ("rf-final", {"best": True})
    assert len(result) == 8


def test_svc_builder_returns_scaler_then_svc_pipeline():
    model = MultiModelTrainer(_config(["svc"])).model_builders["svc"]()

    assert isinstance(model, SklearnPipeline)
    assert list(model.named_steps) == ["scaler", "svc"]
    assert isinstance(model.named_steps["scaler"], StandardScaler)
    assert isinstance(model.named_steps["svc"], SVC)
    assert model.named_steps["svc"].probability is False


def test_rf_and_xgb_builders_remain_unscaled_direct_estimators():
    trainer = MultiModelTrainer(_config(["rf", "xgb"]))

    rf = trainer.model_builders["rf"]()
    xgb = trainer.model_builders["xgb"]()

    assert isinstance(rf, RandomForestClassifier)
    assert isinstance(xgb, XGBClassifier)
    assert rf.n_jobs == 1
    assert xgb.n_jobs == 1


def test_stacking_builder_returns_expected_stacking_classifier():
    model = MultiModelTrainer(_config(["stack"])).model_builders["stack"]()
    estimators = dict(model.estimators)
    svc = estimators["svc"]
    rf = estimators["rf"]
    xgb = estimators["xgb"]

    assert isinstance(model, StackingClassifier)
    assert model.stack_method == "predict_proba"
    assert model.passthrough is False
    assert model.n_jobs == -1

    assert isinstance(svc, SklearnPipeline)
    assert list(svc.named_steps) == ["scaler", "svc"]
    assert isinstance(svc.named_steps["scaler"], StandardScaler)
    assert isinstance(svc.named_steps["svc"], SVC)
    assert svc.named_steps["svc"].probability is True

    assert isinstance(rf, RandomForestClassifier)
    assert rf.n_jobs == 1

    assert isinstance(xgb, XGBClassifier)
    assert xgb.n_jobs == 1

    assert not isinstance(rf, SklearnPipeline)
    assert not isinstance(xgb, SklearnPipeline)


def test_real_stacking_builder_fits_predicts_and_predicts_probabilities():
    X_stack, y_stack = make_classification(
        n_samples=40,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=2,
        weights=[0.5, 0.5],
        flip_y=0,
        random_state=42,
    )
    model = MultiModelTrainer(_config(["stack"])).model_builders["stack"]()
    model.set_params(rf__n_estimators=5, xgb__n_estimators=5)

    model.fit(X_stack, y_stack)
    predictions = model.predict(X_stack)
    probabilities = model.predict_proba(X_stack)

    classes, counts = np.unique(y_stack, return_counts=True)
    assert classes.tolist() == [0, 1]
    assert counts.min() >= 2
    assert predictions.shape == (40,)
    assert probabilities.shape == (40, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_filter_param_grid_preserves_svc_pipeline_prefixes():
    grid = MultiModelTrainer(_config(["svc"]))._filter_param_grid("svc")

    assert grid == [{"svc__C": [1], "svc__kernel": ["linear"]}]


def test_filter_param_grid_supports_minimal_stack_grid():
    grid = MultiModelTrainer(_config(["stack"]))._filter_param_grid("stack")

    assert grid == [{}]


def test_invalid_active_model_raises_clear_error():
    with pytest.raises(ValueError, match="Unknown active model\\(s\\): knn"):
        MultiModelTrainer(_config(["svc", "knn"]))


def test_empty_active_models_raises_clear_error():
    with pytest.raises(ValueError, match="ACTIVE_MODELS must contain at least one"):
        MultiModelTrainer(_config([]))
