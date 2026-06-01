# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for MELITE model training selection behavior."""

from types import SimpleNamespace

import numpy as np
import pytest

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

import melite.model_training as model_training
from melite.model_training import MultiModelTrainer


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
            "n_repeats": 1,
            "random_state": 42,
        },
    )


def _trainer_with_fake_training(active_models):
    trainer = MultiModelTrainer(_config(active_models))
    trainer.model_builders = {
        "svc": lambda: "svc-estimator",
        "rf": lambda: "rf-estimator",
        "xgb": lambda: "xgb-estimator",
        "stack": lambda: "stack-estimator",
    }
    calls = []

    def fake_grid_search(model, X_train, y_train, param_grid):
        model_name = model.split("-")[0]
        calls.append((model_name, param_grid))
        return f"{model_name}-tuned", {"model": model_name}

    def fake_cross_validate(model, X_train, y_train):
        model_name = model.split("-")[0]
        scores = {"svc": 0.7, "rf": 0.8, "xgb": 0.6}
        return scores[model_name], 0.01, 0.75, 0.02, 0.85, 0.03

    trainer.perform_grid_search = fake_grid_search
    trainer.cross_validate_model = fake_cross_validate
    return trainer, calls


def test_active_models_svc_trains_only_svc():
    trainer, calls = _trainer_with_fake_training(["svc"])

    result = trainer.train_and_select_best_model(
        np.ones((4, 2)),
        np.array([0, 1, 0, 1]),
        "PCA",
        70,
    )

    assert [model_name for model_name, _ in calls] == ["svc"]
    assert result[0] == "svc-tuned"
    assert result[1] == {"model": "svc"}


def test_active_models_rf_trains_only_rf():
    trainer, calls = _trainer_with_fake_training(["rf"])

    result = trainer.train_and_select_best_model(
        np.ones((4, 2)),
        np.array([0, 1, 0, 1]),
        "PCA",
        70,
    )

    assert [model_name for model_name, _ in calls] == ["rf"]
    assert result[0] == "rf-tuned"
    assert result[1] == {"model": "rf"}


def test_svc_builder_returns_scaler_then_svc_pipeline():
    trainer = MultiModelTrainer(_config(["svc"]))

    model = trainer.model_builders["svc"]()

    assert isinstance(model, SklearnPipeline)
    assert list(model.named_steps) == ["scaler", "svc"]
    assert isinstance(model.named_steps["scaler"], StandardScaler)
    assert isinstance(model.named_steps["svc"], SVC)
    assert model.named_steps["svc"].probability is True


def test_rf_and_xgb_builders_remain_unscaled_direct_estimators():
    trainer = MultiModelTrainer(_config(["rf", "xgb"]))

    rf = trainer.model_builders["rf"]()
    xgb = trainer.model_builders["xgb"]()

    assert isinstance(rf, RandomForestClassifier)
    assert isinstance(xgb, XGBClassifier)
    assert not isinstance(rf, SklearnPipeline)
    assert not isinstance(xgb, SklearnPipeline)


def test_stacking_builder_returns_experimental_stacking_classifier():
    trainer = MultiModelTrainer(_config(["stack"]))

    model = trainer.model_builders["stack"]()
    estimators = dict(model.estimators)
    svc = estimators["svc"]
    rf = estimators["rf"]
    xgb = estimators["xgb"]

    assert isinstance(model, StackingClassifier)
    assert model.stack_method == "predict_proba"
    assert model.passthrough is False
    assert isinstance(svc, SklearnPipeline)
    assert list(svc.named_steps) == ["scaler", "svc"]
    assert isinstance(svc.named_steps["scaler"], StandardScaler)
    assert isinstance(svc.named_steps["svc"], SVC)
    assert svc.named_steps["svc"].probability is True
    assert isinstance(rf, RandomForestClassifier)
    assert isinstance(xgb, XGBClassifier)
    assert not isinstance(rf, SklearnPipeline)
    assert not isinstance(xgb, SklearnPipeline)


def test_stacking_builder_reuses_repeated_stratified_cv_strategy():
    trainer = MultiModelTrainer(_config(["stack"]))

    model = trainer.model_builders["stack"]()

    assert isinstance(model.cv, RepeatedStratifiedKFold)
    assert model.cv.cvargs["n_splits"] == 2
    assert model.cv.n_repeats == 1
    assert model.cv.random_state == 42


def test_filter_param_grid_preserves_svc_pipeline_prefixes():
    trainer = MultiModelTrainer(_config(["svc"]))

    grid = trainer._filter_param_grid("svc")

    assert grid == [{"svc__C": [1], "svc__kernel": ["linear"]}]


def test_filter_param_grid_supports_minimal_stack_grid():
    trainer = MultiModelTrainer(_config(["stack"]))

    grid = trainer._filter_param_grid("stack")

    assert grid == [{}]


def test_grid_search_uses_f1_macro_scoring(monkeypatch):
    captured = {}

    class DummyGridSearchCV:
        def __init__(self, model, param_grid, scoring, cv, n_jobs):
            captured.update({
                "model": model,
                "param_grid": param_grid,
                "scoring": scoring,
                "cv": cv,
                "n_jobs": n_jobs,
            })
            self.best_estimator_ = "best-estimator"
            self.best_params_ = {"best": True}

        def fit(self, X_train, y_train):
            captured["fit_shape"] = X_train.shape

    monkeypatch.setattr(model_training, "GridSearchCV", DummyGridSearchCV)
    trainer = MultiModelTrainer(_config(["svc"]))

    estimator, params = trainer.perform_grid_search(
        "estimator",
        np.ones((4, 2)),
        np.array([0, 1, 0, 1]),
        [{"param": [1]}],
    )

    assert captured["scoring"] == "f1_macro"
    assert isinstance(captured["cv"], RepeatedStratifiedKFold)
    assert estimator == "best-estimator"
    assert params == {"best": True}


def test_invalid_active_model_raises_clear_error():
    with pytest.raises(ValueError, match="Unknown active model\\(s\\): knn"):
        MultiModelTrainer(_config(["svc", "knn"]))


def test_empty_active_models_raises_clear_error():
    with pytest.raises(ValueError, match="ACTIVE_MODELS must contain at least one"):
        MultiModelTrainer(_config([]))
