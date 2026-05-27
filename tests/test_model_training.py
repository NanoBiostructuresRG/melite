# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for MELITE model training selection behavior."""

from types import SimpleNamespace

import numpy as np
import pytest

from melite.model_training import MultiModelTrainer


def _config(active_models):
    return SimpleNamespace(
        ACTIVE_MODELS=active_models,
        RANDOM_STATE=42,
        PARAM_GRID=[
            {"model": ["svc"], "C": [1], "kernel": ["linear"]},
            {"model": ["rf"], "n_estimators": [10]},
            {"model": ["xgb"], "n_estimators": [10]},
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


def test_invalid_active_model_raises_clear_error():
    with pytest.raises(ValueError, match="Unknown active model\\(s\\): knn"):
        MultiModelTrainer(_config(["svc", "knn"]))


def test_empty_active_models_raises_clear_error():
    with pytest.raises(ValueError, match="ACTIVE_MODELS must contain at least one"):
        MultiModelTrainer(_config([]))
