# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for MELITE's canonical scientific model catalog."""

from melite.model_catalog import MODEL_CATALOG


def test_model_catalog_has_unique_identities_and_expected_availability():
    implemented = {
        model.key for model in MODEL_CATALOG if model.availability == "implemented"
    }
    planned = [
        model for model in MODEL_CATALOG if model.availability == "planned"
    ]

    assert len(MODEL_CATALOG) == 15
    assert len({model.key for model in MODEL_CATALOG}) == 15
    assert len({model.public_name for model in MODEL_CATALOG}) == 15
    assert implemented == {
        "svc",
        "lr",
        "gnb",
        "knn",
        "dt",
        "rf",
        "ada",
        "xgb",
        "lgbm",
        "catboost",
        "stack",
    }
    assert {model.key for model in planned} == {"linear", "poly", "ridge", "lasso"}
    assert all(model.prediction_task == "regression" for model in planned)
