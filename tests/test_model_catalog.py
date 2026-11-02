# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for MELITE's canonical scientific model catalog."""

from melite.model_catalog import MODEL_CATALOG


def test_model_catalog_has_unique_identities_and_expected_availability():
    implemented = [
        model for model in MODEL_CATALOG if model.availability == "implemented"
    ]
    implemented_regressors = {
        model.key for model in implemented if model.prediction_task == "regression"
    }
    planned = {model.key for model in MODEL_CATALOG if model.availability == "planned"}

    assert len(MODEL_CATALOG) == 15
    assert len({model.key for model in MODEL_CATALOG}) == 15
    assert len({model.public_name for model in MODEL_CATALOG}) == 15
    assert len(implemented) == 15
    assert planned == set()
    assert implemented_regressors == {"linear", "poly", "ridge", "lasso"}


def test_regression_catalog_identities_are_implemented_and_unchanged():
    regressors = {
        model.key: (
            model.public_name,
            model.prediction_task,
            model.conceptual_branch,
            model.strategy_or_group,
            model.availability,
        )
        for model in MODEL_CATALOG
        if model.prediction_task == "regression"
    }

    assert regressors == {
        "linear": (
            "LinearRegression",
            "regression",
            "supervised_learning",
            "linear_regression",
            "implemented",
        ),
        "poly": (
            "PolynomialRegression",
            "regression",
            "supervised_learning",
            "polynomial_regression",
            "implemented",
        ),
        "ridge": (
            "Ridge",
            "regression",
            "supervised_learning",
            "ridge_lasso_regression_l2_regularization",
            "implemented",
        ),
        "lasso": (
            "Lasso",
            "regression",
            "supervised_learning",
            "ridge_lasso_regression_l1_regularization",
            "implemented",
        ),
    }


def test_classification_catalog_metadata_remains_unchanged():
    classifiers = {
        model.key: (
            model.public_name,
            model.prediction_task,
            model.conceptual_branch,
            model.strategy_or_group,
            model.availability,
        )
        for model in MODEL_CATALOG
        if model.prediction_task == "classification"
    }

    assert classifiers == {
        "svc": (
            "SVC",
            "classification",
            "supervised_learning",
            "support_vector_machine",
            "implemented",
        ),
        "lr": (
            "LogisticRegression",
            "classification",
            "supervised_learning",
            "logistic_regression",
            "implemented",
        ),
        "gnb": (
            "GaussianNB",
            "classification",
            "supervised_learning",
            "gaussian_naive_bayes",
            "implemented",
        ),
        "knn": (
            "KNeighborsClassifier",
            "classification",
            "supervised_learning",
            "nearest_neighbors",
            "implemented",
        ),
        "dt": (
            "DecisionTreeClassifier",
            "classification",
            "supervised_learning",
            "decision_tree",
            "implemented",
        ),
        "rf": (
            "RandomForestClassifier",
            "classification",
            "ensemble_learning",
            "bagging_derived",
            "implemented",
        ),
        "ada": (
            "AdaBoostClassifier",
            "classification",
            "ensemble_learning",
            "adaptive_boosting",
            "implemented",
        ),
        "xgb": (
            "XGBClassifier",
            "classification",
            "ensemble_learning",
            "gradient_boosting",
            "implemented",
        ),
        "lgbm": (
            "LGBMClassifier",
            "classification",
            "ensemble_learning",
            "gradient_boosting",
            "implemented",
        ),
        "catboost": (
            "CatBoostClassifier",
            "classification",
            "ensemble_learning",
            "gradient_boosting",
            "implemented",
        ),
        "stack": (
            "StackingClassifier",
            "classification",
            "ensemble_learning",
            "stacking",
            "implemented",
        ),
    }
