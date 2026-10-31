# SPDX-License-Identifier: LGPL-3.0-or-later
"""Canonical scientific model identities known to MELITE.

Catalog membership describes a supervised-learning model concept. Runtime
availability remains the responsibility of MELITE's model builders and
validation; the conceptual ensemble branch is therefore not a separate
learning paradigm from supervised learning.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Immutable metadata for one conceptual supervised-learning model."""

    key: str
    public_name: str
    prediction_task: Literal["classification", "regression"]
    conceptual_branch: Literal["supervised_learning", "ensemble_learning"]
    strategy_or_group: str
    availability: Literal["implemented", "planned"]


MODEL_CATALOG: tuple[ModelMetadata, ...] = (
    ModelMetadata(
        "svc",
        "SVC",
        "classification",
        "supervised_learning",
        "support_vector_machine",
        "implemented",
    ),
    ModelMetadata(
        "lr",
        "LogisticRegression",
        "classification",
        "supervised_learning",
        "logistic_regression",
        "implemented",
    ),
    ModelMetadata(
        "gnb",
        "GaussianNB",
        "classification",
        "supervised_learning",
        "gaussian_naive_bayes",
        "implemented",
    ),
    ModelMetadata(
        "knn",
        "KNeighborsClassifier",
        "classification",
        "supervised_learning",
        "nearest_neighbors",
        "implemented",
    ),
    ModelMetadata(
        "dt",
        "DecisionTreeClassifier",
        "classification",
        "supervised_learning",
        "decision_tree",
        "implemented",
    ),
    ModelMetadata(
        "rf",
        "RandomForestClassifier",
        "classification",
        "ensemble_learning",
        "bagging_derived",
        "implemented",
    ),
    ModelMetadata(
        "ada",
        "AdaBoostClassifier",
        "classification",
        "ensemble_learning",
        "adaptive_boosting",
        "implemented",
    ),
    ModelMetadata(
        "xgb",
        "XGBClassifier",
        "classification",
        "ensemble_learning",
        "gradient_boosting",
        "implemented",
    ),
    ModelMetadata(
        "lgbm",
        "LGBMClassifier",
        "classification",
        "ensemble_learning",
        "gradient_boosting",
        "implemented",
    ),
    ModelMetadata(
        "catboost",
        "CatBoostClassifier",
        "classification",
        "ensemble_learning",
        "gradient_boosting",
        "implemented",
    ),
    ModelMetadata(
        "stack",
        "StackingClassifier",
        "classification",
        "ensemble_learning",
        "stacking",
        "implemented",
    ),
    ModelMetadata(
        "linear",
        "LinearRegression",
        "regression",
        "supervised_learning",
        "linear_regression",
        "planned",
    ),
    ModelMetadata(
        "poly",
        "PolynomialRegression",
        "regression",
        "supervised_learning",
        "polynomial_regression",
        "planned",
    ),
    ModelMetadata(
        "ridge",
        "Ridge",
        "regression",
        "supervised_learning",
        "ridge_lasso_regression_l2_regularization",
        "planned",
    ),
    ModelMetadata(
        "lasso",
        "Lasso",
        "regression",
        "supervised_learning",
        "ridge_lasso_regression_l1_regularization",
        "planned",
    ),
)
