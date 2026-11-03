# SPDX-License-Identifier: LGPL-3.0-or-later
"""Canonical regression estimator construction and numerical validation."""

from math import comb

import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

__all__ = [
    "MAX_POLYNOMIAL_FEATURES",
    "REGRESSION_MODEL_KEYS",
    "build_regression_model",
    "expected_polynomial_feature_count",
    "validate_polynomial_expansion",
    "validate_regression_inputs",
]

MAX_POLYNOMIAL_FEATURES = 10_000
REGRESSION_MODEL_KEYS = ("linear", "poly", "ridge", "lasso")
_POLYNOMIAL_DEGREE = 2


def expected_polynomial_feature_count(
    n_features: int, degree: int = _POLYNOMIAL_DEGREE
) -> int:
    """Return the number of polynomial terms through ``degree``, sans bias."""
    if n_features < 1:
        raise ValueError("n_features must be at least 1.")
    if degree < 1:
        raise ValueError("degree must be at least 1.")
    return comb(n_features + degree, degree) - 1


def validate_polynomial_expansion(n_features: int) -> int:
    """Reject a degree-two expansion beyond MELITE's feature limit."""
    expanded_features = expected_polynomial_feature_count(n_features)
    if expanded_features > MAX_POLYNOMIAL_FEATURES:
        raise ValueError(
            "PolynomialRegression expansion exceeds 10,000 features: "
            f"original dimensions={n_features}, "
            f"expanded dimensions={expanded_features}."
        )
    return expanded_features


def validate_regression_inputs(X, y) -> tuple[np.ndarray, np.ndarray]:
    """Validate MELITE's finite real-numerical regression input contract."""
    X_array = np.asarray(X)
    y_array = np.asarray(y)
    if X_array.ndim != 2:
        raise ValueError("Regression features must be a two-dimensional matrix.")
    if y_array.ndim != 1:
        raise ValueError("Regression targets must be a one-dimensional vector.")
    if X_array.shape[0] != y_array.shape[0]:
        raise ValueError("Regression features and targets must have equal row counts.")
    if X_array.shape[1] < 1:
        raise ValueError("Regression features must contain at least one column.")
    for name, values in (("features", X_array), ("targets", y_array)):
        if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
            values.dtype, np.complexfloating
        ):
            raise ValueError(f"Regression {name} must be real numeric values.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Regression {name} must contain only finite values.")
    return X_array, y_array


def build_regression_model(regressor_key: str):
    """Construct one fresh canonical MELITE regression estimator."""
    if regressor_key == "linear":
        return LinearRegression()
    if regressor_key == "poly":
        return SklearnPipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "poly",
                    PolynomialFeatures(
                        degree=_POLYNOMIAL_DEGREE,
                        include_bias=False,
                    ),
                ),
                ("linear", LinearRegression()),
            ]
        )
    if regressor_key == "ridge":
        return SklearnPipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge()),
            ]
        )
    if regressor_key == "lasso":
        return SklearnPipeline(
            [
                ("scaler", StandardScaler()),
                ("lasso", Lasso(max_iter=10_000)),
            ]
        )
    raise ValueError(f"Unsupported regressor key: {regressor_key}")
