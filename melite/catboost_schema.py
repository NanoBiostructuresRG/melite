# SPDX-License-Identifier: LGPL-3.0-or-later
"""Persisted feature-schema helpers for categorical CatBoost models."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

CATBOOST_SCHEMA_METADATA_KEY = "melite_feature_schema"


def _is_numeric_dtype(dtype: Any) -> bool:
    try:
        return bool(np.issubdtype(dtype, np.number))
    except TypeError:
        return False


def dataframe_feature_issues(
    features: pd.DataFrame,
    categorical_features: list[str] | tuple[str, ...],
) -> tuple[list[str], list[str], list[str]]:
    """Return missing categories, invalid categories, and non-numeric predictors."""
    missing_categorical = [
        column for column in categorical_features if column not in features.columns
    ]
    invalid_categorical = []
    for column in categorical_features:
        if column not in features.columns:
            continue
        values = features[column]
        if values.isna().any() or any(
            not isinstance(value, str) or not value.strip() for value in values
        ):
            invalid_categorical.append(column)

    non_numeric_columns = [
        str(column)
        for column, dtype in features.dtypes.items()
        if column not in categorical_features and not _is_numeric_dtype(dtype)
    ]
    return missing_categorical, invalid_categorical, non_numeric_columns


def build_catboost_feature_schema(
    features: pd.DataFrame,
    categorical_features: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Build the schema stored inside an exported categorical CatBoost model."""
    return {
        "columns": [str(column) for column in features.columns],
        "categorical_features": list(categorical_features),
        "dtypes": [str(dtype) for dtype in features.dtypes],
    }


def encode_catboost_feature_schema(schema: dict[str, Any]) -> str:
    """Encode a feature schema for CatBoost model metadata."""
    return json.dumps(schema, separators=(",", ":"), ensure_ascii=False)


def read_catboost_feature_schema(model: Any) -> dict[str, Any] | None:
    """Read MELITE schema metadata from a fitted CatBoost model, when present."""
    get_metadata = getattr(model, "get_metadata", None)
    if not callable(get_metadata):
        return None
    encoded = dict(get_metadata()).get(CATBOOST_SCHEMA_METADATA_KEY)
    if encoded is None:
        return None
    schema = json.loads(encoded)
    if not isinstance(schema, dict):
        raise ValueError("Exported CatBoost feature schema must be a JSON object.")
    return schema
