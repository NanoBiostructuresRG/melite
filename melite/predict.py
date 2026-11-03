# SPDX-License-Identifier: LGPL-3.0-or-later
"""Inference module for exported MELITE classification and regression models.

This module provides :func:`predict`, which loads a ``.pkl`` model artifact
produced by ``melite export`` and runs inference on a new feature matrix.
Classification artifacts return class predictions and, when requested and
supported, class probabilities. Regression artifacts return continuous
predictions without probabilities.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import is_regressor

__all__ = ["predict"]

logger = logging.getLogger(__name__)


def predict(
    model_path: Path | str,
    X: np.ndarray,
    return_proba: bool = True,
) -> dict[str, Any]:
    """Run classification or regression inference with an exported model.

    Parameters
    ----------
    model_path : str or pathlib.Path
        Path to a ``.pkl`` file produced by ``melite export``.
    X : numpy.ndarray
        Numeric feature matrix of shape ``(n_samples, n_features)``. It must
        be two-dimensional and use the same feature representation, column
        order, and feature count expected by the exported model. MELITE does
        not validate feature names or semantic feature identity.
    return_proba : bool, optional
        Whether to request class probabilities for a classification model when
        the loaded model supports ``predict_proba``. Default is ``True``. If
        disabled or unsupported, or if the artifact is a regressor, the
        returned ``probabilities`` value is ``None``.

    Returns
    -------
    dict[str, Any]
        Dictionary with the following keys:

        - ``"predictions"`` : numpy.ndarray
          One-dimensional output with shape ``(n_samples,)``: predicted class
          labels for classification or continuous numeric predictions for
          regression.
        - ``"probabilities"`` : numpy.ndarray or None
          Classification probabilities with shape
          ``(n_samples, n_classes)``, or ``None`` when unavailable, not
          requested, or performing regression.
        - ``"model_path"`` : str
          Resolved path to the loaded model artifact.
        - ``"n_samples"`` : int
          Number of samples in ``X``.

    Raises
    ------
    FileNotFoundError
        If ``model_path`` does not exist.
    ValueError
        If ``X`` is not a two-dimensional NumPy array. For regression
        artifacts, also raised when ``X`` is not finite real numerical data or
        its feature count differs from ``n_features_in_`` exposed by the fitted
        model.

    Notes
    -----
    This function is intended for fitted model artifacts created by
    ``melite export``. Regression input validation checks array structure,
    numerical finiteness, and the fitted model's feature count when available;
    it cannot verify that columns have the same scientific meaning as during
    training.

    Examples
    --------
    >>> import numpy as np
    >>> from melite import predict
    >>> X_new = np.random.default_rng(42).random((4, 5)).astype(np.float32)
    >>> result = predict("output/Model_SVC_sample_tabular.pkl", X_new)
    >>> result["predictions"].shape == (4,)
    True
    >>> result["n_samples"]
    4
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}. "
            "Run 'melite export' first to generate a .pkl artifact."
        )

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError(
            f"X must be a 2-D numpy array, got shape {getattr(X, 'shape', type(X))}."
        )

    logger.info("Loading model from %s", model_path.resolve())
    model = joblib.load(model_path)
    regression_model = is_regressor(model)
    if regression_model:
        if not np.issubdtype(X.dtype, np.number) or np.issubdtype(
            X.dtype, np.complexfloating
        ):
            raise ValueError("Regression inference X must contain real numeric values.")
        if not np.all(np.isfinite(X)):
            raise ValueError("Regression inference X must contain only finite values.")
        expected_features = getattr(model, "n_features_in_", None)
        if expected_features is not None and X.shape[1] != expected_features:
            raise ValueError(
                "Regression inference feature count mismatch: "
                f"model expects {expected_features}, got {X.shape[1]}."
            )

    predictions = model.predict(X)
    logger.info("Predictions computed: shape=%s", predictions.shape)

    probabilities = None
    if return_proba and not regression_model and hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        logger.info("Probabilities computed: shape=%s", probabilities.shape)
    elif return_proba and not regression_model:
        logger.warning(
            "Model %s does not support predict_proba; probabilities set to None.",
            type(model).__name__,
        )

    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "model_path": str(model_path.resolve()),
        "n_samples": len(X),
    }
