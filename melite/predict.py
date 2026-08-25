# SPDX-License-Identifier: LGPL-3.0-or-later
"""Inference module for MELITE.

This module provides :func:`predict`, which loads a ``.pkl`` model artifact
produced by ``melite export`` and runs inference on a new feature matrix.
Both class predictions and class probabilities are returned when the model
supports them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

__all__ = ["predict"]

logger = logging.getLogger(__name__)


def predict(
    model_path: Path | str,
    X: np.ndarray,
    return_proba: bool = True,
) -> dict[str, Any]:
    """Run inference with a model artifact produced by ``melite export``.

    Parameters
    ----------
    model_path : str or pathlib.Path
        Path to a ``.pkl`` file produced by ``melite export``.
    X : numpy.ndarray
        Numeric feature matrix of shape ``(n_samples, n_features)``. It must
        use the same feature representation and number of features expected
        by the exported model, and it must be two-dimensional.
    return_proba : bool, optional
        Whether to request class probabilities when the loaded model supports
        ``predict_proba``. Default is ``True``. If disabled or unsupported,
        the returned ``probabilities`` value is ``None``.

    Returns
    -------
    dict[str, Any]
        Dictionary with the following keys:

        - ``"predictions"`` : numpy.ndarray
          Predicted class labels with shape ``(n_samples,)``.
        - ``"probabilities"`` : numpy.ndarray or None
          Class probabilities with shape ``(n_samples, n_classes)``, or ``None``
          when unavailable or not requested.
        - ``"model_path"`` : str
          Resolved path to the loaded model artifact.
        - ``"n_samples"`` : int
          Number of samples in ``X``.

    Raises
    ------
    FileNotFoundError
        If ``model_path`` does not exist.
    ValueError
        If ``X`` is not a two-dimensional NumPy array.

    Notes
    -----
    This function is intended for fitted model artifacts created by
    ``melite export``.

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

    predictions = model.predict(X)
    logger.info("Predictions computed: shape=%s", predictions.shape)

    probabilities = None
    if return_proba and hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        logger.info("Probabilities computed: shape=%s", probabilities.shape)
    elif return_proba:
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
