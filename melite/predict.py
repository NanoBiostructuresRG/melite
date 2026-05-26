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
) -> dict:
    """Load a MELITE model artifact and run inference on new data.

    Parameters
    ----------
    model_path : str or pathlib.Path
        Path to a ``.pkl`` file produced by ``melite export``.
    X : numpy.ndarray
        Feature matrix of shape ``(n_samples, n_features)``. Must be a 2-D
        array and should use the same reduction method and level as the
        training data (e.g. PCA70 → 37 features).
    return_proba : bool, optional
        If ``True`` (default) and the loaded model exposes a
        ``predict_proba`` method, class probabilities are computed and
        included in the output. If ``False``, or if the model does not
        support probability estimates, ``probabilities`` is ``None``.

    Returns
    -------
    dict
        Dictionary with the following keys:

        - ``"predictions"`` : :class:`numpy.ndarray`, shape ``(n_samples,)``
          — predicted class labels.
        - ``"probabilities"`` : :class:`numpy.ndarray` or ``None``,
          shape ``(n_samples, n_classes)`` — class probability estimates,
          or ``None`` if not available or not requested.
        - ``"model_path"`` : str — resolved path to the loaded model file.
        - ``"n_samples"`` : int — number of samples in ``X``.

    Raises
    ------
    FileNotFoundError
        If *model_path* does not exist. The error message includes the path
        and a hint to run ``melite export`` first.
    ValueError
        If *X* is not a 2-D numpy array.

    Notes
    -----
    The ``.pkl`` artifacts produced by ``melite export`` are serialised with
    :func:`joblib.dump`. All scikit-learn compatible estimators (SVC,
    RandomForestClassifier, XGBClassifier) are supported.

    Examples
    --------
    Load a previously exported SVC model and predict on new data:

    >>> import numpy as np
    >>> from melite import predict
    >>> X_new = np.random.rand(10, 37).astype(np.float32)
    >>> result = predict("output/Model_SVC_PCA70.pkl", X_new)
    >>> result["predictions"].shape
    (10,)
    >>> result["probabilities"].shape
    (10, 2)
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
