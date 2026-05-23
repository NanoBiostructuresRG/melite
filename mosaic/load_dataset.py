# SPDX-License-Identifier: LGPL-3.0-or-later
"""Dataset loading and label consistency validation for MOSAIC.

This module provides a single public function, :func:`load_dataset`, that
reads pre-computed ``.npz`` feature matrices from ``data/`` and the
authoritative label vector from ``raw/labels.npy``.

If a ``.npz`` file contains an embedded ``y`` array, it is compared
element-wise against ``raw/labels.npy``. A :exc:`ValueError` is raised if
the two arrays do not match, preventing silent feature-label mismatches
from propagating into model training.
"""

import os
import logging
import numpy as np
from pathlib import Path

__all__ = ["load_dataset"]

logger = logging.getLogger(__name__)


def load_dataset(config, reduction_type: str, levels: list) -> dict:
    """Load reduced feature matrices and labels for benchmarking.

    Reads ``raw/labels.npy`` as the authoritative label vector, then loads
    each ``{reduction_type}{level}.npz`` file from ``data/``. If a file
    contains an embedded ``y`` array, it is validated against
    ``raw/labels.npy``.

    Parameters
    ----------
    config : mosaic.config.Config
        MOSAIC configuration object. Must have ``PATHS["INPUT"]`` and
        ``PATHS["DATASET"]`` set.
    reduction_type : str
        Dimensionality reduction method prefix, e.g. ``"PCA"`` or ``"UMAP"``.
    levels : list of int
        Variance retention levels to load, e.g. ``[70, 75, 80, 85, 90, 95]``.

    Returns
    -------
    dict
        Mapping from dataset key (e.g. ``"PCA70"``) to a tuple ``(X, y)``
        where ``X`` is the feature matrix and ``y`` is the label vector.
        Keys for files that are missing or fail validation are omitted.

    Raises
    ------
    ValueError
        If a ``.npz`` file does not contain the required ``"X"`` key, or if
        its embedded ``"y"`` array does not match ``raw/labels.npy``.

    Notes
    -----
    - Missing files produce a ``WARNING`` log and are skipped silently.
    - Label mismatch errors include both array shapes and the number of
      differing elements to aid debugging.

    Examples
    --------
    >>> from mosaic import Config, load_dataset
    >>> cfg = Config()
    >>> cfg.setup()
    >>> dataset = load_dataset(cfg, "PCA", [70, 85])
    >>> X, y = dataset["PCA70"]
    >>> X.shape
    (182, 37)
    """
    try:
        labels_path = os.path.join(config.PATHS["INPUT"], "labels.npy")
        y = np.load(labels_path)
        logger.info("Labels loaded: %s (shape=%s)", labels_path, y.shape)
    except Exception as exc:
        logger.error("Error loading labels '%s': %s", labels_path, exc)
        return {}

    reductions = {}
    loaded = 0

    for level in levels:
        data_file = f"{reduction_type}{level}.npz"
        data_path = os.path.join(config.PATHS["DATASET"], data_file)

        try:
            if not os.path.exists(data_path):
                logger.warning(
                    "Expected file not found: %s\n"
                    "           Place the reduced feature matrix at this path and retry.",
                    data_path,
                )
                continue

            data = np.load(data_path)
            logger.info("Keys in %s: %s", data_file, data.files)

            if "X" not in data.files:
                raise ValueError(
                    f"Required key 'X' not found in {data_path}.\n"
                    f"           Available keys: {list(data.files)}"
                )

            X = data["X"]

            if "y" in data.files:
                y_from_file = data["y"]
                if not np.array_equal(y_from_file, y):
                    n_diff = (
                        int(np.sum(y_from_file != y))
                        if y_from_file.shape == y.shape
                        else "N/A"
                    )
                    raise ValueError(
                        f"Label mismatch in {data_path}:\n"
                        f"           embedded y (shape={y_from_file.shape}) does not match\n"
                        f"           {labels_path} (shape={y.shape}).\n"
                        f"           Differing elements: {n_diff}/"
                        f"{y.shape[0] if y_from_file.shape == y.shape else '?'}."
                    )

            reductions[f"{reduction_type}{level}"] = (X, y)
            logger.info(
                "Loaded %s: X shape=%s, y shape=%s", data_file, X.shape, y.shape
            )
            loaded += 1

        except ValueError:
            raise
        except Exception as exc:
            logger.error("Error loading %s: %s", data_file, exc)

    if loaded == 0:
        logger.warning(
            "No datasets loaded for %s with levels %s", reduction_type, levels
        )
    else:
        logger.info(
            "Loaded %d/%d datasets for %s", loaded, len(levels), reduction_type
        )

    return reductions
