# SPDX-License-Identifier: LGPL-3.0-or-later
"""Dataset loading and label consistency validation for MELITE.

This module provides :func:`load_datasets` for the generalized dataset
registry.

If a ``.npz`` file contains an embedded ``y`` array, it is compared
element-wise against ``raw/labels.npy``. A :exc:`ValueError` is raised if
the two arrays do not match, preventing silent feature-label mismatches
from propagating into model training.
"""

import logging
import numpy as np
from pathlib import Path

__all__ = ["load_datasets"]

logger = logging.getLogger(__name__)


def _count_differences(left: np.ndarray, right: np.ndarray) -> int | str:
    return int(np.sum(left != right)) if left.shape == right.shape else "N/A"


def _load_one_dataset(dataset_id: str, spec: dict) -> dict:
    data_path = Path(spec["path"])
    label_path = Path(spec["label_path"])
    metadata = dict(spec.get("metadata", {}))

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset '{dataset_id}' file not found: {data_path}"
        )
    if not label_path.exists():
        raise FileNotFoundError(
            f"Dataset '{dataset_id}' label_path not found: {label_path}"
        )

    y = np.load(label_path)
    data = np.load(data_path)
    logger.info("Keys in %s: %s", data_path, data.files)

    if "X" not in data.files:
        raise ValueError(
            f"Required key 'X' not found in {data_path}.\n"
            f"           Available keys: {list(data.files)}"
        )

    X = data["X"]
    if X.ndim != 2:
        raise ValueError(
            f"Dataset '{dataset_id}' X must be 2D; got shape {X.shape}."
        )
    if not np.issubdtype(X.dtype, np.number):
        raise ValueError(
            f"Dataset '{dataset_id}' X must be numeric; got dtype {X.dtype}."
        )
    if len(y) != X.shape[0]:
        raise ValueError(
            f"Dataset '{dataset_id}' X/y length mismatch: "
            f"X has {X.shape[0]} rows, y has {len(y)} labels."
        )

    if "y" in data.files:
        y_from_file = data["y"]
        if not np.array_equal(y_from_file, y):
            n_diff = _count_differences(y_from_file, y)
            raise ValueError(
                f"Label mismatch in {data_path}:\n"
                f"           embedded y (shape={y_from_file.shape}) does not match\n"
                f"           {label_path} (shape={y.shape}).\n"
                f"           Differing elements: {n_diff}/"
                f"{y.shape[0] if y_from_file.shape == y.shape else '?'}."
            )

    return {"X": X, "y": y, "metadata": metadata}


def load_datasets(config) -> dict:
    """Load all datasets from ``config.DATASETS``.

    Returns
    -------
    dict
        Mapping of dataset id to dictionaries with ``X``, ``y``, and
        ``metadata`` keys. Dataset ids are user-defined identifiers and are
        not interpreted as method names.
    """
    loaded = {}
    for dataset_id, spec in config.DATASETS.items():
        loaded[dataset_id] = _load_one_dataset(dataset_id, spec)
        logger.info(
            "Loaded %s: X shape=%s, y shape=%s",
            dataset_id,
            loaded[dataset_id]["X"].shape,
            loaded[dataset_id]["y"].shape,
        )
    return loaded


def _load_dataset_legacy(config, reduction_type: str, levels: list) -> dict:
    """Load reduced feature matrices and labels for benchmarking.

    Reads ``raw/labels.npy`` as the authoritative label vector, then loads
    each ``{reduction_type}{level}.npz`` file from ``data/``. If a file
    contains an embedded ``y`` array, it is validated against
    ``raw/labels.npy``.

    Parameters
    ----------
    config : melite.config.Config
        MELITE configuration object. Must have ``PATHS["INPUT"]`` and
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
    >>> from melite import Config
    >>> cfg = Config()
    >>> cfg.setup()
    >>> from melite.load_dataset import _load_dataset_legacy
    >>> dataset = _load_dataset_legacy(cfg, "PCA", [70, 85])
    >>> X, y = dataset["PCA70"]
    >>> X.shape
    (182, 37)
    """
    reductions = {}
    loaded = 0

    for level in levels:
        dataset_id = f"{reduction_type}{level}"
        spec = {
            "path": Path(config.PATHS["DATASET"]) / f"{dataset_id}.npz",
            "label_path": Path(config.PATHS["INPUT"]) / "labels.npy",
            "metadata": {
                "family": "dimensionality",
                "method": reduction_type,
                "level": level,
            },
        }

        try:
            dataset = _load_one_dataset(dataset_id, spec)
            reductions[dataset_id] = (dataset["X"], dataset["y"])
            logger.info(
                "Loaded %s: X shape=%s, y shape=%s",
                dataset_id,
                dataset["X"].shape,
                dataset["y"].shape,
            )
            loaded += 1

        except ValueError:
            raise
        except FileNotFoundError as exc:
            logger.warning(
                "Expected file not found: %s\n"
                "           Place the feature matrix and labels at the configured paths and retry.",
                exc,
            )
        except Exception as exc:
            logger.error("Error loading %s: %s", dataset_id, exc)

    if loaded == 0:
        logger.warning(
            "No datasets loaded for %s with levels %s", reduction_type, levels
        )
    else:
        logger.info(
            "Loaded %d/%d datasets for %s", loaded, len(levels), reduction_type
        )

    return reductions
