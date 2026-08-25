# SPDX-License-Identifier: LGPL-3.0-or-later
"""Dataset loading and label consistency validation for MELITE.

The normalized dataset specification's ``label_path`` supplies the
authoritative labels. If a dataset archive contains an embedded ``y``, MELITE
verifies it against that authoritative label vector and fails explicitly on a
mismatch before classifier evaluation.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config

__all__ = ["load_datasets"]

logger = logging.getLogger(__name__)


def _count_differences(left: np.ndarray, right: np.ndarray) -> int:
    return int(np.sum(left != right))


def _load_one_dataset(dataset_id: str, spec: dict) -> dict:
    data_path = Path(spec["path"])
    label_path = Path(spec["label_path"])
    metadata = dict(spec.get("metadata", {}))

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset '{dataset_id}' file not found: {data_path}")
    if not label_path.exists():
        raise FileNotFoundError(
            f"Dataset '{dataset_id}' label_path not found: {label_path}"
        )

    y = np.load(label_path)
    if y.ndim != 1:
        raise ValueError(
            f"Dataset '{dataset_id}' authoritative y must be 1D; got shape {y.shape}."
        )

    data = np.load(data_path)
    logger.info("Keys in %s: %s", data_path, data.files)

    if "X" not in data.files:
        raise ValueError(
            f"Required key 'X' not found in {data_path}.\n"
            f"           Available keys: {list(data.files)}"
        )

    X = data["X"]
    if X.ndim != 2:
        raise ValueError(f"Dataset '{dataset_id}' X must be 2D; got shape {X.shape}.")
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
        if y_from_file.ndim != 1:
            raise ValueError(
                f"Dataset '{dataset_id}' embedded y must be 1D; "
                f"got shape {y_from_file.shape}."
            )
        if y_from_file.shape != y.shape:
            raise ValueError(
                f"Dataset '{dataset_id}' embedded y shape "
                f"{y_from_file.shape} does not match authoritative y shape "
                f"{y.shape}."
            )
        if not np.array_equal(y_from_file, y):
            n_diff = _count_differences(y_from_file, y)
            raise ValueError(
                f"Label mismatch in {data_path}:\n"
                f"           embedded y (shape={y_from_file.shape}) does not match\n"
                f"           {label_path} (shape={y.shape}).\n"
                f"           Differing elements: {n_diff}/{y.shape[0]}."
            )

    return {"X": X, "y": y, "metadata": metadata}


def load_datasets(config: Config) -> dict[str, dict[str, Any]]:
    """Load every normalized dataset in a MELITE configuration.

    Parameters
    ----------
    config : melite.config.Config
        Normalized MELITE configuration containing ``DATASETS``.

    Returns
    -------
    dict[str, dict[str, Any]]
        Dictionary keyed by user-defined dataset id. Each dataset value is a
        dictionary containing:

        - ``"X"`` : numpy.ndarray
          Two-dimensional numeric feature matrix.
        - ``"y"`` : numpy.ndarray
          One-dimensional authoritative label vector loaded from
          ``label_path``.
        - ``"metadata"`` : dict
          Shallow copy of the dataset metadata dictionary. Metadata keys are
          transported but not interpreted by ``load_datasets``; nested mutable
          values are not deep-copied.

        Dataset ids are not interpreted as method, representation, reduction,
        or classifier names. The configured ``label_path`` is authoritative;
        an embedded ``y`` in the dataset archive is used only as a consistency
        check.

    Raises
    ------
    FileNotFoundError
        If a configured dataset file or authoritative label file does not
        exist.
    ValueError
        If the dataset archive lacks ``X``; if ``X`` is not two-dimensional or
        numeric; if authoritative ``y`` is not one-dimensional; if the row
        count of ``X`` differs from the authoritative label count; or if an
        embedded ``y`` is not one-dimensional, has a different shape, or has
        different values from authoritative ``y``.

    Examples
    --------
    >>> import numpy as np
    >>> from pathlib import Path
    >>> from tempfile import TemporaryDirectory
    >>> from melite import Config, load_datasets
    >>> with TemporaryDirectory() as temporary_directory:
    ...     root = Path(temporary_directory).resolve()
    ...     data_path = root / "sample_tabular.npz"
    ...     label_path = root / "labels.npy"
    ...     config_path = root / "config.toml"
    ...     X = np.array([[0.0, 1.0], [1.0, 0.0]])
    ...     y = np.array(["class_a", "class_b"])
    ...     np.savez(data_path, X=X, y=y)
    ...     np.save(label_path, y)
    ...     config_text = (
    ...         "[datasets.sample_tabular]\\n"
    ...         f'path = "{data_path.as_posix()}"\\n'
    ...         f'label_path = "{label_path.as_posix()}"\\n'
    ...         'description = "Neutral numeric example"\\n'
    ...     )
    ...     _ = config_path.write_text(config_text, encoding="utf-8")
    ...     cfg = Config(user_config=config_path)
    ...     loaded = load_datasets(cfg)
    ...     loaded["sample_tabular"]["X"].shape
    (2, 2)
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
    """Load reduced feature matrices and labels for evaluation.

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
        # Legacy loading is best-effort: log one unexpected failure and continue.
        except Exception as exc:  # noqa: BLE001
            logger.error("Error loading %s: %s", dataset_id, exc)

    if loaded == 0:
        logger.warning(
            "No datasets loaded for %s with levels %s", reduction_type, levels
        )
    else:
        logger.info("Loaded %d/%d datasets for %s", loaded, len(levels), reduction_type)

    return reductions
