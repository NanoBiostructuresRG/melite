# SPDX-License-Identifier: LGPL-3.0-or-later
"""Load registered NPZ and CSV datasets for MELITE.

For NPZ datasets, ``label_path`` supplies the authoritative labels and an
embedded ``y`` is checked only for consistency. For CSV datasets, the
configured ``label_column`` inside the table supplies the labels and is
removed from the numeric feature matrix.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from melite.config import Config

__all__ = ["load_datasets"]

logger = logging.getLogger(__name__)


def _count_differences(left: np.ndarray, right: np.ndarray) -> int:
    return int(np.sum(left != right))


def _is_numeric_dtype(dtype: Any) -> bool:
    try:
        return bool(np.issubdtype(dtype, np.number))
    except TypeError:
        return False


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


def _load_csv_dataset(dataset_id: str, spec: dict) -> dict:
    data_path = Path(spec["path"])
    label_column = spec["label_column"]
    metadata = dict(spec.get("metadata", {}))

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset '{dataset_id}' file not found: {data_path}")

    try:
        table = pd.read_csv(data_path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"Dataset '{dataset_id}' could not parse CSV file '{data_path}': {exc}"
        ) from exc

    unnamed_columns = [
        str(column) for column in table.columns if str(column).startswith("Unnamed:")
    ]
    if unnamed_columns:
        raise ValueError(
            f"Dataset '{dataset_id}' contains unnamed CSV column(s) "
            f"{unnamed_columns}. This commonly comes from exporting a pandas "
            "index; write the CSV with index=False."
        )

    if label_column not in table.columns:
        raise ValueError(
            f"Dataset '{dataset_id}' label_column '{label_column}' was not found "
            f"in CSV file '{data_path}'."
        )

    features = table.drop(columns=[label_column])
    if features.shape[1] == 0:
        raise ValueError(
            f"Dataset '{dataset_id}' CSV must contain at least one feature column "
            f"in addition to label_column '{label_column}'."
        )

    non_numeric_columns = [
        str(column)
        for column, dtype in features.dtypes.items()
        if not _is_numeric_dtype(dtype)
    ]
    if non_numeric_columns:
        raise ValueError(
            f"Dataset '{dataset_id}' feature columns must be numeric; "
            f"non-numeric column(s): {non_numeric_columns}."
        )

    X = features.to_numpy()
    y = table[label_column].to_numpy()
    if X.ndim != 2:
        raise ValueError(f"Dataset '{dataset_id}' X must be 2D; got shape {X.shape}.")
    if y.ndim != 1:
        raise ValueError(f"Dataset '{dataset_id}' y must be 1D; got shape {y.shape}.")
    if len(y) != X.shape[0]:
        raise ValueError(
            f"Dataset '{dataset_id}' X/y length mismatch: "
            f"X has {X.shape[0]} rows, y has {len(y)} labels."
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
          One-dimensional label vector loaded from ``label_path`` for NPZ or
          from ``label_column`` for CSV.
        - ``"metadata"`` : dict
          Shallow copy of the dataset metadata dictionary. Metadata keys are
          transported but not interpreted by ``load_datasets``; nested mutable
          values are not deep-copied.

        Dataset ids are not interpreted as method, representation, reduction,
        or classifier names. For NPZ, the configured ``label_path`` is
        authoritative and an embedded ``y`` is used only as a consistency
        check. For CSV, the configured ``label_column`` supplies ``y`` and all
        remaining columns supply ``X`` in their original order.

    Raises
    ------
    FileNotFoundError
        If a configured dataset file or authoritative label file does not
        exist.
    ValueError
        If an NPZ dataset violates its feature or label consistency contract,
        or if a CSV cannot be parsed, lacks its configured label or any feature
        columns, contains an ``Unnamed:`` column or non-numeric feature column,
        or produces invalid feature or label dimensions or row counts.

    Examples
    --------
    >>> import pandas as pd
    >>> from pathlib import Path
    >>> from tempfile import TemporaryDirectory
    >>> from melite import Config, load_datasets
    >>> with TemporaryDirectory() as temporary_directory:
    ...     root = Path(temporary_directory).resolve()
    ...     data_path = root / "sample_tabular.csv"
    ...     config_path = root / "config.toml"
    ...     table = pd.DataFrame(
    ...         {
    ...             "feature_a": [0.0, 1.0],
    ...             "feature_b": [1.0, 0.0],
    ...             "Outcome": ["class_a", "class_b"],
    ...         }
    ...     )
    ...     table.to_csv(data_path, index=False)
    ...     config_text = (
    ...         "[datasets.sample_tabular]\\n"
    ...         f'path = "{data_path.as_posix()}"\\n'
    ...         'label_column = "Outcome"\\n'
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
        suffix = Path(spec["path"]).suffix.lower()
        if suffix == ".npz":
            loaded[dataset_id] = _load_one_dataset(dataset_id, spec)
        elif suffix == ".csv":
            loaded[dataset_id] = _load_csv_dataset(dataset_id, spec)
        else:
            raise ValueError(
                f"Dataset '{dataset_id}' has unsupported extension "
                f"'{suffix or '<none>'}'. .npz and .csv are the supported "
                "registered-dataset formats."
            )
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
