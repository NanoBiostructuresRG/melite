# SPDX-License-Identifier: LGPL-3.0-or-later
# load_dataset.py - Load datasets for training and evaluation
import os
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_dataset(config, reduction_type, levels):
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
            # B1 — missing file: full path + actionable hint
            if not os.path.exists(data_path):
                logger.warning(
                    "Expected file not found: %s\n"
                    "           Place the reduced feature matrix at this path and retry.",
                    data_path,
                )
                continue

            data = np.load(data_path)
            logger.info("Keys in %s: %s", data_file, data.files)

            # B2 — missing X key: filename + available keys
            if "X" not in data.files:
                raise ValueError(
                    f"Required key 'X' not found in {data_path}.\n"
                    f"           Available keys: {list(data.files)}"
                )

            X = data["X"]

            # B3 — label mismatch: shapes + differing element count
            if "y" in data.files:
                y_from_file = data["y"]
                if not np.array_equal(y_from_file, y):
                    n_diff = int(np.sum(y_from_file != y)) if y_from_file.shape == y.shape else "N/A"
                    raise ValueError(
                        f"Label mismatch in {data_path}:\n"
                        f"           embedded y (shape={y_from_file.shape}) does not match\n"
                        f"           {labels_path} (shape={y.shape}).\n"
                        f"           Differing elements: {n_diff}/{y.shape[0] if y_from_file.shape == y.shape else '?'}."
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
