# SPDX-License-Identifier: LGPL-3.0-or-later
# load_dataset.py - Load datasets for training and evaluation
import os
import numpy as np
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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
        data_path = os.path.join(config.PATHS['DATASET'], data_file)

        try:
            if not os.path.exists(data_path):
                logger.warning("File not found: %s", data_file)
                continue

            data = np.load(data_path)
            logger.info("Keys in %s: %s", data_file, data.files)
                
            X = data["X"]   # Features

            if "y" in data.files:
                y_from_file = data["y"]
                if not np.array_equal(y_from_file, y):
                    raise ValueError(
                        f"Label mismatch in {data_file}: "
                        "embedded y does not match raw/labels.npy"
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
        logger.warning("No datasets loaded for %s with levels %s", reduction_type, levels)
    else:
        logger.info("Loaded %d/%d datasets for %s", loaded, len(levels), reduction_type)

    return reductions
