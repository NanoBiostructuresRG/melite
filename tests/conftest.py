# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared pytest fixtures for MELITE test suite."""

import csv

import joblib
import numpy as np
import pytest
from sklearn.svm import SVC


# ------------------------------------------------------------------ #
# Synthetic data constants
# ------------------------------------------------------------------ #
N_SAMPLES = 20
N_FEATURES = 5
LABELS = np.array([0, 1] * (N_SAMPLES // 2), dtype=np.int64)


# ------------------------------------------------------------------ #
# Label fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def tmp_labels(tmp_path):
    """Write synthetic labels.npy to tmp_path/raw/."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    labels_path = raw_dir / "labels.npy"
    np.save(labels_path, LABELS)
    return labels_path


# ------------------------------------------------------------------ #
# NPZ fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def tmp_npz_valid(tmp_path):
    """Valid .npz with X and matching y."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    path = data_dir / "PCA70.npz"
    X = np.random.rand(N_SAMPLES, N_FEATURES).astype(np.float32)
    np.savez(path, X=X, y=LABELS)
    return path


@pytest.fixture
def tmp_npz_no_y(tmp_path):
    """Valid .npz with X only, no y key."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    path = data_dir / "PCA70.npz"
    X = np.random.rand(N_SAMPLES, N_FEATURES).astype(np.float32)
    np.savez(path, X=X)
    return path


@pytest.fixture
def tmp_npz_missing_X(tmp_path):
    """.npz without X key."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    path = data_dir / "PCA70.npz"
    np.savez(path, y=LABELS)
    return path


@pytest.fixture
def tmp_npz_mismatched_y(tmp_path):
    """.npz with y that does not match labels.npy."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    path = data_dir / "PCA70.npz"
    X = np.random.rand(N_SAMPLES, N_FEATURES).astype(np.float32)
    bad_y = np.ones(N_SAMPLES, dtype=np.int64)
    np.savez(path, X=X, y=bad_y)
    return path


# ------------------------------------------------------------------ #
# Results CSV fixture
# ------------------------------------------------------------------ #

@pytest.fixture
def tmp_results_csv(tmp_path):
    """Minimal results.csv with one non-smoke row and one smoke row."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    csv_path = output_dir / "results.csv"

    fieldnames = [
        "reduction_type", "level", "classifier_name", "parameters",
        "f1_macro", "f1_std", "accuracy", "acc_std", "auc_roc", "auc_std", "smoke",
    ]
    rows = [
        {
            "reduction_type": "PCA", "level": 70, "classifier_name": "SVC",
            "parameters": "{'kernel': 'linear', 'C': 1}",
            "f1_macro": 0.85, "f1_std": 0.02,
            "accuracy": 0.86, "acc_std": 0.02,
            "auc_roc": 0.90, "auc_std": 0.01,
            "smoke": False,
        },
        {
            "reduction_type": "PCA", "level": 75, "classifier_name": "SVC",
            "parameters": "{'kernel': 'linear', 'C': 1}",
            "f1_macro": 0.72, "f1_std": 0.04,
            "accuracy": 0.73, "acc_std": 0.04,
            "auc_roc": 0.80, "auc_std": 0.03,
            "smoke": True,
        },
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


# ------------------------------------------------------------------ #
# Config fixture
# ------------------------------------------------------------------ #

@pytest.fixture
def base_config(tmp_path):
    """Config instance pointing to tmp_path directories, without calling setup()."""
    from melite.config import Config
    cfg = Config()
    cfg.PATHS = {
        "INPUT":   str(tmp_path / "raw") + "/",
        "DATASET": str(tmp_path / "data") + "/",
        "OUTPUT":  str(tmp_path / "output") + "/",
    }
    cfg.RESULTS_FILE = str(tmp_path / "output" / "results.txt")
    return cfg


# ------------------------------------------------------------------ #
# Trained model fixture (for test_predict.py)
# ------------------------------------------------------------------ #

@pytest.fixture
def tmp_model(tmp_path):
    """Train a minimal SVC and save as .pkl. Returns the path."""
    X = np.random.rand(N_SAMPLES, N_FEATURES).astype(np.float32)
    y = LABELS.copy()
    model = SVC(kernel="linear", C=1, probability=True, random_state=42)
    model.fit(X, y)
    model_path = tmp_path / "test_model.pkl"
    joblib.dump(model, model_path)
    return model_path
