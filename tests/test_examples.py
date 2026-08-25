# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the synthetic example dataset integrity."""

import subprocess
import sys
from pathlib import Path

import numpy as np

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
LABELS_PATH  = EXAMPLES_DIR / "sample_labels.npy"
NPZ_PATH     = EXAMPLES_DIR / "sample_PCA70.npz"


def test_labels_file_exists():
    assert LABELS_PATH.exists(), f"Missing: {LABELS_PATH}"


def test_npz_file_exists():
    assert NPZ_PATH.exists(), f"Missing: {NPZ_PATH}"


def test_labels_shape():
    y = np.load(LABELS_PATH)
    assert y.shape == (100,)


def test_labels_binary():
    y = np.load(LABELS_PATH)
    assert set(y.tolist()) == {0, 1}


def test_labels_balanced():
    y = np.load(LABELS_PATH)
    assert (y == 0).sum() == 50
    assert (y == 1).sum() == 50


def test_npz_contains_X_key():
    data = np.load(NPZ_PATH)
    assert "X" in data.files


def test_npz_contains_y_key():
    data = np.load(NPZ_PATH)
    assert "y" in data.files


def test_X_shape():
    data = np.load(NPZ_PATH)
    assert data["X"].shape == (100, 37)


def test_y_shape():
    data = np.load(NPZ_PATH)
    assert data["y"].shape == (100,)


def test_y_matches_labels():
    y_labels = np.load(LABELS_PATH)
    data = np.load(NPZ_PATH)
    assert np.array_equal(data["y"], y_labels)


def test_generate_script_is_deterministic():
    """Running generate_sample_data.py twice produces identical files."""
    script = EXAMPLES_DIR / "generate_sample_data.py"

    first_result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True
    )
    assert first_result.returncode == 0

    first_labels = np.load(LABELS_PATH).copy()
    with np.load(NPZ_PATH) as first_data:
        first_X = first_data["X"].copy()
        first_embedded_y = first_data["y"].copy()

    second_result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True
    )
    assert second_result.returncode == 0

    second_labels = np.load(LABELS_PATH)
    with np.load(NPZ_PATH) as second_data:
        second_X = second_data["X"]
        second_embedded_y = second_data["y"]

    assert np.array_equal(first_labels, second_labels)
    assert np.array_equal(first_X, second_X)
    assert np.array_equal(first_embedded_y, second_embedded_y)
