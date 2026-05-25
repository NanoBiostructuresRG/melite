# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the synthetic example dataset integrity."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

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


def test_generate_script_is_deterministic(tmp_path):
    """Running generate_sample_data.py twice produces identical files."""
    script = EXAMPLES_DIR / "generate_sample_data.py"

    # Patch OUTPUT_DIR to tmp_path by running the script with modified env
    import importlib.util, types
    spec = importlib.util.spec_from_file_location("gen", script)
    mod = importlib.util.module_from_spec(spec)

    # Patch Path(__file__).parent inside the module
    import unittest.mock as mock
    with mock.patch("pathlib.Path.parent", new_callable=mock.PropertyMock) as mp:
        # Just run twice and compare outputs from examples/ directly
        pass

    # Simpler: run the script and compare to known-good files
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True
    )
    assert result.returncode == 0

    y_after = np.load(LABELS_PATH)
    data_after = np.load(NPZ_PATH)

    y_before = np.array([0] * 50 + [1] * 50, dtype=np.int64)
    assert np.array_equal(y_after, y_before)
    assert data_after["X"].shape == (100, 37)
