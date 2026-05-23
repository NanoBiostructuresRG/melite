# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for mosaic.load_dataset."""

import pytest
from mosaic.load_dataset import load_dataset


def _make_config(tmp_path):
    from mosaic.config import Config
    cfg = Config()
    cfg.PATHS = {
        "INPUT":   str(tmp_path / "raw") + "/",
        "DATASET": str(tmp_path / "data") + "/",
        "OUTPUT":  str(tmp_path / "output") + "/",
    }
    return cfg


def test_valid_npz_with_matching_y_loads(tmp_path, tmp_labels, tmp_npz_valid):
    cfg = _make_config(tmp_path)
    result = load_dataset(cfg, "PCA", [70])
    assert "PCA70" in result
    X, y = result["PCA70"]
    assert X.shape == (20, 5)
    assert y.shape == (20,)


def test_valid_npz_without_y_loads(tmp_path, tmp_labels, tmp_npz_no_y):
    cfg = _make_config(tmp_path)
    result = load_dataset(cfg, "PCA", [70])
    assert "PCA70" in result


def test_missing_file_warns_and_skips(tmp_path, tmp_labels, caplog):
    import logging
    cfg = _make_config(tmp_path)
    with caplog.at_level(logging.WARNING, logger="mosaic.load_dataset"):
        result = load_dataset(cfg, "PCA", [70])
    assert result == {}
    assert any("not found" in msg.lower() for msg in caplog.messages)


def test_missing_X_key_raises_value_error(tmp_path, tmp_labels, tmp_npz_missing_X):
    cfg = _make_config(tmp_path)
    with pytest.raises(ValueError, match="Required key 'X' not found"):
        load_dataset(cfg, "PCA", [70])


def test_missing_X_error_includes_available_keys(tmp_path, tmp_labels, tmp_npz_missing_X):
    cfg = _make_config(tmp_path)
    with pytest.raises(ValueError, match="Available keys"):
        load_dataset(cfg, "PCA", [70])


def test_mismatched_y_raises_value_error(tmp_path, tmp_labels, tmp_npz_mismatched_y):
    cfg = _make_config(tmp_path)
    with pytest.raises(ValueError, match="Label mismatch"):
        load_dataset(cfg, "PCA", [70])


def test_mismatched_y_error_includes_shapes(tmp_path, tmp_labels, tmp_npz_mismatched_y):
    cfg = _make_config(tmp_path)
    with pytest.raises(ValueError, match=r"shape=\(20,\)"):
        load_dataset(cfg, "PCA", [70])


def test_mismatched_y_error_includes_diff_count(tmp_path, tmp_labels, tmp_npz_mismatched_y):
    cfg = _make_config(tmp_path)
    with pytest.raises(ValueError, match="Differing elements"):
        load_dataset(cfg, "PCA", [70])
