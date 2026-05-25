# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for mosaic.predict."""

import numpy as np
import pytest
from mosaic.predict import predict


def test_predict_returns_dict(tmp_model):
    X = np.random.rand(10, 5).astype(np.float32)
    result = predict(tmp_model, X)
    assert isinstance(result, dict)


def test_predict_keys(tmp_model):
    X = np.random.rand(10, 5).astype(np.float32)
    result = predict(tmp_model, X)
    assert set(result.keys()) == {"predictions", "probabilities", "model_path", "n_samples"}


def test_predictions_shape(tmp_model):
    X = np.random.rand(10, 5).astype(np.float32)
    result = predict(tmp_model, X)
    assert result["predictions"].shape == (10,)


def test_probabilities_shape_with_return_proba(tmp_model):
    X = np.random.rand(10, 5).astype(np.float32)
    result = predict(tmp_model, X, return_proba=True)
    assert result["probabilities"] is not None
    assert result["probabilities"].shape == (10, 2)


def test_probabilities_none_when_return_proba_false(tmp_model):
    X = np.random.rand(10, 5).astype(np.float32)
    result = predict(tmp_model, X, return_proba=False)
    assert result["probabilities"] is None


def test_n_samples(tmp_model):
    X = np.random.rand(15, 5).astype(np.float32)
    result = predict(tmp_model, X)
    assert result["n_samples"] == 15


def test_model_path_in_result(tmp_model):
    X = np.random.rand(10, 5).astype(np.float32)
    result = predict(tmp_model, X)
    assert str(tmp_model) in result["model_path"]


def test_missing_model_raises_file_not_found(tmp_path):
    X = np.random.rand(10, 5).astype(np.float32)
    missing = tmp_path / "nonexistent.pkl"
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        predict(missing, X)


def test_missing_model_error_includes_hint(tmp_path):
    X = np.random.rand(10, 5).astype(np.float32)
    missing = tmp_path / "nonexistent.pkl"
    with pytest.raises(FileNotFoundError, match="mosaic export"):
        predict(missing, X)


def test_non_2d_input_raises_value_error(tmp_model):
    X_1d = np.random.rand(10).astype(np.float32)
    with pytest.raises(ValueError, match="2-D numpy array"):
        predict(tmp_model, X_1d)


def test_3d_input_raises_value_error(tmp_model):
    X_3d = np.random.rand(10, 5, 3).astype(np.float32)
    with pytest.raises(ValueError, match="2-D numpy array"):
        predict(tmp_model, X_3d)
