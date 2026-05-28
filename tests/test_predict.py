# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for melite.predict."""

import numpy as np
import pytest
import joblib
from melite.predict import predict
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


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
    with pytest.raises(FileNotFoundError, match="melite export"):
        predict(missing, X)


def test_non_2d_input_raises_value_error(tmp_model):
    X_1d = np.random.rand(10).astype(np.float32)
    with pytest.raises(ValueError, match="2-D numpy array"):
        predict(tmp_model, X_1d)


def test_3d_input_raises_value_error(tmp_model):
    X_3d = np.random.rand(10, 5, 3).astype(np.float32)
    with pytest.raises(ValueError, match="2-D numpy array"):
        predict(tmp_model, X_3d)


def test_predict_loads_exported_svc_pipeline(tmp_path):
    X_train = np.random.rand(20, 5).astype(np.float32)
    y_train = np.array([0, 1] * 10, dtype=np.int64)
    model = SklearnPipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="linear", C=1, probability=True, random_state=42)),
    ])
    model.fit(X_train, y_train)
    model_path = tmp_path / "Model_SVC_toy.pkl"
    joblib.dump(model, model_path)

    result = predict(model_path, np.random.rand(4, 5).astype(np.float32))

    assert result["predictions"].shape == (4,)
    assert result["probabilities"].shape == (4, 2)
