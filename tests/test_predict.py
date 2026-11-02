# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for melite.predict."""

import inspect
import logging

import numpy as np
import pytest
import joblib
from melite.export_best_model import Finalizer
from melite.predict import predict
from melite.regression_models import build_regression_model
from sklearn.datasets import make_regression
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
    assert set(result.keys()) == {
        "predictions",
        "probabilities",
        "model_path",
        "n_samples",
    }


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
    model = Finalizer._build_model(
        "SVC",
        "{'svc__kernel': 'linear', 'svc__C': 1}",
        random_state=42,
    )
    model.fit(X_train, y_train)
    model_path = tmp_path / "Model_SVC_toy.pkl"
    joblib.dump(model, model_path)

    result = predict(model_path, np.random.rand(4, 5).astype(np.float32))

    assert result["predictions"].shape == (4,)
    assert result["probabilities"].shape == (4, 2)


def test_predict_neutral_exported_model_usage(tmp_path):
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(20, 5)).astype(np.float32)
    y_train = np.array([0, 1] * 10, dtype=np.int64)
    model = SklearnPipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="linear", C=1, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    model_path = output_dir / "Model_SVC_sample_tabular.pkl"
    joblib.dump(model, model_path)

    X_new = rng.normal(size=(4, 5)).astype(np.float32)
    result = predict(model_path, X_new)

    assert result["predictions"].shape == (4,)
    assert result["n_samples"] == 4


def test_predict_loads_exported_stacking_classifier(tmp_path):
    X_train = np.random.rand(20, 5).astype(np.float32)
    y_train = np.array([0, 1] * 10, dtype=np.int64)
    model = Finalizer._build_model(
        "StackingClassifier",
        "{'rf__n_estimators': 2, 'xgb__n_estimators': 2}",
        cv_config={
            "n_splits": 2,
            "n_repeats": 1,
            "inner_n_splits": 2,
        },
        random_state=42,
    )
    model.fit(X_train, y_train)
    model_path = tmp_path / "Model_StackingClassifier_toy.pkl"
    joblib.dump(model, model_path)

    result = predict(model_path, np.random.rand(4, 5).astype(np.float32))

    assert result["predictions"].shape == (4,)
    assert result["probabilities"].shape == (4, 2)


def _regression_artifact(tmp_path):
    X, y = make_regression(n_samples=20, n_features=3, random_state=42)
    model = build_regression_model("ridge").set_params(ridge__alpha=0.25)
    model.fit(X, y)
    model_path = tmp_path / "Model_Ridge_quantitative.pkl"
    joblib.dump(model, model_path)
    return model_path, X


def test_predict_public_signature_and_return_keys_remain_unchanged(tmp_path):
    assert list(inspect.signature(predict).parameters) == [
        "model_path",
        "X",
        "return_proba",
    ]
    model_path, X = _regression_artifact(tmp_path)

    result = predict(model_path, X[:4])

    assert set(result) == {
        "predictions",
        "probabilities",
        "model_path",
        "n_samples",
    }


def test_regression_prediction_is_quantitative_without_probability_warning(
    caplog, tmp_path
):
    model_path, X = _regression_artifact(tmp_path)

    with caplog.at_level(logging.WARNING, logger="melite.predict"):
        result = predict(model_path, X[:4])

    assert result["predictions"].shape == (4,)
    assert np.issubdtype(result["predictions"].dtype, np.number)
    assert np.all(np.isfinite(result["predictions"]))
    assert result["probabilities"] is None
    assert result["n_samples"] == 4
    assert result["model_path"] == str(model_path.resolve())
    assert not [
        record for record in caplog.records if "predict_proba" in record.message
    ]


@pytest.mark.parametrize(
    ("X", "message"),
    [
        (np.ones((4, 2)), "feature count mismatch"),
        (np.array([[1.0, 2.0, np.nan]]), "only finite"),
        (np.array([["a", "b", "c"]], dtype=object), "real numeric"),
    ],
)
def test_regression_prediction_rejects_invalid_numerical_inputs(
    tmp_path, X, message
):
    model_path, _ = _regression_artifact(tmp_path)

    with pytest.raises(ValueError, match=message):
        predict(model_path, X)
