# SPDX-License-Identifier: LGPL-3.0-or-later
"""Regression export, reconstruction, and workflow integration tests."""

import csv
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes, make_regression
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import melite.regression_optimization as regression_optimization
import melite.regression_training as regression_training
from melite.config import Config
from melite.export_best_model import Finalizer, REGRESSION_RESULT_COLUMNS
from melite.main import Main
from melite.predict import predict
from melite.regression_models import build_regression_model


REGRESSOR_NAMES = {
    "linear": "LinearRegression",
    "poly": "PolynomialRegression",
    "ridge": "Ridge",
    "lasso": "Lasso",
}


def _write_regression_results(
    path,
    *,
    dataset="quantitative",
    regressor_key="linear",
    parameters=None,
    smoke=False,
    task="regression",
    regressor_name=None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "dataset": dataset,
        "task": task,
        "family": "tabular",
        "method": "numeric",
        "variant": "",
        "level": "",
        "description": "Regression export fixture",
        "reduction_type": "",
        "regressor_key": regressor_key,
        "regressor_name": regressor_name or REGRESSOR_NAMES.get(regressor_key, ""),
        "parameters": json.dumps(
            parameters if parameters is not None else {},
            sort_keys=True,
            separators=(",", ":"),
        ),
        "rmse": 1.2345678912345,
        "rmse_std": 0.1234567891234,
        "mae": 1.0123456789123,
        "mae_std": 0.1012345678912,
        "r2": 0.75,
        "r2_std": 0.05,
        "smoke": smoke,
    }
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=REGRESSION_RESULT_COLUMNS)
        writer.writeheader()
        writer.writerow(row)


def _config(tmp_path, dataset_id, spec):
    cfg = Config()
    cfg.PATHS = {
        "INPUT": str(tmp_path / "raw") + "/",
        "DATASET": str(tmp_path / "data") + "/",
        "OUTPUT": str(tmp_path / "output") + "/",
    }
    cfg.DATASETS = {dataset_id: spec}
    cfg.CV_CONFIG = {"n_splits": 2, "n_repeats": 1, "inner_n_splits": 2}
    cfg.RANDOM_STATE = 42
    return cfg


@pytest.mark.parametrize(
    ("regressor_key", "parameters"),
    [
        ("linear", {}),
        ("poly", {}),
        ("ridge", {"ridge__alpha": 0.123456789012345}),
        ("lasso", {"lasso__alpha": 9.87654321098765}),
    ],
)
def test_regression_reconstruction_uses_canonical_estimators_and_exact_parameters(
    regressor_key, parameters
):
    model = Finalizer._build_regression_model(
        regressor_key,
        REGRESSOR_NAMES[regressor_key],
        json.dumps(parameters),
    )

    if regressor_key == "linear":
        assert isinstance(model, LinearRegression)
    else:
        assert isinstance(model, SklearnPipeline)
        assert isinstance(model.named_steps["scaler"], StandardScaler)
    if regressor_key == "poly":
        assert list(model.named_steps) == ["scaler", "poly", "linear"]
        polynomial = model.named_steps["poly"]
        assert isinstance(polynomial, PolynomialFeatures)
        assert polynomial.degree == 2
        assert polynomial.include_bias is False
        assert isinstance(model.named_steps["linear"], LinearRegression)
    if regressor_key == "ridge":
        assert list(model.named_steps) == ["scaler", "ridge"]
        assert isinstance(model.named_steps["ridge"], Ridge)
        assert model.get_params()["ridge__alpha"] == 0.123456789012345
    if regressor_key == "lasso":
        assert list(model.named_steps) == ["scaler", "lasso"]
        assert isinstance(model.named_steps["lasso"], Lasso)
        assert model.named_steps["lasso"].max_iter == 10_000
        assert model.get_params()["lasso__alpha"] == 9.87654321098765


@pytest.mark.parametrize(
    ("key", "name", "parameters", "message"),
    [
        ("unknown", "Unknown", "{}", "Unsupported regression model key"),
        ("ridge", "Lasso", '{"ridge__alpha":1.0}', "identity mismatch"),
        ("linear", "LinearRegression", "{'bad': 1}", "strict JSON"),
        ("linear", "LinearRegression", "[]", "must contain an object"),
        ("poly", "PolynomialRegression", '{"degree":3}', "Invalid parameters"),
        ("ridge", "Ridge", '{"alpha":1.0}', "Invalid parameters"),
        (
            "ridge",
            "Ridge",
            '{"ridge__alpha":1.0,"ridge__alpha":2.0}',
            "duplicate key",
        ),
        ("lasso", "Lasso", '{"lasso__alpha":NaN}', "non-finite JSON"),
        ("ridge", "Ridge", '{"ridge__alpha":1000.0}', "approved domain"),
    ],
)
def test_regression_reconstruction_rejects_invalid_identity_or_parameters(
    key, name, parameters, message
):
    with pytest.raises(ValueError, match=message):
        Finalizer._build_regression_model(key, name, parameters)


def test_regression_results_require_the_exact_explicit_schema(tmp_path):
    csv_path = tmp_path / "regression_results.csv"
    pd.DataFrame(
        [
            {
                "dataset": "quantitative",
                "task": "regression",
                "regressor_key": "linear",
                "regressor_name": "LinearRegression",
                "parameters": "{}",
            }
        ]
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="exact regression_results.csv schema"):
        Finalizer(csv_path, tmp_path, Config(), row_index=0)


def test_classification_schema_cannot_declare_regression_task(tmp_path):
    csv_path = tmp_path / "results.csv"
    pd.DataFrame(
        [
            {
                "dataset": "quantitative",
                "task": "regression",
                "classifier_name": "LogisticRegression",
                "parameters": "{}",
                "smoke": False,
            }
        ]
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="classification and regression task"):
        Finalizer(csv_path, tmp_path, Config(), row_index=0)


@pytest.mark.parametrize(
    ("task", "key", "name", "configured_task", "error"),
    [
        ("classification", "linear", "LinearRegression", "regression", "declare task"),
        ("regression", "unknown", "Unknown", "regression", "Unsupported"),
        ("regression", "linear", "Ridge", "regression", "identity mismatch"),
        (
            "regression",
            "linear",
            "LinearRegression",
            "classification",
            "not 'regression'",
        ),
    ],
)
def test_regression_row_rejects_inconsistent_task_and_model_identity(
    tmp_path, task, key, name, configured_task, error
):
    csv_path = tmp_path / "regression_results.csv"
    _write_regression_results(
        csv_path,
        task=task,
        regressor_key=key,
        regressor_name=name,
    )
    cfg = _config(
        tmp_path,
        "quantitative",
        {"task": configured_task, "path": "unused.csv", "label_column": "target"},
    )
    finalizer = Finalizer(csv_path, tmp_path, cfg, row_index=0)

    with pytest.raises((ValueError, KeyError), match=error):
        finalizer._validate_regression_row(finalizer._get_selected_row())


def test_regression_row_requires_registered_dataset(tmp_path):
    csv_path = tmp_path / "regression_results.csv"
    _write_regression_results(csv_path, dataset="missing")
    cfg = _config(tmp_path, "different", {"task": "regression"})
    finalizer = Finalizer(csv_path, tmp_path, cfg, row_index=0)

    with pytest.raises(KeyError, match="not registered"):
        finalizer._validate_regression_row(finalizer._get_selected_row())


def test_polynomial_export_rejects_expansion_over_limit_before_fitting(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    dataset_path = data_dir / "wide.csv"
    table = pd.DataFrame(np.ones((4, 140)))
    table["target"] = np.arange(4, dtype=float)
    table.to_csv(dataset_path, index=False)
    cfg = _config(
        tmp_path,
        "quantitative",
        {
            "task": "regression",
            "path": str(dataset_path),
            "label_column": "target",
            "metadata": {},
        },
    )
    csv_path = tmp_path / "regression_results.csv"
    _write_regression_results(csv_path, regressor_key="poly")

    with pytest.raises(
        ValueError,
        match="original dimensions=140, expanded dimensions=10010",
    ):
        Finalizer(csv_path, tmp_path / "models", cfg, row_index=0).run()


def test_npz_external_npy_regression_export_preserves_pipeline_and_alpha(
    monkeypatch, tmp_path
):
    raw_dir = tmp_path / "raw"
    data_dir = tmp_path / "data"
    raw_dir.mkdir()
    data_dir.mkdir()
    X, y = make_regression(n_samples=24, n_features=3, noise=0.1, random_state=42)
    dataset_path = data_dir / "quantitative.npz"
    label_path = raw_dir / "targets.npy"
    np.savez(dataset_path, X=X)
    np.save(label_path, y)
    cfg = _config(
        tmp_path,
        "quantitative",
        {
            "task": "regression",
            "path": str(dataset_path),
            "label_path": str(label_path),
            "metadata": {},
        },
    )
    alpha = 0.123456789012345
    csv_path = tmp_path / "regression_results.csv"
    _write_regression_results(
        csv_path,
        regressor_key="ridge",
        parameters={"ridge__alpha": alpha},
    )
    ridge_row = pd.read_csv(csv_path)
    linear_row = ridge_row.copy()
    linear_row.loc[0, "regressor_key"] = "linear"
    linear_row.loc[0, "regressor_name"] = "LinearRegression"
    linear_row.loc[0, "parameters"] = "{}"
    pd.concat([linear_row, ridge_row], ignore_index=True).to_csv(csv_path, index=False)
    output_dir = tmp_path / "models"
    captured = {}
    dump_model = joblib.dump

    def capture_fitted_model(model, path):
        captured["model"] = model
        return dump_model(model, path)

    monkeypatch.setattr(joblib, "dump", capture_fitted_model)

    Finalizer(csv_path, output_dir, cfg, row_index=1).run()

    artifact = output_dir / "Model_Ridge_quantitative.pkl"
    exported = joblib.load(artifact)
    expected = build_regression_model("ridge").set_params(ridge__alpha=alpha).fit(X, y)
    assert isinstance(exported, SklearnPipeline)
    assert exported.get_params()["ridge__alpha"] == alpha
    assert np.allclose(exported.predict(X), captured["model"].predict(X))
    assert np.allclose(exported.predict(X), expected.predict(X))


def test_regression_smoke_guard_runs_before_loading_or_construction(
    monkeypatch, tmp_path
):
    csv_path = tmp_path / "regression_results.csv"
    _write_regression_results(csv_path, smoke=True)
    cfg = _config(tmp_path, "quantitative", {"task": "regression"})
    finalizer = Finalizer(csv_path, tmp_path / "models", cfg, row_index=0)
    monkeypatch.setattr(
        finalizer._loader,
        "load_row",
        lambda *_args: pytest.fail("smoke guard must run before dataset loading"),
    )
    monkeypatch.setattr(
        Finalizer,
        "_build_regression_model",
        staticmethod(
            lambda *_args: pytest.fail("smoke guard must run before construction")
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        finalizer.run()
    assert exc_info.value.code == 1


def test_force_allows_regression_smoke_export_with_warning(capsys, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    X, y = make_regression(n_samples=12, n_features=2, random_state=42)
    dataset_path = data_dir / "quantitative.csv"
    table = pd.DataFrame(X, columns=["x1", "x2"])
    table["target"] = y
    table.to_csv(dataset_path, index=False)
    cfg = _config(
        tmp_path,
        "quantitative",
        {
            "task": "regression",
            "path": str(dataset_path),
            "label_column": "target",
            "metadata": {},
        },
    )
    csv_path = tmp_path / "regression_results.csv"
    _write_regression_results(csv_path, smoke=True)
    output_dir = tmp_path / "models"

    Finalizer(csv_path, output_dir, cfg, row_index=0, force=True).run()

    assert "Exporting a smoke-mode result" in capsys.readouterr().out
    assert (output_dir / "Model_LinearRegression_quantitative.pkl").exists()


def test_diabetes_run_export_load_predict_workflow_without_export_optimization(
    monkeypatch, tmp_path
):
    raw_dir = tmp_path / "raw"
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    raw_dir.mkdir()
    data_dir.mkdir()
    X, y = load_diabetes(return_X_y=True)
    dataset_path = data_dir / "diabetes.csv"
    table = pd.DataFrame(X, columns=[f"feature_{index}" for index in range(X.shape[1])])
    table["target"] = y
    table.to_csv(dataset_path, index=False)
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f'''
[paths]
input = "{raw_dir.as_posix()}/"
dataset = "{data_dir.as_posix()}/"
output = "{output_dir.as_posix()}/"

[datasets.diabetes]
path = "{dataset_path.as_posix()}"
label_column = "target"
task = "regression"

[classifiers]
active = []

[regressors]
active = ["linear"]

[cv]
n_splits = 2
n_repeats = 1
inner_n_splits = 2

[optimization]
n_trials = 1
''',
        encoding="utf-8",
    )
    Main(user_config=config_path).run()
    cfg = Config(user_config=config_path)
    monkeypatch.setattr(
        regression_optimization.optuna,
        "create_study",
        lambda *_args, **_kwargs: pytest.fail("export must not start Optuna"),
    )
    monkeypatch.setattr(
        regression_training,
        "cross_validate",
        lambda *_args, **_kwargs: pytest.fail("export must not run cross-validation"),
    )

    results_path = output_dir / "regression_results.csv"
    Finalizer(results_path, output_dir, cfg, row_index=0).run()

    artifact = output_dir / "Model_LinearRegression_diabetes.pkl"
    exported = joblib.load(artifact)
    expected = build_regression_model("linear").fit(X, y)
    assert np.allclose(exported.predict(X), expected.predict(X))
    prediction_result = predict(artifact, X[:5])
    assert prediction_result["predictions"].shape == (5,)
    assert prediction_result["probabilities"] is None
