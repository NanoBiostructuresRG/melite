# SPDX-License-Identifier: LGPL-3.0-or-later
"""Regression execution and reporting integration tests."""

import csv
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes, make_regression

import melite.main as main_module
from melite.main import Main
from melite.result_manager import ResultManager


REGRESSION_ARTIFACTS = {
    "regression_results.txt",
    "regression_results.csv",
    "regression_evaluations.csv",
    "regression_evaluation_folds.csv",
    "regression_optimization_searches.csv",
    "regression_optimization_provenance.json",
}
REGRESSION_IDENTITY_FIELDS = [
    "dataset",
    "task",
    "family",
    "method",
    "variant",
    "level",
    "description",
    "reduction_type",
    "regressor_key",
    "regressor_name",
]
REGRESSION_RESULT_FIELDS = [
    *REGRESSION_IDENTITY_FIELDS,
    "parameters",
    "rmse",
    "rmse_std",
    "mae",
    "mae_std",
    "r2",
    "r2_std",
    "smoke",
]
REGRESSION_EVALUATION_FIELDS = [
    *REGRESSION_IDENTITY_FIELDS,
    "rmse",
    "rmse_std",
    "mae",
    "mae_std",
    "r2",
    "r2_std",
    "selected",
    "smoke",
]
REGRESSION_FOLD_FIELDS = [
    *REGRESSION_IDENTITY_FIELDS,
    "outer_split",
    "outer_repeat",
    "outer_fold",
    "rmse",
    "mae",
    "r2",
    "selected",
    "smoke",
]
REGRESSION_OPTIMIZATION_FIELDS = [
    *REGRESSION_IDENTITY_FIELDS,
    "search_scope",
    "outer_split",
    "outer_repeat",
    "outer_fold",
    "best_inner_rmse",
    "best_params",
    "n_trials_requested",
    "n_trials_complete",
    "n_trials_failed",
    "selected",
    "smoke",
]


def _read_csv(path):
    with open(path, encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def _paths(tmp_path):
    raw_dir = tmp_path / "raw"
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    raw_dir.mkdir()
    data_dir.mkdir()
    return raw_dir, data_dir, output_dir


def _base_config(raw_dir, data_dir, output_dir):
    return f'''
[paths]
input = "{raw_dir.as_posix()}/"
dataset = "{data_dir.as_posix()}/"
output = "{output_dir.as_posix()}/"

[cv]
n_splits = 2
n_repeats = 1
inner_n_splits = 2

[optimization]
n_trials = 1
'''


def _write_regression_csv(path, X, y):
    table = pd.DataFrame(X, columns=[f"feature_{index}" for index in range(X.shape[1])])
    table["target"] = y
    table.to_csv(path, index=False)


def test_real_diabetes_csv_regression_writes_exact_independent_artifacts(
    monkeypatch, tmp_path
):
    class ForbiddenClassificationPipeline:
        def __init__(self, _config):
            pytest.fail("Regression-only execution must not initialize classifiers")

    monkeypatch.setattr(main_module, "Pipeline", ForbiddenClassificationPipeline)
    raw_dir, data_dir, output_dir = _paths(tmp_path)
    X, y = load_diabetes(return_X_y=True)
    csv_path = data_dir / "diabetes.csv"
    _write_regression_csv(csv_path, X, y)
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        _base_config(raw_dir, data_dir, output_dir)
        + f'''
[datasets.diabetes]
path = "{csv_path.as_posix()}"
label_column = "target"
task = "regression"
family = "clinical"
description = "Scikit-learn diabetes data"

[classifiers]
active = []

[regressors]
active = ["linear", "poly"]
''',
        encoding="utf-8",
    )

    main = Main(user_config=config_path)
    main.run()

    assert {path.name for path in output_dir.iterdir()} == REGRESSION_ARTIFACTS
    result_fields, result_rows = _read_csv(output_dir / "regression_results.csv")
    evaluation_fields, evaluation_rows = _read_csv(
        output_dir / "regression_evaluations.csv"
    )
    fold_fields, fold_rows = _read_csv(
        output_dir / "regression_evaluation_folds.csv"
    )
    optimization_fields, optimization_rows = _read_csv(
        output_dir / "regression_optimization_searches.csv"
    )

    assert result_fields == REGRESSION_RESULT_FIELDS
    assert evaluation_fields == REGRESSION_EVALUATION_FIELDS
    assert fold_fields == REGRESSION_FOLD_FIELDS
    assert optimization_fields == REGRESSION_OPTIMIZATION_FIELDS
    assert len(result_rows) == 1
    assert [row["regressor_key"] for row in evaluation_rows] == ["linear", "poly"]
    assert sum(row["selected"] == "True" for row in evaluation_rows) == 1
    assert len(fold_rows) == 4
    assert optimization_rows == []
    assert json.loads(result_rows[0]["parameters"]) == {}
    assert result_rows[0]["task"] == "regression"
    assert float(result_rows[0]["rmse"]) >= 0.0
    assert float(result_rows[0]["mae"]) >= 0.0
    assert result_rows[0]["rmse"] == str(main.regression_csv_rows[0]["rmse"])
    assert all(float(row["rmse"]) >= 0.0 for row in fold_rows)
    assert all(float(row["mae"]) >= 0.0 for row in fold_rows)

    report = (output_dir / "regression_results.txt").read_text(encoding="utf-8")
    assert "Supervised Regression Evaluator" in report
    assert "Regressor selected:" in report
    assert "RMSE (outer-CV mean):" in report
    assert "Multi-Model Classifier Evaluator" not in report

    provenance = json.loads(
        (output_dir / "regression_optimization_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert provenance["task"] == "regression"
    assert provenance["active_regressors"] == ["linear", "poly"]
    assert provenance["cv"] == {
        "inner": {"n_splits": 2, "shuffle": True, "strategy": "KFold"},
        "outer": {
            "n_repeats": 1,
            "n_splits": 2,
            "strategy": "RepeatedKFold",
        },
    }
    assert provenance["optimization"]["objective"] == "rmse"
    assert provenance["optimization"]["direction"] == "minimize"
    assert provenance["optimization"]["sampler"] == "tpe"
    assert provenance["optimization"]["effective_n_trials"] == 1
    assert provenance["search_spaces"] == {"linear": None, "poly": None}


def test_npz_with_external_npy_executes_regression_without_loader_changes(tmp_path):
    raw_dir, data_dir, output_dir = _paths(tmp_path)
    X, y = make_regression(n_samples=24, n_features=3, random_state=42)
    dataset_path = data_dir / "quantitative.npz"
    label_path = raw_dir / "targets.npy"
    np.savez(dataset_path, X=X)
    np.save(label_path, y)
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        _base_config(raw_dir, data_dir, output_dir)
        + f'''
[datasets.quantitative]
path = "{dataset_path.as_posix()}"
label_path = "{label_path.as_posix()}"
task = "regression"

[classifiers]
active = []

[regressors]
active = ["linear"]
''',
        encoding="utf-8",
    )

    main = Main(user_config=config_path)
    main.run()

    _, rows = _read_csv(output_dir / "regression_results.csv")
    assert rows[0]["dataset"] == "quantitative"
    assert rows[0]["regressor_key"] == "linear"
    assert (
        main.regression_evaluations_by_dataset["quantitative"][0]["selected"]
        is True
    )


def test_real_ridge_run_persists_outer_and_fresh_final_searches(tmp_path):
    raw_dir, data_dir, output_dir = _paths(tmp_path)
    X, y = make_regression(n_samples=24, n_features=3, noise=0.1, random_state=42)
    csv_path = data_dir / "ridge.csv"
    _write_regression_csv(csv_path, X, y)
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        _base_config(raw_dir, data_dir, output_dir)
        + f'''
[datasets.ridge_data]
path = "{csv_path.as_posix()}"
label_column = "target"
task = "regression"

[classifiers]
active = []

[regressors]
active = ["ridge"]
''',
        encoding="utf-8",
    )

    Main(user_config=config_path).run()

    _, search_rows = _read_csv(output_dir / "regression_optimization_searches.csv")
    assert [row["search_scope"] for row in search_rows] == [
        "outer",
        "outer",
        "final",
    ]
    assert [row["outer_split"] for row in search_rows] == ["0", "1", ""]
    assert all(row["n_trials_requested"] == "1" for row in search_rows)
    assert all(row["n_trials_complete"] == "1" for row in search_rows)
    assert all(row["selected"] == "True" for row in search_rows)
    for row in search_rows:
        alpha = json.loads(row["best_params"])["ridge__alpha"]
        assert 1e-4 <= alpha <= 1e2

    _, result_rows = _read_csv(output_dir / "regression_results.csv")
    assert json.loads(result_rows[0]["parameters"]) == json.loads(
        search_rows[-1]["best_params"]
    )
    provenance = json.loads(
        (output_dir / "regression_optimization_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    alpha_domain = provenance["search_spaces"]["ridge"]["parameters"][0]
    assert alpha_domain == {
        "domain": {"high": 100.0, "log": True, "low": 0.0001, "type": "float"},
        "name": "alpha",
        "target": "ridge__alpha",
    }


def test_mixed_execution_dispatches_in_registry_order_and_separates_outputs(
    monkeypatch, tmp_path
):
    events = []

    class FakeClassificationPipeline:
        def __init__(self, _config):
            pass

        def run_with_evaluations(self, *_args):
            events.append("classification")
            selected = (object(), {}, 0.8, 0.01, 0.82, 0.02, 0.9, 0.03)
            evaluation = {
                "classifier_key": "svc",
                "f1_macro": 0.8,
                "f1_std": 0.01,
                "accuracy": 0.82,
                "acc_std": 0.02,
                "auc_roc": 0.9,
                "auc_std": 0.03,
                "outer_scores": [
                    {
                        "outer_split": 0,
                        "outer_repeat": 0,
                        "outer_fold": 0,
                        "f1_macro": 0.8,
                        "accuracy": 0.82,
                        "auc_roc": 0.9,
                    }
                ],
                "optimization_searches": [],
                "final_optimization_search": None,
                "selected": True,
            }
            return selected, [evaluation]

    class FakeRegressionTrainer:
        def __init__(self, _config):
            pass

        def evaluate_and_select_models(self, *_args):
            events.append("regression")
            selected = (object(), {}, 1.25, 0.125, 1.0, 0.1, -0.25, 0.05)
            evaluation = {
                "regressor_key": "linear",
                "rmse": 1.25,
                "rmse_std": 0.125,
                "mae": 1.0,
                "mae_std": 0.1,
                "r2": -0.25,
                "r2_std": 0.05,
                "outer_scores": [
                    {
                        "outer_split": 0,
                        "outer_repeat": 0,
                        "outer_fold": 0,
                        "rmse": 1.23456789123,
                        "mae": 1.01234567891,
                        "r2": -0.34567891234,
                    }
                ],
                "optimization_searches": [],
                "final_optimization_search": None,
                "selected": True,
            }
            return selected, [evaluation]

    monkeypatch.setattr(main_module, "Pipeline", FakeClassificationPipeline)
    monkeypatch.setattr(main_module, "RegressionModelTrainer", FakeRegressionTrainer)
    monkeypatch.setattr(
        main_module.ResultManager,
        "write_evaluation_figures",
        lambda *_args, **_kwargs: None,
    )
    raw_dir, data_dir, output_dir = _paths(tmp_path)
    classification_path = data_dir / "classification.csv"
    regression_path = data_dir / "regression.csv"
    pd.DataFrame({"feature": [0.0, 1.0, 0.0, 1.0], "label": [0, 1, 0, 1]}).to_csv(
        classification_path, index=False
    )
    pd.DataFrame(
        {"feature": [0.0, 1.0, 2.0, 3.0], "target": [0.1, 1.2, 2.4, 3.8]}
    ).to_csv(regression_path, index=False)
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        _base_config(raw_dir, data_dir, output_dir)
        + f'''
[datasets.class_data]
path = "{classification_path.as_posix()}"
label_column = "label"

[datasets.reg_data]
path = "{regression_path.as_posix()}"
label_column = "target"
task = "regression"

[classifiers]
active = ["svc"]

[regressors]
active = ["linear"]
''',
        encoding="utf-8",
    )

    Main(user_config=config_path).run()

    assert events == ["classification", "regression"]
    assert (output_dir / "results.csv").exists()
    assert (output_dir / "optimization_provenance.json").exists()
    assert all((output_dir / filename).exists() for filename in REGRESSION_ARTIFACTS)
    _, classification_rows = _read_csv(output_dir / "results.csv")
    _, regression_rows = _read_csv(output_dir / "regression_results.csv")
    assert [row["dataset"] for row in classification_rows] == ["class_data"]
    assert [row["dataset"] for row in regression_rows] == ["reg_data"]


def test_regression_csv_preserves_signed_unrounded_metrics(tmp_path):
    manager = ResultManager(tmp_path / "results.txt")
    row = {
        "dataset": "signed",
        "task": "regression",
        "family": None,
        "method": None,
        "variant": None,
        "level": None,
        "description": None,
        "reduction_type": None,
        "regressor_key": "linear",
        "regressor_name": "LinearRegression",
        "parameters": {},
        "rmse": 1.2345678912345,
        "rmse_std": 0.1234567891234,
        "mae": 1.0123456789123,
        "mae_std": 0.1012345678912,
        "r2": -0.3456789123456,
        "r2_std": 0.0345678912345,
    }

    path = tmp_path / "regression_results.csv"
    manager.write_regression_results_csv([row], path)

    _, rows = _read_csv(path)
    assert rows[0]["rmse"] == "1.2345678912345"
    assert rows[0]["mae"] == "1.0123456789123"
    assert rows[0]["r2"] == "-0.3456789123456"
    assert rows[0]["parameters"] == "{}"


def test_regression_provenance_is_deterministic_and_rejects_nonfinite_json(tmp_path):
    manager = ResultManager(tmp_path / "results.txt")
    provenance = {
        "melite_version": "test",
        "task": "regression",
        "optimization_backend": {"name": "optuna", "version": "test"},
        "smoke": False,
        "random_state": 42,
        "active_regressors": ["ridge"],
        "cv": {
            "outer": {"strategy": "RepeatedKFold", "n_splits": 2, "n_repeats": 1},
            "inner": {"strategy": "KFold", "n_splits": 2, "shuffle": True},
        },
        "optimization": {
            "objective": "rmse",
            "direction": "minimize",
            "sampler": "tpe",
            "effective_n_trials": 1,
        },
        "search_spaces": {"ridge": None},
    }
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    manager.write_regression_optimization_provenance_json(provenance, first)
    manager.write_regression_optimization_provenance_json(provenance, second)

    assert first.read_bytes() == second.read_bytes()
    invalid = {**provenance, "random_state": float("nan")}
    invalid_path = tmp_path / "invalid.json"
    with pytest.raises(ValueError, match="Out of range float values"):
        manager.write_regression_optimization_provenance_json(invalid, invalid_path)
    assert not invalid_path.exists()


def test_regression_smoke_indicator_and_provenance(monkeypatch, tmp_path):
    class FakeRegressionTrainer:
        def __init__(self, _config):
            pass

        def evaluate_and_select_models(self, *_args):
            selected = (object(), {}, 1.0, 0.1, 0.8, 0.08, 0.5, 0.05)
            evaluation = {
                "regressor_key": "linear",
                "rmse": 1.0,
                "rmse_std": 0.1,
                "mae": 0.8,
                "mae_std": 0.08,
                "r2": 0.5,
                "r2_std": 0.05,
                "outer_scores": [],
                "optimization_searches": [],
                "final_optimization_search": None,
                "selected": True,
            }
            return selected, [evaluation]

    monkeypatch.setattr(main_module, "RegressionModelTrainer", FakeRegressionTrainer)
    raw_dir, data_dir, output_dir = _paths(tmp_path)
    data_path = data_dir / "smoke.csv"
    pd.DataFrame({"feature": [0.0, 1.0], "target": [0.2, 1.4]}).to_csv(
        data_path, index=False
    )
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        _base_config(raw_dir, data_dir, output_dir)
        + f'''
[datasets.smoke_data]
path = "{data_path.as_posix()}"
label_column = "target"
task = "regression"

[classifiers]
active = []

[regressors]
active = ["linear"]
''',
        encoding="utf-8",
    )

    Main(smoke=True, user_config=config_path).run()

    for filename in (
        "regression_results.csv",
        "regression_evaluations.csv",
        "regression_evaluation_folds.csv",
        "regression_optimization_searches.csv",
    ):
        fields, rows = _read_csv(output_dir / filename)
        assert fields[-1] == "smoke"
        assert all(row["smoke"] == "True" for row in rows)
    provenance = json.loads(
        (output_dir / "regression_optimization_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert provenance["smoke"] is True
    assert provenance["optimization"]["effective_n_trials"] == 5
