# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for melite.export_best_model."""

import ast
import csv

import joblib
import numpy as np
import pandas as pd
import pytest

from melite.config import Config
from melite.export_best_model import DatasetLoader, Finalizer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


class DummyModel:
    def fit(self, X, y):
        self.shape_ = X.shape
        return self


def _make_config(tmp_path):
    cfg = Config()
    cfg.PATHS = {
        "INPUT": str(tmp_path / "raw") + "/",
        "DATASET": str(tmp_path / "data") + "/",
        "OUTPUT": str(tmp_path / "output") + "/",
    }
    cfg.DATASETS = {}
    return cfg


def _write_labels(tmp_path, n_samples=20):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(exist_ok=True)
    y = np.array([0, 1] * (n_samples // 2), dtype=np.int64)
    path = raw_dir / "labels.npy"
    np.save(path, y)
    return path, y


def _write_npz(tmp_path, name, X, y):
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    path = data_dir / f"{name}.npz"
    np.savez(path, X=X, y=y)
    return path


def _write_npz_without_X(tmp_path, name, Z):
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    path = data_dir / f"{name}.npz"
    np.savez(path, Z=Z)
    return path


def _write_results_csv(path, fieldnames, row):
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def test_missing_csv_raises_file_not_found_error(tmp_path):
    cfg = _make_config(tmp_path)
    missing_csv = tmp_path / "output" / "results.csv"
    with pytest.raises(FileNotFoundError, match="Results file not found"):
        Finalizer(missing_csv, tmp_path / "output", cfg)


def test_missing_csv_error_includes_hint(tmp_path):
    cfg = _make_config(tmp_path)
    missing_csv = tmp_path / "output" / "results.csv"
    with pytest.raises(FileNotFoundError, match="melite run"):
        Finalizer(missing_csv, tmp_path / "output", cfg)


def test_legacy_model_name_column_fails_with_migration_message(tmp_path):
    cfg = _make_config(tmp_path)
    csv_path = tmp_path / "output" / "results.csv"
    _write_results_csv(
        csv_path,
        ["dataset", "model_name", "parameters", "smoke"],
        {
            "dataset": "sample_tabular",
            "model_name": "SVC",
            "parameters": "{'svc__kernel': 'linear', 'svc__C': 1}",
            "smoke": False,
        },
    )

    with pytest.raises(
        ValueError,
        match=("'model_name' was renamed to 'classifier_name' in MELITE v0\\.2\\.4"),
    ):
        Finalizer(csv_path, tmp_path / "output", cfg)


def test_get_selected_row_valid_index(tmp_path, tmp_results_csv):
    cfg = _make_config(tmp_path)
    output_dir = tmp_results_csv.parent
    finalizer = Finalizer(tmp_results_csv, output_dir, cfg, row_index=0)
    row = finalizer._get_selected_row()
    assert row["classifier_name"] == "SVC"
    assert int(row["level"]) == 70


def test_get_selected_row_invalid_index_raises(tmp_path, tmp_results_csv):
    cfg = _make_config(tmp_path)
    output_dir = tmp_results_csv.parent
    finalizer = Finalizer(tmp_results_csv, output_dir, cfg, row_index=99)
    with pytest.raises(ValueError, match="Invalid row index"):
        finalizer._get_selected_row()


def test_smoke_guard_blocks_without_force(tmp_path, tmp_results_csv):
    cfg = _make_config(tmp_path)
    output_dir = tmp_results_csv.parent
    finalizer = Finalizer(tmp_results_csv, output_dir, cfg, row_index=1, force=False)
    row = finalizer._get_selected_row()
    with pytest.raises(SystemExit) as exc_info:
        finalizer._check_smoke_guard(row)
    assert exc_info.value.code == 1


def test_smoke_guard_allows_with_force(tmp_path, tmp_results_csv):
    cfg = _make_config(tmp_path)
    output_dir = tmp_results_csv.parent
    finalizer = Finalizer(tmp_results_csv, output_dir, cfg, row_index=1, force=True)
    row = finalizer._get_selected_row()
    finalizer._check_smoke_guard(row)


def test_smoke_guard_allows_non_smoke_row(tmp_path, tmp_results_csv):
    cfg = _make_config(tmp_path)
    output_dir = tmp_results_csv.parent
    finalizer = Finalizer(tmp_results_csv, output_dir, cfg, row_index=0, force=False)
    row = finalizer._get_selected_row()
    finalizer._check_smoke_guard(row)


def test_export_dataset_row_uses_dataset_id_for_artifact(monkeypatch, tmp_path):
    label_path, y = _write_labels(tmp_path)
    dataset_path = _write_npz(tmp_path, "morgan_r2_2048", np.ones((20, 5)), y)
    cfg = _make_config(tmp_path)
    cfg.RANDOM_STATE = 17
    cfg.DATASETS = {
        "morgan_r2_2048": {
            "path": str(dataset_path),
            "label_path": str(label_path),
            "metadata": {"family": "fingerprints", "method": "Morgan"},
        }
    }
    csv_path = tmp_path / "output" / "results.csv"
    _write_results_csv(
        csv_path,
        [
            "dataset",
            "family",
            "method",
            "variant",
            "level",
            "description",
            "reduction_type",
            "classifier_name",
            "parameters",
            "f1_macro",
            "accuracy",
            "auc_roc",
            "smoke",
        ],
        {
            "dataset": "morgan_r2_2048",
            "family": "fingerprints",
            "method": "Morgan",
            "variant": "",
            "level": "",
            "description": "",
            "reduction_type": "",
            "classifier_name": "SVC",
            "parameters": "{'svc__kernel': 'linear', 'svc__C': 1}",
            "f1_macro": 0.8,
            "accuracy": 0.8,
            "auc_roc": 0.9,
            "smoke": False,
        },
    )
    build_calls = []

    def fake_build_model(*args, **kwargs):
        build_calls.append((args, kwargs))
        return DummyModel()

    monkeypatch.setattr(Finalizer, "_build_model", staticmethod(fake_build_model))
    Finalizer(csv_path, tmp_path / "output", cfg, row_index=0).run()

    assert build_calls[0][1]["cv_config"] is cfg.CV_CONFIG
    assert build_calls[0][1]["random_state"] == 17
    assert (tmp_path / "output" / "Model_SVC_morgan_r2_2048.pkl").exists()


def test_export_dataset_row_uses_strict_load_datasets(monkeypatch, tmp_path):
    import melite.export_best_model as export_module

    label_path, y = _write_labels(tmp_path)
    cfg = _make_config(tmp_path)
    cfg.DATASETS = {
        "maccs": {
            "path": str(tmp_path / "data" / "maccs.npz"),
            "label_path": str(label_path),
            "metadata": {"family": "fingerprints", "method": "MACCS"},
        }
    }
    csv_path = tmp_path / "output" / "results.csv"
    _write_results_csv(
        csv_path,
        ["dataset", "classifier_name", "parameters", "smoke"],
        {
            "dataset": "maccs",
            "classifier_name": "SVC",
            "parameters": "{'svc__kernel': 'linear', 'svc__C': 1}",
            "smoke": False,
        },
    )
    calls = []

    def fake_load_datasets(config):
        calls.append(config)
        return {
            "maccs": {
                "X": np.ones((20, 5)),
                "y": y,
                "metadata": {"family": "fingerprints", "method": "MACCS"},
            }
        }

    monkeypatch.setattr(export_module, "load_datasets", fake_load_datasets)
    monkeypatch.setattr(
        Finalizer, "_build_model", staticmethod(lambda *_, **__: DummyModel())
    )
    Finalizer(csv_path, tmp_path / "output", cfg, row_index=0).run()

    assert calls == [cfg]
    assert (tmp_path / "output" / "Model_SVC_maccs.pkl").exists()


def test_export_dataset_loader_resolves_registered_csv_through_canonical_loader(
    tmp_path,
):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv_path = data_dir / "sample_tabular.csv"
    pd.DataFrame(
        {
            "feature_b": [2.5, 3.5, 4.5],
            "Outcome": ["class_a", "class_b", "class_a"],
            "feature_a": [10, 20, 30],
        }
    ).to_csv(csv_path, index=False)
    cfg = _make_config(tmp_path)
    cfg.DATASETS = {
        "sample_tabular": {
            "path": str(csv_path),
            "label_column": "Outcome",
            "metadata": {"family": "tabular"},
        }
    }

    X, y = DatasetLoader(cfg).load_row(pd.Series({"dataset": "sample_tabular"}))

    assert np.array_equal(X, np.array([[2.5, 10.0], [3.5, 20.0], [4.5, 30.0]]))
    assert np.array_equal(y, np.array(["class_a", "class_b", "class_a"]))


def test_export_dataset_npz_without_X_fails_clearly(monkeypatch, tmp_path):
    label_path, _ = _write_labels(tmp_path)
    dataset_path = _write_npz_without_X(tmp_path, "maccs", np.ones((20, 5)))
    cfg = _make_config(tmp_path)
    cfg.DATASETS = {
        "maccs": {
            "path": str(dataset_path),
            "label_path": str(label_path),
            "metadata": {"family": "fingerprints", "method": "MACCS"},
        }
    }
    csv_path = tmp_path / "output" / "results.csv"
    _write_results_csv(
        csv_path,
        ["dataset", "classifier_name", "parameters", "smoke"],
        {
            "dataset": "maccs",
            "classifier_name": "SVC",
            "parameters": "{'svc__kernel': 'linear', 'svc__C': 1}",
            "smoke": False,
        },
    )
    monkeypatch.setattr(
        Finalizer, "_build_model", staticmethod(lambda *_, **__: DummyModel())
    )
    with pytest.raises(ValueError, match="Required key 'X' not found"):
        Finalizer(csv_path, tmp_path / "output", cfg, row_index=0).run()


def test_export_legacy_row_with_valid_X_and_labels_succeeds(monkeypatch, tmp_path):
    _, y = _write_labels(tmp_path)
    _write_npz(tmp_path, "PCA70", np.ones((20, 5)), y)
    cfg = _make_config(tmp_path)
    csv_path = tmp_path / "output" / "results.csv"
    _write_results_csv(
        csv_path,
        [
            "reduction_type",
            "level",
            "classifier_name",
            "parameters",
            "f1_macro",
            "accuracy",
            "auc_roc",
            "smoke",
        ],
        {
            "reduction_type": "PCA",
            "level": 70,
            "classifier_name": "SVC",
            "parameters": "{'svc__kernel': 'linear', 'svc__C': 1}",
            "f1_macro": 0.8,
            "accuracy": 0.8,
            "auc_roc": 0.9,
            "smoke": False,
        },
    )
    monkeypatch.setattr(
        Finalizer, "_build_model", staticmethod(lambda *_, **__: DummyModel())
    )
    Finalizer(csv_path, tmp_path / "output", cfg, row_index=0).run()

    assert (tmp_path / "output" / "Model_SVC_PCA70.pkl").exists()


def test_export_svc_saves_scaler_pipeline(monkeypatch, tmp_path):
    label_path, y = _write_labels(tmp_path)
    dataset_path = _write_npz(tmp_path, "toy", np.ones((20, 5)), y)
    cfg = _make_config(tmp_path)
    cfg.DATASETS = {
        "toy": {
            "path": str(dataset_path),
            "label_path": str(label_path),
            "metadata": {"family": "smoke", "method": "toy"},
        }
    }
    csv_path = tmp_path / "output" / "results.csv"
    _write_results_csv(
        csv_path,
        ["dataset", "classifier_name", "parameters", "smoke"],
        {
            "dataset": "toy",
            "classifier_name": "SVC",
            "parameters": "{'svc__kernel': 'linear', 'svc__C': 1}",
            "smoke": False,
        },
    )
    Finalizer(csv_path, tmp_path / "output", cfg, row_index=0).run()

    model = joblib.load(tmp_path / "output" / "Model_SVC_toy.pkl")
    assert isinstance(model, SklearnPipeline)
    assert list(model.named_steps) == ["scaler", "svc"]
    assert isinstance(model.named_steps["scaler"], StandardScaler)
    assert isinstance(model.named_steps["svc"], SVC)
    assert model.named_steps["svc"].kernel == "linear"


def test_export_svc_accepts_legacy_unprefixed_parameters():
    model = Finalizer._build_model(
        "SVC", "{'kernel': 'linear', 'C': 1}", random_state=42
    )

    assert isinstance(model, SklearnPipeline)
    assert model.named_steps["svc"].kernel == "linear"
    assert model.named_steps["svc"].C == 1


def test_export_reconstructs_exact_persisted_continuous_parameters(tmp_path):
    final_params = {
        "learning_rate": 0.123456789012345,
        "max_depth": 7,
        "gamma": 0.0123456789012345,
    }
    csv_path = tmp_path / "results.csv"
    _write_results_csv(
        csv_path,
        ["dataset", "classifier_name", "parameters", "smoke"],
        {
            "dataset": "sample_tabular",
            "classifier_name": "XGBClassifier",
            "parameters": str(final_params),
            "smoke": False,
        },
    )

    with open(csv_path, encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    recovered_params = ast.literal_eval(row["parameters"])
    model = Finalizer._build_model(
        row["classifier_name"], row["parameters"], random_state=42
    )

    assert recovered_params == final_params
    assert {
        key: model.get_params()[key] for key in ("learning_rate", "max_depth", "gamma")
    } == final_params


def test_export_rejects_unknown_classifier_with_classifier_vocabulary():
    with pytest.raises(ValueError, match="Unsupported classifier type: KNN"):
        Finalizer._build_model("KNN", "{}", random_state=42)


def test_export_builds_stacking_classifier_with_expected_contract():
    cv_config = {
        "n_splits": 2,
        "n_repeats": 1,
        "inner_n_splits": 3,
    }

    model = Finalizer._build_model(
        "StackingClassifier",
        "{}",
        cv_config=cv_config,
        random_state=17,
    )
    estimators = dict(model.estimators)
    svc = estimators["svc"]
    rf = estimators["rf"]
    xgb = estimators["xgb"]

    assert isinstance(model, StackingClassifier)
    assert model.stack_method == "predict_proba"
    assert model.passthrough is False
    assert model.n_jobs == -1

    assert isinstance(model.cv, StratifiedKFold)
    assert model.cv.n_splits == cv_config["inner_n_splits"]
    assert model.cv.shuffle is True
    assert model.cv.random_state == 17
    assert isinstance(svc, SklearnPipeline)
    assert list(svc.named_steps) == ["scaler", "svc"]
    assert isinstance(svc.named_steps["scaler"], StandardScaler)
    assert isinstance(svc.named_steps["svc"], SVC)
    assert svc.named_steps["svc"].probability is True

    assert isinstance(rf, RandomForestClassifier)
    assert rf.n_jobs == 1

    assert isinstance(xgb, XGBClassifier)
    assert xgb.n_jobs == 1

    assert not isinstance(rf, SklearnPipeline)
    assert not isinstance(xgb, SklearnPipeline)


def test_export_can_rebuild_and_save_stacking_model(monkeypatch, tmp_path):
    label_path, y = _write_labels(tmp_path)
    dataset_path = _write_npz(tmp_path, "toy", np.ones((20, 5)), y)
    cfg = _make_config(tmp_path)
    cfg.CV_CONFIG = {
        "n_splits": 2,
        "n_repeats": 1,
        "inner_n_splits": 2,
    }
    cfg.DATASETS = {
        "toy": {
            "path": str(dataset_path),
            "label_path": str(label_path),
            "metadata": {"family": "smoke", "method": "toy"},
        }
    }
    csv_path = tmp_path / "output" / "results.csv"
    _write_results_csv(
        csv_path,
        ["dataset", "classifier_name", "parameters", "smoke"],
        {
            "dataset": "toy",
            "classifier_name": "StackingClassifier",
            "parameters": "{'rf__n_estimators': 2, 'xgb__n_estimators': 2}",
            "smoke": False,
        },
    )
    Finalizer(csv_path, tmp_path / "output", cfg, row_index=0).run()

    model = joblib.load(tmp_path / "output" / "Model_StackingClassifier_toy.pkl")
    assert isinstance(model, StackingClassifier)
    assert model.stack_method == "predict_proba"


def test_export_legacy_npz_without_X_does_not_fallback_to_first_key(
    monkeypatch, tmp_path
):
    _write_labels(tmp_path)
    _write_npz_without_X(tmp_path, "PCA70", np.ones((20, 5)))
    cfg = _make_config(tmp_path)
    csv_path = tmp_path / "output" / "results.csv"
    _write_results_csv(
        csv_path,
        [
            "reduction_type",
            "level",
            "classifier_name",
            "parameters",
            "f1_macro",
            "accuracy",
            "auc_roc",
            "smoke",
        ],
        {
            "reduction_type": "PCA",
            "level": 70,
            "classifier_name": "SVC",
            "parameters": "{'svc__kernel': 'linear', 'svc__C': 1}",
            "f1_macro": 0.8,
            "accuracy": 0.8,
            "auc_roc": 0.9,
            "smoke": False,
        },
    )
    monkeypatch.setattr(
        Finalizer, "_build_model", staticmethod(lambda *_, **__: DummyModel())
    )
    with pytest.raises(ValueError, match="Required key 'X' not found"):
        Finalizer(csv_path, tmp_path / "output", cfg, row_index=0).run()
