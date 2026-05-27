# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for melite.export_best_model."""

import csv

import numpy as np
import pytest

from melite.config import Config
from melite.export_best_model import Finalizer


class DummyModel:
    def fit(self, X, y):
        self.shape_ = X.shape
        return self


def _make_config(tmp_path):
    cfg = Config()
    cfg.PATHS = {
        "INPUT":   str(tmp_path / "raw") + "/",
        "DATASET": str(tmp_path / "data") + "/",
        "OUTPUT":  str(tmp_path / "output") + "/",
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


def test_get_selected_row_valid_index(tmp_path, tmp_results_csv):
    cfg = _make_config(tmp_path)
    output_dir = tmp_results_csv.parent
    finalizer = Finalizer(tmp_results_csv, output_dir, cfg, row_index=0)
    row = finalizer._get_selected_row()
    assert row["model_name"] == "SVC"
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
            "dataset", "family", "method", "variant", "level", "description",
            "reduction_type", "model_name", "parameters", "f1_macro", "accuracy",
            "auc_roc", "smoke",
        ],
        {
            "dataset": "morgan_r2_2048",
            "family": "fingerprints",
            "method": "Morgan",
            "variant": "",
            "level": "",
            "description": "",
            "reduction_type": "",
            "model_name": "SVC",
            "parameters": "{'kernel': 'linear', 'C': 1}",
            "f1_macro": 0.8,
            "accuracy": 0.8,
            "auc_roc": 0.9,
            "smoke": False,
        },
    )
    monkeypatch.setattr(Finalizer, "_build_model", staticmethod(lambda *_: DummyModel()))
    monkeypatch.setattr(Finalizer, "_cv_and_plot", lambda *args, **kwargs: None)

    Finalizer(csv_path, tmp_path / "output", cfg, row_index=0).run()

    assert (tmp_path / "output" / "Model_SVC_morgan_r2_2048.pkl").exists()


def test_export_legacy_row_falls_back_to_reduction_and_level(monkeypatch, tmp_path):
    _, y = _write_labels(tmp_path)
    _write_npz(tmp_path, "PCA70", np.ones((20, 5)), y)
    cfg = _make_config(tmp_path)
    csv_path = tmp_path / "output" / "results.csv"
    _write_results_csv(
        csv_path,
        [
            "reduction_type", "level", "model_name", "parameters",
            "f1_macro", "accuracy", "auc_roc", "smoke",
        ],
        {
            "reduction_type": "PCA",
            "level": 70,
            "model_name": "SVC",
            "parameters": "{'kernel': 'linear', 'C': 1}",
            "f1_macro": 0.8,
            "accuracy": 0.8,
            "auc_roc": 0.9,
            "smoke": False,
        },
    )
    monkeypatch.setattr(Finalizer, "_build_model", staticmethod(lambda *_: DummyModel()))
    monkeypatch.setattr(Finalizer, "_cv_and_plot", lambda *args, **kwargs: None)

    Finalizer(csv_path, tmp_path / "output", cfg, row_index=0).run()

    assert (tmp_path / "output" / "Model_SVC_PCA70.pkl").exists()


def test_cv_plot_uses_dataset_id_for_figure(monkeypatch, tmp_path):
    import melite.export_best_model as export_module

    cfg = _make_config(tmp_path)
    csv_path = tmp_path / "output" / "results.csv"
    _write_results_csv(
        csv_path,
        ["dataset", "model_name", "parameters", "smoke"],
        {
            "dataset": "rdkit_descriptors",
            "model_name": "SVC",
            "parameters": "{'kernel': 'linear', 'C': 1}",
            "smoke": False,
        },
    )
    saved = {}
    monkeypatch.setattr(
        export_module,
        "cross_validate",
        lambda *args, **kwargs: {
            "test_f1": np.array([0.8]),
            "test_acc": np.array([0.8]),
            "test_auc": np.array([0.9]),
        },
    )
    monkeypatch.setattr(
        export_module,
        "plot_cv_distributions",
        lambda *args, **kwargs: saved.update({"save_to": kwargs["save_to"]}),
    )
    finalizer = Finalizer(csv_path, tmp_path / "output", cfg, row_index=0)
    row = finalizer._get_selected_row()

    finalizer._cv_and_plot(
        DummyModel(), np.ones((20, 5)), np.array([0, 1] * 10), row, tmp_path
    )

    assert saved["save_to"] == tmp_path / "SVC_rdkit_descriptors.png"
