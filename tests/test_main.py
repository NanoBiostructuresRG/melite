# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for melite.main orchestration."""

import csv

import numpy as np
import pytest

import melite.main as main_module
from melite.main import Main, Pipeline


class SVC:
    pass


class DummyPipeline:
    calls = []

    def __init__(self, config):
        self.config = config

    def run(self, X_train, y_train, reduction_type, level):
        self.calls.append({
            "shape": X_train.shape,
            "n_labels": len(y_train),
            "reduction_type": reduction_type,
            "level": level,
        })
        return (
            SVC(),
            {"C": 1.0, "kernel": "linear"},
            0.8,
            0.01,
            0.82,
            0.02,
            0.9,
            0.03,
        )

    def run_with_evaluations(self, X_train, y_train, reduction_type, level):
        selected_result = self.run(X_train, y_train, reduction_type, level)
        evaluations = [{
            "model_key": "svc",
            "dataset_marker": reduction_type,
            "selected": True,
        }]
        return selected_result, evaluations


class DummyTrainer:
    def __init__(self):
        self.legacy_result = object()
        self.rich_result = (object(), [{"model_key": "svc", "selected": True}])
        self.calls = []

    def train_and_select_best_model(self, *args):
        self.calls.append(("legacy", args))
        return self.legacy_result

    def evaluate_and_select_models(self, *args):
        self.calls.append(("rich", args))
        return self.rich_result


def _write_labels(path, n_samples=8):
    path.mkdir(parents=True, exist_ok=True)
    y = np.array([0, 1] * (n_samples // 2), dtype=np.int64)
    label_path = path / "labels.npy"
    np.save(label_path, y)
    return label_path, y


def _write_npz(path, name, X, y):
    path.mkdir(parents=True, exist_ok=True)
    data_path = path / f"{name}.npz"
    np.savez(data_path, X=X, y=y)
    return data_path


def _rows(csv_path):
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_pipeline_run_preserves_legacy_trainer_result():
    trainer = DummyTrainer()
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.model_trainer = trainer

    result = pipeline.run("X", "y", "PCA", 70)

    assert result is trainer.legacy_result
    assert trainer.calls == [("legacy", ("X", "y", "PCA", 70))]


def test_pipeline_run_with_evaluations_returns_rich_result_unchanged():
    trainer = DummyTrainer()
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.model_trainer = trainer

    result = pipeline.run_with_evaluations("X", "y", "PCA", 70)

    assert result is trainer.rich_result
    assert trainer.calls == [("rich", ("X", "y", "PCA", 70))]


def test_main_run_uses_arbitrary_dataset_ids(monkeypatch, tmp_path):
    DummyPipeline.calls = []
    monkeypatch.setattr(main_module, "Pipeline", DummyPipeline)

    raw_dir = tmp_path / "raw"
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    label_path, y = _write_labels(raw_dir)
    morgan_path = _write_npz(data_dir, "morgan_r2_2048", np.ones((8, 4)), y)
    desc_path = _write_npz(data_dir, "rdkit_descriptors", np.ones((8, 3)), y)
    pca_path = _write_npz(data_dir, "PCA85", np.ones((8, 2)), y)
    config_path = tmp_path / "config.toml"
    config_path.write_text(f'''
[paths]
input = "{raw_dir.as_posix()}/"
dataset = "{data_dir.as_posix()}/"
output = "{output_dir.as_posix()}/"

[datasets.morgan_r2_2048]
path = "{morgan_path.as_posix()}"
label_path = "{label_path.as_posix()}"
family = "fingerprints"
method = "Morgan"
variant = "r2_2048"
description = "Morgan radius 2 fingerprint"

[datasets.rdkit_descriptors]
path = "{desc_path.as_posix()}"
label_path = "{label_path.as_posix()}"
family = "descriptors"
method = "RDKit"

[datasets.pca85]
path = "{pca_path.as_posix()}"
label_path = "{label_path.as_posix()}"
family = "dimensionality"
method = "PCA"
level = 85
''')

    main = Main(user_config=config_path)
    main.run()

    rows = _rows(output_dir / "results.csv")
    assert [row["dataset"] for row in rows] == [
        "morgan_r2_2048",
        "rdkit_descriptors",
        "pca85",
    ]
    assert rows[0]["family"] == "fingerprints"
    assert rows[0]["method"] == "Morgan"
    assert rows[0]["variant"] == "r2_2048"
    assert rows[0]["description"] == "Morgan radius 2 fingerprint"
    assert rows[0]["reduction_type"] == ""
    assert rows[1]["method"] == "RDKit"
    assert rows[1]["reduction_type"] == ""
    assert rows[2]["method"] == "PCA"
    assert rows[2]["reduction_type"] == "PCA"
    assert rows[2]["level"] == "85"
    assert DummyPipeline.calls[0]["reduction_type"] == "morgan_r2_2048"
    assert DummyPipeline.calls[1]["reduction_type"] == "rdkit_descriptors"
    assert DummyPipeline.calls[2]["reduction_type"] == "PCA"
    assert DummyPipeline.calls[2]["level"] == 85
    assert list(main.evaluations_by_dataset) == [
        "morgan_r2_2048",
        "rdkit_descriptors",
        "pca85",
    ]
    assert main.evaluations_by_dataset["morgan_r2_2048"] == [{
        "model_key": "svc",
        "dataset_marker": "morgan_r2_2048",
        "selected": True,
    }]
    assert "outer_scores" not in rows[0]
    assert "selected" not in rows[0]


def test_main_run_uses_legacy_registry_metadata(monkeypatch, tmp_path):
    DummyPipeline.calls = []
    monkeypatch.setattr(main_module, "Pipeline", DummyPipeline)

    raw_dir = tmp_path / "raw"
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    label_path, y = _write_labels(raw_dir)
    _write_npz(data_dir, "PCA70", np.ones((8, 2)), y)
    config_path = tmp_path / "legacy.toml"
    config_path.write_text(f'''
[paths]
input = "{raw_dir.as_posix()}/"
dataset = "{data_dir.as_posix()}/"
output = "{output_dir.as_posix()}/"

[benchmark]
reduction_types = ["PCA"]
levels = [70]
''')

    Main(user_config=config_path).run()

    rows = _rows(output_dir / "results.csv")
    assert rows[0]["dataset"] == "PCA70"
    assert rows[0]["family"] == "dimensionality"
    assert rows[0]["method"] == "PCA"
    assert rows[0]["reduction_type"] == "PCA"
    assert rows[0]["level"] == "70"
    assert DummyPipeline.calls == [{
        "shape": (8, 2),
        "n_labels": 8,
        "reduction_type": "PCA",
        "level": 70,
    }]


def test_main_run_missing_registered_dataset_fails(monkeypatch, tmp_path):
    DummyPipeline.calls = []
    monkeypatch.setattr(main_module, "Pipeline", DummyPipeline)

    raw_dir = tmp_path / "raw"
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    label_path, _ = _write_labels(raw_dir)
    missing_path = data_dir / "missing.npz"
    config_path = tmp_path / "missing.toml"
    config_path.write_text(f'''
[paths]
input = "{raw_dir.as_posix()}/"
dataset = "{data_dir.as_posix()}/"
output = "{output_dir.as_posix()}/"

[datasets.maccs]
path = "{missing_path.as_posix()}"
label_path = "{label_path.as_posix()}"
family = "fingerprints"
''')

    with pytest.raises(FileNotFoundError, match="maccs"):
        Main(user_config=config_path).run()
    assert DummyPipeline.calls == []
