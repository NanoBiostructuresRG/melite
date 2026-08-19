# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for melite.result_manager."""

import csv
import pytest
from pathlib import Path

import melite.result_manager as result_manager_module
from melite.result_manager import ResultManager
from melite.version import __version__


SAMPLE_ROWS = [
    {
        "reduction_type": "PCA", "level": 70, "model_name": "SVC",
        "parameters": "{'kernel': 'linear', 'C': 1}",
        "f1_macro": 0.85, "f1_std": 0.02,
        "accuracy": 0.86, "acc_std": 0.02,
        "auc_roc": 0.90, "auc_std": 0.01,
    }
]

EVALUATION_FIELDS = [
    "dataset", "family", "method", "variant", "level", "description",
    "reduction_type", "model_name", "f1_macro", "f1_std", "accuracy",
    "acc_std", "auc_roc", "auc_std", "selected", "smoke",
]

FOLD_FIELDS = [
    "dataset", "family", "method", "variant", "level", "description",
    "reduction_type", "model_name", "outer_split", "outer_repeat",
    "outer_fold", "f1_macro", "accuracy", "auc_roc", "selected", "smoke",
]

EVALUATION_ROW = {
    "dataset": "morgan",
    "family": "fingerprints",
    "method": "Morgan",
    "variant": "r2_2048",
    "level": None,
    "description": "Morgan fingerprint",
    "reduction_type": None,
    "model_name": "SVC",
    "f1_macro": 0.8123456789,
    "f1_std": 0.0123456789,
    "accuracy": 0.8234567891,
    "acc_std": 0.0234567891,
    "auc_roc": None,
    "auc_std": None,
    "selected": True,
}

FOLD_ROW = {
    "dataset": "morgan",
    "family": "fingerprints",
    "method": "Morgan",
    "variant": "r2_2048",
    "level": None,
    "description": "Morgan fingerprint",
    "reduction_type": None,
    "model_name": "SVC",
    "outer_split": 3,
    "outer_repeat": 1,
    "outer_fold": 1,
    "f1_macro": 0.8012345678,
    "accuracy": 0.8123456789,
    "auc_roc": None,
    "selected": True,
}


def test_write_results_creates_file(tmp_path):
    output_file = tmp_path / "results.txt"
    rm = ResultManager(str(output_file))
    rm.write_results("test content")
    assert output_file.exists()


def test_write_results_header_contains_version(tmp_path):
    output_file = tmp_path / "results.txt"
    rm = ResultManager(str(output_file))
    rm.write_results("")
    content = output_file.read_text(encoding="utf-8")
    assert __version__ in content


def test_write_results_header_contains_content(tmp_path):
    output_file = tmp_path / "results.txt"
    rm = ResultManager(str(output_file))
    rm.write_results("my results here")
    content = output_file.read_text(encoding="utf-8")
    assert "my results here" in content


def test_write_csv_creates_file(tmp_path):
    output_file = tmp_path / "results.txt"
    csv_path = tmp_path / "results.csv"
    rm = ResultManager(str(output_file))
    rm.write_csv(SAMPLE_ROWS, csv_path)
    assert csv_path.exists()


def test_write_csv_correct_fieldnames(tmp_path):
    output_file = tmp_path / "results.txt"
    csv_path = tmp_path / "results.csv"
    rm = ResultManager(str(output_file))
    rm.write_csv(SAMPLE_ROWS, csv_path)
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == [
            "dataset", "family", "method", "variant", "level", "description",
            "reduction_type", "model_name", "parameters", "f1_macro", "f1_std",
            "accuracy", "acc_std", "auc_roc", "auc_std", "smoke",
        ]


def test_write_csv_smoke_true(tmp_path):
    output_file = tmp_path / "results.txt"
    csv_path = tmp_path / "results.csv"
    rm = ResultManager(str(output_file))
    rm.write_csv(SAMPLE_ROWS, csv_path, smoke=True)
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert rows[0]["smoke"] == "True"


def test_write_csv_smoke_false(tmp_path):
    output_file = tmp_path / "results.txt"
    csv_path = tmp_path / "results.csv"
    rm = ResultManager(str(output_file))
    rm.write_csv(SAMPLE_ROWS, csv_path, smoke=False)
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert rows[0]["smoke"] == "False"


def test_write_csv_empty_rows_produces_no_file(tmp_path):
    output_file = tmp_path / "results.txt"
    csv_path = tmp_path / "results.csv"
    rm = ResultManager(str(output_file))
    rm.write_csv([], csv_path)
    assert not csv_path.exists()


def test_write_evaluations_csv_uses_exact_schema_and_raw_values(tmp_path):
    rm = ResultManager(str(tmp_path / "results.txt"))
    csv_path = tmp_path / "nested" / "evaluations.csv"

    rm.write_evaluations_csv([EVALUATION_ROW], csv_path, smoke=True)

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames == EVALUATION_FIELDS
    assert len(rows) == 1
    assert rows[0]["f1_macro"] == "0.8123456789"
    assert rows[0]["auc_roc"] == ""
    assert rows[0]["selected"] == "True"
    assert rows[0]["smoke"] == "True"


def test_write_evaluation_folds_csv_uses_exact_schema_and_raw_values(tmp_path):
    rm = ResultManager(str(tmp_path / "results.txt"))
    csv_path = tmp_path / "nested" / "evaluation_folds.csv"

    rm.write_evaluation_folds_csv([FOLD_ROW], csv_path, smoke=True)

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames == FOLD_FIELDS
    assert len(rows) == 1
    assert rows[0]["outer_split"] == "3"
    assert rows[0]["f1_macro"] == "0.8012345678"
    assert rows[0]["auc_roc"] == ""
    assert rows[0]["selected"] == "True"
    assert rows[0]["smoke"] == "True"


@pytest.mark.parametrize(
    ("method_name", "filename"),
    [
        ("write_evaluations_csv", "evaluations.csv"),
        ("write_evaluation_folds_csv", "evaluation_folds.csv"),
    ],
)
def test_evaluation_csv_writers_skip_empty_rows(tmp_path, method_name, filename):
    rm = ResultManager(str(tmp_path / "results.txt"))
    csv_path = tmp_path / filename

    getattr(rm, method_name)([], csv_path, smoke=True)

    assert not csv_path.exists()


def test_write_evaluation_figures_groups_existing_outer_scores(monkeypatch, tmp_path):
    rows = [
        {
            **FOLD_ROW,
            "outer_split": 0,
            "outer_repeat": 0,
            "outer_fold": 0,
            "f1_macro": 0.80,
        },
        {
            **FOLD_ROW,
            "outer_split": 1,
            "outer_repeat": 0,
            "outer_fold": 1,
            "f1_macro": 0.82,
        },
        {
            **FOLD_ROW,
            "model_name": "RandomForestClassifier",
            "outer_split": 0,
            "outer_repeat": 0,
            "outer_fold": 0,
            "f1_macro": 0.71,
            "selected": False,
        },
    ]
    calls = []
    closed = []
    sentinel_figure = object()

    def fake_plot(**kwargs):
        calls.append(kwargs)
        return sentinel_figure

    monkeypatch.setattr(result_manager_module, "plot_f1_macro_evidence", fake_plot)
    monkeypatch.setattr(result_manager_module.plt, "close", closed.append)
    rm = ResultManager(str(tmp_path / "results.txt"))

    rm.write_evaluation_figures(rows, smoke=True)

    assert calls == [{
        "family_scores": {
            "SVC": [0.80, 0.82],
            "RandomForestClassifier": [0.71],
        },
        "selected_family": "SVC",
        "dataset_id": "morgan",
        "save_to": tmp_path / "figures" / "evaluation_f1_macro_morgan.png",
        "smoke": True,
    }]
    assert closed == [sentinel_figure]
