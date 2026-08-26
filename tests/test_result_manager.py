# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for melite.result_manager."""

import csv
import json
from datetime import datetime

import pytest

import melite.result_manager as result_manager_module
from melite.result_manager import ResultManager
from melite.version import __version__


SAMPLE_ROWS = [
    {
        "reduction_type": "PCA",
        "level": 70,
        "classifier_name": "SVC",
        "parameters": "{'kernel': 'linear', 'C': 1}",
        "f1_macro": 0.85,
        "f1_std": 0.02,
        "accuracy": 0.86,
        "acc_std": 0.02,
        "auc_roc": 0.90,
        "auc_std": 0.01,
    }
]

EVALUATION_FIELDS = [
    "dataset",
    "family",
    "method",
    "variant",
    "level",
    "description",
    "reduction_type",
    "classifier_name",
    "f1_macro",
    "f1_std",
    "accuracy",
    "acc_std",
    "auc_roc",
    "auc_std",
    "selected",
    "smoke",
]

FOLD_FIELDS = [
    "dataset",
    "family",
    "method",
    "variant",
    "level",
    "description",
    "reduction_type",
    "classifier_name",
    "outer_split",
    "outer_repeat",
    "outer_fold",
    "f1_macro",
    "accuracy",
    "auc_roc",
    "selected",
    "smoke",
]

OPTIMIZATION_FIELDS = [
    "dataset",
    "family",
    "method",
    "variant",
    "level",
    "description",
    "reduction_type",
    "classifier_name",
    "search_scope",
    "outer_split",
    "outer_repeat",
    "outer_fold",
    "best_inner_f1_macro",
    "best_params",
    "n_trials_requested",
    "n_trials_complete",
    "n_trials_failed",
    "selected",
    "smoke",
]

EVALUATION_ROW = {
    "dataset": "morgan",
    "family": "fingerprints",
    "method": "Morgan",
    "variant": "r2_2048",
    "level": None,
    "description": "Morgan fingerprint",
    "reduction_type": None,
    "classifier_name": "SVC",
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
    "classifier_name": "SVC",
    "outer_split": 3,
    "outer_repeat": 1,
    "outer_fold": 1,
    "f1_macro": 0.8012345678,
    "accuracy": 0.8123456789,
    "auc_roc": None,
    "selected": True,
}

OPTIMIZATION_ROW = {
    "dataset": "morgan",
    "family": "fingerprints",
    "method": "Morgan",
    "variant": "r2_2048",
    "level": None,
    "description": "Morgan fingerprint",
    "reduction_type": None,
    "classifier_name": "SVC",
    "search_scope": "outer",
    "outer_split": 3,
    "outer_repeat": 1,
    "outer_fold": 1,
    "best_inner_f1_macro": 0.8123456789012345,
    "best_params": {"svc__kernel": "rbf", "svc__C": 0.123456789012345},
    "n_trials_requested": 100,
    "n_trials_complete": 97,
    "n_trials_failed": 3,
    "selected": True,
}


def _provenance():
    return {
        "melite_version": __version__,
        "optimization_backend": {"name": "optuna", "version": "4.9.0"},
        "smoke": False,
        "random_state": 42,
        "active_classifiers": ["svc"],
        "cv": {"n_splits": 5, "n_repeats": 3, "inner_n_splits": 3},
        "optimization": {"effective_n_trials": 100, "policy": {}},
        "search_spaces": {"svc": {"classifier": "svc"}},
    }


def _raise_simulated_write_failure(*args, **kwargs):
    raise OSError("simulated write failure")


def test_write_results_creates_file(tmp_path):
    output_file = tmp_path / "results.txt"
    rm = ResultManager(str(output_file))
    rm.write_results("test content")
    assert output_file.exists()


def test_write_results_header_is_exactly_preserved(monkeypatch, tmp_path):
    class FixedDatetime:
        @staticmethod
        def now():
            return datetime(2024, 1, 2, 3, 4, 5)

    monkeypatch.setattr(result_manager_module, "datetime", FixedDatetime)
    output_file = tmp_path / "results.txt"
    rm = ResultManager(str(output_file))
    rm.write_results("report body")
    content = output_file.read_text(encoding="utf-8")
    assert (
        content
        == f"""
=====================================================
                       MELITE
            Multi-Model Classifier Evaluator
-----------------------------------------------------
Classifiers: SVC, RandomForest, XGBoost, Stacking (opt-in)
CLI: melite run | melite export
Package: melite
Version: {__version__}
Licence: LGPL-3.0-or-later
Execution Date: 2024-01-02 03:04:05
-----------------------------------------------------
Repository: https://github.com/NanoBiostructuresRG/melite
=====================================================

report body"""
    )


def test_write_results_propagates_write_failure(monkeypatch, tmp_path):
    rm = ResultManager(str(tmp_path / "results.txt"))
    monkeypatch.setattr("builtins.open", _raise_simulated_write_failure)

    with pytest.raises(OSError, match="simulated write failure"):
        rm.write_results("report body")


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
            "f1_std",
            "accuracy",
            "acc_std",
            "auc_roc",
            "auc_std",
            "smoke",
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


def test_write_csv_propagates_write_failure(monkeypatch, tmp_path):
    rm = ResultManager(str(tmp_path / "results.txt"))
    monkeypatch.setattr("builtins.open", _raise_simulated_write_failure)

    with pytest.raises(OSError, match="simulated write failure"):
        rm.write_csv(SAMPLE_ROWS, tmp_path / "results.csv")


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


@pytest.mark.parametrize(
    ("method_name", "rows", "filename"),
    [
        ("write_evaluations_csv", [EVALUATION_ROW], "evaluations.csv"),
        (
            "write_evaluation_folds_csv",
            [FOLD_ROW],
            "evaluation_folds.csv",
        ),
    ],
)
def test_evaluation_csv_writers_propagate_write_failure(
    monkeypatch, tmp_path, method_name, rows, filename
):
    rm = ResultManager(str(tmp_path / "results.txt"))
    monkeypatch.setattr("builtins.open", _raise_simulated_write_failure)

    with pytest.raises(OSError, match="simulated write failure"):
        getattr(rm, method_name)(rows, tmp_path / filename)


def test_write_optimization_searches_csv_uses_exact_schema_and_canonical_json(
    tmp_path,
):
    rm = ResultManager(str(tmp_path / "results.txt"))
    csv_path = tmp_path / "nested" / "optimization_searches.csv"

    rm.write_optimization_searches_csv([OPTIMIZATION_ROW], csv_path, smoke=True)

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames == OPTIMIZATION_FIELDS
    assert len(rows) == 1
    assert rows[0]["best_inner_f1_macro"] == "0.8123456789012345"
    assert rows[0]["best_params"] == (
        '{"svc__C":0.123456789012345,"svc__kernel":"rbf"}'
    )
    assert json.loads(rows[0]["best_params"]) == OPTIMIZATION_ROW["best_params"]
    assert rows[0]["n_trials_requested"] == "100"
    assert rows[0]["n_trials_complete"] == "97"
    assert rows[0]["n_trials_failed"] == "3"
    assert rows[0]["selected"] == "True"
    assert rows[0]["smoke"] == "True"
    assert "n_trials_pruned" not in rows[0]


def test_optimization_search_serialization_failure_leaves_no_partial_csv(tmp_path):
    rm = ResultManager(str(tmp_path / "results.txt"))
    csv_path = tmp_path / "nested" / "optimization_searches.csv"
    rows = [
        OPTIMIZATION_ROW,
        {
            **OPTIMIZATION_ROW,
            "outer_split": 4,
            "best_params": {"svc__C": float("nan")},
        },
    ]

    with pytest.raises(ValueError, match="Out of range float values"):
        rm.write_optimization_searches_csv(rows, csv_path)

    assert not csv_path.exists()


def test_write_optimization_searches_csv_creates_header_when_empty(tmp_path):
    rm = ResultManager(str(tmp_path / "results.txt"))
    csv_path = tmp_path / "optimization_searches.csv"

    rm.write_optimization_searches_csv([], csv_path)

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == OPTIMIZATION_FIELDS
        assert list(reader) == []


def test_write_optimization_searches_csv_accepts_absent_best_params(tmp_path):
    rm = ResultManager(str(tmp_path / "results.txt"))
    csv_path = tmp_path / "optimization_searches.csv"

    rm.write_optimization_searches_csv([{}], csv_path)

    with open(csv_path, encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert row["best_params"] == ""


def test_write_optimization_provenance_json_is_deterministic(tmp_path):
    rm = ResultManager(str(tmp_path / "results.txt"))
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    provenance = _provenance()

    rm.write_optimization_provenance_json(provenance, first_path)
    rm.write_optimization_provenance_json(provenance, second_path)

    first = first_path.read_bytes()
    assert first == second_path.read_bytes()
    assert first.endswith(b"\n")
    assert not first.endswith(b"\n\n")
    assert json.loads(first) == provenance
    assert set(json.loads(first)) == set(
        result_manager_module._OPTIMIZATION_PROVENANCE_KEYS
    )
    assert "schema_version" not in json.loads(first)


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_write_optimization_provenance_json_rejects_invalid_keys(tmp_path, change):
    rm = ResultManager(str(tmp_path / "results.txt"))
    provenance = _provenance()
    if change == "missing":
        provenance.pop("cv")
    else:
        provenance["schema_version"] = 1

    with pytest.raises(ValueError, match="must contain exactly"):
        rm.write_optimization_provenance_json(provenance, tmp_path / "invalid.json")


def test_write_optimization_provenance_json_rejects_nan(tmp_path):
    rm = ResultManager(str(tmp_path / "results.txt"))
    provenance = _provenance()
    provenance["optimization"]["effective_n_trials"] = float("nan")
    json_path = tmp_path / "invalid.json"

    with pytest.raises(ValueError, match="Out of range float values"):
        rm.write_optimization_provenance_json(provenance, json_path)

    assert not json_path.exists()


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
            "classifier_name": "RandomForestClassifier",
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

    assert calls == [
        {
            "classifier_scores": {
                "SVC": [0.80, 0.82],
                "RandomForestClassifier": [0.71],
            },
            "selected_classifier": "SVC",
            "dataset_id": "morgan",
            "save_to": tmp_path / "figures" / "evaluation_f1_macro_morgan.png",
            "smoke": True,
        }
    ]
    assert closed == [sentinel_figure]
