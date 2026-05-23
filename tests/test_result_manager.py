# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for mosaic.result_manager."""

import csv
import pytest
from pathlib import Path
from mosaic.result_manager import ResultManager
from mosaic.version import __version__


SAMPLE_ROWS = [
    {
        "reduction_type": "PCA", "level": 70, "model_name": "SVC",
        "parameters": "{'kernel': 'linear', 'C': 1}",
        "f1_macro": 0.85, "f1_std": 0.02,
        "accuracy": 0.86, "acc_std": 0.02,
        "auc_roc": 0.90, "auc_std": 0.01,
    }
]


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
        assert "smoke" in reader.fieldnames
        assert "f1_macro" in reader.fieldnames


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
