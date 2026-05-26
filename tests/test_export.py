# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for melite.export_best_model."""

import pytest
from pathlib import Path
from melite.export_best_model import Finalizer
from melite.config import Config


def _make_config(tmp_path):
    cfg = Config()
    cfg.PATHS = {
        "INPUT":   str(tmp_path / "raw") + "/",
        "DATASET": str(tmp_path / "data") + "/",
        "OUTPUT":  str(tmp_path / "output") + "/",
    }
    return cfg


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
    # Should not raise
    finalizer._check_smoke_guard(row)


def test_smoke_guard_allows_non_smoke_row(tmp_path, tmp_results_csv):
    cfg = _make_config(tmp_path)
    output_dir = tmp_results_csv.parent
    finalizer = Finalizer(tmp_results_csv, output_dir, cfg, row_index=0, force=False)
    row = finalizer._get_selected_row()
    # Row 0 is smoke=False — should not raise
    finalizer._check_smoke_guard(row)
