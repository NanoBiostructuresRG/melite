# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the development-only v0.3.0 characterization calibration."""

import csv
import io
import json
from pathlib import Path

import pytest

from scripts import characterize_v030 as characterization


def test_generator_parameters_are_exact():
    assert characterization.GENERATOR_PARAMETERS == {
        "n_samples": 240,
        "n_features": 20,
        "n_informative": 12,
        "n_redundant": 4,
        "n_repeated": 0,
        "n_classes": 2,
        "n_clusters_per_class": 2,
        "flip_y": 0.05,
        "weights": [0.5, 0.5],
        "random_state": 42,
    }


def test_class_sep_candidate_order_is_exact():
    assert characterization.CLASS_SEP_CANDIDATES == (0.60, 0.70, 0.80, 0.90, 1.00)


def test_dataset_generation_is_deterministic():
    first = characterization.generate_dataset_bytes(0.8)
    second = characterization.generate_dataset_bytes(0.8)
    assert first == second
    assert characterization.sha256_bytes(first) == characterization.sha256_bytes(second)


def test_class_sep_changes_dataset_bytes_and_hash():
    lower = characterization.generate_dataset_bytes(0.7)
    higher = characterization.generate_dataset_bytes(0.8)
    assert lower != higher
    assert characterization.sha256_bytes(lower) != characterization.sha256_bytes(higher)


def test_dataset_has_twenty_features_and_label_without_index():
    payload = characterization.generate_dataset_bytes(0.8).decode("utf-8")
    reader = csv.DictReader(io.StringIO(payload))
    assert reader.fieldnames == [
        *(f"feature_{index:03d}" for index in range(20)),
        "label",
    ]
    assert len(list(reader)) == 240
    assert not any(name.startswith("Unnamed:") for name in reader.fieldnames)


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0.649999, False), (0.65, True), (0.95, True), (0.950001, False)],
)
def test_svc_eligibility_boundaries_are_inclusive(score, expected):
    assert characterization.svc_profile_is_eligible(score) is expected


def test_profile_selection_uses_closest_mean_to_point_eight():
    entries = [
        {"class_sep": 0.6, "svc_mean_outer_f1_macro": 0.71, "eligible": True},
        {"class_sep": 0.7, "svc_mean_outer_f1_macro": 0.79, "eligible": True},
        {"class_sep": 0.8, "svc_mean_outer_f1_macro": 0.85, "eligible": True},
    ]
    assert characterization.select_svc_profile(entries) is entries[1]


def test_profile_selection_breaks_exact_tie_with_lower_class_sep():
    entries = [
        {"class_sep": 0.9, "svc_mean_outer_f1_macro": 0.81, "eligible": True},
        {"class_sep": 0.7, "svc_mean_outer_f1_macro": 0.79, "eligible": True},
    ]
    assert characterization.select_svc_profile(entries) is entries[1]


def test_profile_selection_fails_when_none_are_eligible():
    entries = [{"class_sep": 0.6, "svc_mean_outer_f1_macro": 0.64, "eligible": False}]
    with pytest.raises(characterization.CalibrationError, match="No SVC"):
        characterization.select_svc_profile(entries)


def test_three_classifier_acceptance_bands_are_inclusive():
    means = {
        "SVC": 0.65,
        "RandomForestClassifier": 0.95,
        "XGBClassifier": 0.80,
    }
    result = characterization.acceptance_result(means, "XGBClassifier")
    assert result["classifier_band_checks"] == {
        "SVC": True,
        "RandomForestClassifier": True,
        "XGBClassifier": True,
    }
    assert result["overall_pass"] is True


@pytest.mark.parametrize("winner_score", [0.699999, 0.900001])
def test_winner_outside_band_fails_validation(winner_score):
    means = {
        "SVC": winner_score,
        "RandomForestClassifier": 0.80,
        "XGBClassifier": 0.80,
    }
    result = characterization.acceptance_result(means, "SVC")
    assert result["winner_band_check"] is False
    assert result["overall_pass"] is False


def test_validation_failure_does_not_apply_fallback_axis():
    means = {
        "SVC": 0.64,
        "RandomForestClassifier": 0.80,
        "XGBClassifier": 0.80,
    }
    result = characterization.acceptance_result(means, "RandomForestClassifier")
    assert result["overall_pass"] is False
    assert characterization.predefined_reopen_axis() == {
        "name": "n_informative",
        "candidates": [10, 14, 8, 16],
        "applied": False,
    }


def test_historical_candidate_counts_are_exact():
    assert characterization.HISTORICAL_CANDIDATE_COUNTS == {
        "svc": 804,
        "rf": 120,
        "xgb": 1728,
    }


def test_fit_count_accounting_is_exact():
    accounting = characterization.fit_count_accounting("SVC")
    assert accounting["outer_search_fit_count"] == {
        "svc": 12065,
        "rf": 1805,
        "xgb": 25925,
    }
    assert accounting["outer_search_total_fit_count"] == 39795
    assert accounting["possible_final_search_fit_count"] == {
        "svc": 2413,
        "rf": 361,
        "xgb": 5185,
    }
    assert accounting["expected_total_fit_count"] == 42208
    assert "best-model refit" in accounting["definition"]


def test_fold_means_are_recomputed_from_fold_evidence(tmp_path):
    path = tmp_path / "evaluation_folds.csv"
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["classifier_name", "f1_macro"])
        writer.writeheader()
        for score in (0.7, 0.8, 0.9, 0.6, 1.0):
            writer.writerow({"classifier_name": "SVC", "f1_macro": score})
    assert characterization.fold_means_from_csv(path, ("SVC",)) == {"SVC": 0.8}


def test_effective_config_mismatch_is_explicit():
    with pytest.raises(characterization.CalibrationError, match="mismatch"):
        characterization.require_effective_config(
            {"random_state": 41}, {"random_state": 42}
        )


def test_effective_config_checks_actual_absolute_dataset_path(monkeypatch, tmp_path):
    dataset_path = tmp_path / "profiles" / "class_sep_0.70.csv"
    actual = {
        "active_classifiers": ["svc"],
        "random_state": 42,
        "cv": characterization.CV_CONTRACT,
        "dataset": {
            "id": "b5_calibration",
            "path": str((tmp_path / "wrong.csv").resolve()),
            "label_column": "label",
        },
    }
    monkeypatch.setattr(
        characterization, "_run", lambda *args, **kwargs: json.dumps(actual)
    )

    with pytest.raises(characterization.CalibrationError, match="mismatch"):
        characterization._effective_config(
            Path("python"),
            tmp_path / "config.toml",
            ("svc",),
            dataset_path,
            {},
        )


def test_effective_config_persists_stable_dataset_evidence(monkeypatch, tmp_path):
    dataset_path = tmp_path / "profiles" / "class_sep_0.70.csv"
    actual = {
        "active_classifiers": ["svc"],
        "random_state": 42,
        "cv": characterization.CV_CONTRACT,
        "dataset": {
            "id": "b5_calibration",
            "path": str(dataset_path.resolve()),
            "label_column": "label",
        },
    }
    monkeypatch.setattr(
        characterization, "_run", lambda *args, **kwargs: json.dumps(actual)
    )

    effective = characterization._effective_config(
        Path("python"),
        tmp_path / "config.toml",
        ("svc",),
        dataset_path,
        {},
    )

    assert effective["dataset"] == {
        "id": "b5_calibration",
        "label_column": "label",
        "path_verified": True,
    }
    assert str(dataset_path.resolve()) not in json.dumps(effective)


def test_report_serialization_is_deterministic_and_rejects_nan():
    report = {"z": 1, "a": {"value": 2}}
    assert characterization.serialize_report(
        report
    ) == characterization.serialize_report(report)
    assert (
        characterization.serialize_report(report)
        .splitlines()[1]
        .strip()
        .startswith('"a"')
    )
    with pytest.raises(ValueError, match="Out of range float values"):
        characterization.serialize_report({"value": float("nan")})


def test_report_has_exactly_one_trailing_newline(tmp_path):
    path = tmp_path / "report.json"
    characterization.write_report({"status": "passed"}, path)
    payload = path.read_bytes()
    assert payload.endswith(b"\n")
    assert not payload.endswith(b"\n\n")
    assert json.loads(payload) == {"status": "passed"}


def test_calibration_parser_cannot_invoke_candidate_characterization():
    parser = characterization.build_parser()
    assert parser.parse_args(["calibrate"]).mode == "calibrate"
    with pytest.raises(SystemExit):
        parser.parse_args(["candidate"])
