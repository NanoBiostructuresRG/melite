# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the development-only v0.3.0 characterization calibration."""

import csv
import io
import json
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import characterize_v030 as characterization


@pytest.fixture
def baseline_evidence():
    return json.loads(characterization.REPORT_PATH.read_text(encoding="utf-8"))


def _valid_provenance(optuna_version="4.9.0"):
    return {
        "melite_version": "0.3.0",
        "optimization_backend": {"name": "optuna", "version": optuna_version},
        "smoke": False,
        "random_state": 42,
        "active_classifiers": ["svc", "rf", "xgb"],
        "cv": {"n_splits": 5, "n_repeats": 1, "inner_n_splits": 3},
        "optimization": {
            "effective_n_trials": 100,
            "policy": asdict(characterization.OPTIMIZATION_POLICY),
        },
        "search_spaces": characterization.expected_search_spaces(),
    }


def _valid_search_rows(selected_classifier="SVC"):
    rows = []
    for classifier_name in characterization.CLASSIFIER_NAMES.values():
        for outer_split in range(5):
            rows.append(
                {
                    "classifier_name": classifier_name,
                    "search_scope": "outer",
                    "outer_split": str(outer_split),
                    "outer_repeat": "0",
                    "outer_fold": str(outer_split),
                    "n_trials_requested": "100",
                    "n_trials_complete": "100",
                    "n_trials_failed": "0",
                    "selected": str(classifier_name == selected_classifier),
                    "smoke": "False",
                }
            )
    rows.append(
        {
            "classifier_name": selected_classifier,
            "search_scope": "final",
            "outer_split": "",
            "outer_repeat": "",
            "outer_fold": "",
            "n_trials_requested": "100",
            "n_trials_complete": "100",
            "n_trials_failed": "0",
            "selected": "",
            "smoke": "False",
        }
    )
    return rows


def _write_candidate_artifacts(output_dir, fold_score=0.8):
    output_dir.mkdir()
    (output_dir / "evaluations.csv").write_text("classifier_name\n", encoding="utf-8")
    with open(output_dir / "results.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["classifier_name", "smoke"])
        writer.writeheader()
        writer.writerow({"classifier_name": "SVC", "smoke": "False"})
    with open(
        output_dir / "evaluation_folds.csv", "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=["classifier_name", "f1_macro"])
        writer.writeheader()
        for classifier_name in characterization.CLASSIFIER_NAMES.values():
            for _ in range(5):
                writer.writerow(
                    {"classifier_name": classifier_name, "f1_macro": fold_score}
                )
    rows = _valid_search_rows()
    with open(
        output_dir / "optimization_searches.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "optimization_provenance.json").write_text(
        json.dumps(_valid_provenance()), encoding="utf-8"
    )


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


def test_characterization_parser_accepts_exact_modes():
    parser = characterization.build_parser()
    assert parser.parse_args(["calibrate"]).mode == "calibrate"
    assert parser.parse_args(["candidate"]).mode == "candidate"
    with pytest.raises(SystemExit):
        parser.parse_args(["other"])


def test_missing_baseline_evidence_fails(tmp_path):
    with pytest.raises(characterization.CalibrationError, match="does not exist"):
        characterization.load_baseline_evidence(tmp_path / "missing.json")


def test_malformed_baseline_json_fails(tmp_path):
    path = tmp_path / "baseline.json"
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(characterization.CalibrationError, match="invalid JSON"):
        characterization.load_baseline_evidence(path)


def test_nonpassing_baseline_status_fails(baseline_evidence):
    baseline_evidence["calibration_status"] = "failed"
    with pytest.raises(characterization.CalibrationError, match="calibration_status"):
        characterization.validate_baseline_evidence(baseline_evidence)


def test_repository_baseline_matches_committed_head():
    identity = characterization.committed_baseline_identity()
    assert identity["sha256"] == characterization.sha256_bytes(
        characterization.REPORT_PATH.read_bytes()
    )
    assert identity["git_blob"]
    assert identity["git_commit"]
    assert identity["passed"] is True


def test_working_tree_modified_baseline_fails_committed_gate():
    committed = b'{"calibration_status":"passed"}\n'
    with pytest.raises(characterization.CalibrationError, match="working-tree"):
        characterization.validate_committed_baseline_bytes(
            committed + b"modified",
            committed,
            committed,
            git_commit="commit",
            git_blob="blob",
        )


def test_staged_baseline_difference_fails_committed_gate():
    with pytest.raises(characterization.CalibrationError, match="staged"):
        characterization.validate_committed_baseline_bytes(
            b"committed",
            b"staged",
            b"committed",
            git_commit="commit",
            git_blob="blob",
        )


def test_modified_baseline_fails_before_candidate_execution(monkeypatch, tmp_path):
    def reject_modified_baseline():
        return characterization.validate_committed_baseline_bytes(
            b"modified",
            b"committed",
            b"committed",
            git_commit="commit",
            git_blob="blob",
        )

    def unexpected_candidate_execution(*args, **kwargs):
        pytest.fail("candidate execution started before the committed-baseline gate")

    monkeypatch.setattr(
        characterization, "committed_baseline_identity", reject_modified_baseline
    )
    monkeypatch.setattr(characterization, "_run", unexpected_candidate_execution)

    assert (
        characterization.candidate(
            report_path=tmp_path / "failure.json",
            console_log_path=tmp_path / "console.log",
        )
        is False
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("class_sep", 0.8, "class_sep"),
        ("dataset_sha256", "wrong", "SHA-256"),
    ],
)
def test_selected_profile_identity_mismatch_fails(
    baseline_evidence, field, value, message
):
    baseline_evidence["selected_profile"][field] = value
    with pytest.raises(characterization.CalibrationError, match=message):
        characterization.validate_baseline_evidence(baseline_evidence)


def test_matching_python_version_passes():
    assert characterization.validate_python_version("3.12.3", "3.12.3", "python") == {
        "expected": "3.12.3",
        "actual": "3.12.3",
        "passed": True,
    }


def test_wrong_python_version_fails_with_complete_diagnostic():
    with pytest.raises(characterization.CalibrationError) as error:
        characterization.validate_python_version("3.12.3", "3.11.9", "/isolated/python")
    message = str(error.value)
    assert "expected 3.12.3" in message
    assert "actual 3.11.9" in message
    assert "executable /isolated/python" in message


def test_common_dependency_mismatch_fails(baseline_evidence):
    expected = baseline_evidence["environment"]["installed_packages"]
    actual = {**expected, "numpy": "0.0.0"}
    with pytest.raises(characterization.CalibrationError, match="numpy"):
        characterization.validate_dependency_versions(
            expected, actual, "candidate_venv"
        )


def test_exact_common_dependency_match_passes(baseline_evidence):
    expected = baseline_evidence["environment"]["installed_packages"]
    result = characterization.validate_dependency_versions(
        expected, dict(expected), "orchestrator"
    )
    assert result == {
        "environment": "orchestrator",
        "versions": expected,
        "passed": True,
    }


def test_optuna_pin_is_validated_separately_from_common_dependencies(
    baseline_evidence,
):
    assert "optuna" not in baseline_evidence["environment"]["installed_packages"]
    pin = characterization.expected_optuna_version()
    assert characterization.validate_optuna_version(pin, pin)["passed"] is True
    with pytest.raises(characterization.CalibrationError, match="Optuna"):
        characterization.validate_optuna_version(pin, "0.0.0")


def test_frozen_candidate_dataset_sha_matches(baseline_evidence):
    _, gate = characterization.validate_candidate_dataset(baseline_evidence)
    assert gate == {
        "expected_sha256": characterization.FROZEN_DATASET_SHA256,
        "actual_sha256": characterization.FROZEN_DATASET_SHA256,
        "passed": True,
    }


def test_committed_characterization_fixture_has_frozen_sha256():
    assert (
        characterization.sha256_bytes(characterization.FROZEN_DATASET_PATH.read_bytes())
        == "8fcf49be02395073e63014f6096d587897595c664cf08f3dddf55aac470a29bb"
    )


def test_candidate_dataset_sha_mismatch_fails_before_subprocess(
    monkeypatch, tmp_path, baseline_evidence
):
    baseline_bytes = characterization.REPORT_PATH.read_bytes()
    monkeypatch.setattr(
        characterization, "load_baseline_evidence", lambda: baseline_evidence
    )
    monkeypatch.setattr(
        characterization,
        "installed_distribution_versions",
        lambda: baseline_evidence["environment"]["installed_packages"],
    )
    invalid_fixture = tmp_path / "invalid.csv"
    invalid_fixture.write_bytes(b"bad")
    monkeypatch.setattr(characterization, "FROZEN_DATASET_PATH", invalid_fixture)

    def unexpected_subprocess(*args, **kwargs):
        pytest.fail("candidate SHA failure must occur before any subprocess")

    monkeypatch.setattr(characterization, "_run", unexpected_subprocess)
    assert (
        characterization.candidate(
            report_path=tmp_path / "failure.json",
            console_log_path=tmp_path / "console.log",
        )
        is False
    )
    assert characterization.REPORT_PATH.read_bytes() == baseline_bytes


def test_frozen_candidate_dataset_does_not_depend_on_generator_metadata(
    baseline_evidence,
):
    baseline_evidence["selected_profile"]["generator_parameters"]["n_informative"] = 10
    _, gate = characterization.validate_candidate_dataset(baseline_evidence)
    assert gate["passed"] is True


def test_preexisting_output_is_rejected(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    with pytest.raises(characterization.CalibrationError, match="must not exist"):
        characterization.require_fresh_output(output_dir)


def test_absent_output_passes_fresh_state_gate(tmp_path):
    output_dir = tmp_path / "output"
    assert characterization.require_fresh_output(output_dir) == {
        "fresh_before_execution": True
    }
    assert not output_dir.exists()


def test_candidate_config_freezes_execution_contract(tmp_path):
    config_path = characterization._write_candidate_config(
        tmp_path / "run", tmp_path / "dataset.csv"
    )
    config = config_path.read_text(encoding="utf-8")
    assert 'active = ["svc","rf","xgb"]' in config
    assert "n_trials = 100" in config
    assert "n_splits = 5" in config
    assert "n_repeats = 1" in config
    assert "inner_n_splits = 3" in config
    assert "stack" not in config


def test_candidate_parsing_is_scoped_to_exact_fresh_output(tmp_path, baseline_evidence):
    exact_output = tmp_path / "fresh" / "output"
    exact_output.parent.mkdir()
    _write_candidate_artifacts(exact_output, fold_score=0.8)
    decoy_output = tmp_path / "output"
    decoy_output.mkdir()
    (decoy_output / "evaluation_folds.csv").write_text(
        "classifier_name,f1_macro\nSVC,0.0\n", encoding="utf-8"
    )

    parsed = characterization.load_candidate_outputs(
        exact_output, baseline_evidence, "4.9.0"
    )

    assert parsed["candidate_means"] == {
        "SVC": 0.8,
        "RandomForestClassifier": 0.8,
        "XGBClassifier": 0.8,
    }


def test_correct_candidate_provenance_contract_passes():
    result = characterization.validate_candidate_provenance(
        _valid_provenance(), "4.9.0"
    )
    assert result["overall_pass"] is True


def test_wrong_candidate_melite_version_fails():
    provenance = _valid_provenance()
    provenance["melite_version"] = "0.2.5"
    with pytest.raises(characterization.CalibrationError, match="provenance"):
        characterization.validate_candidate_provenance(provenance, "4.9.0")


@pytest.mark.parametrize(
    "change",
    [
        lambda provenance: provenance.update(active_classifiers=["svc"]),
        lambda provenance: provenance.update(cv={"n_splits": 3}),
        lambda provenance: provenance["optimization"].update(effective_n_trials=99),
    ],
)
def test_wrong_active_cv_or_trial_provenance_fails(change):
    provenance = _valid_provenance()
    change(provenance)
    with pytest.raises(characterization.CalibrationError, match="provenance"):
        characterization.validate_candidate_provenance(provenance, "4.9.0")


def test_wrong_optuna_runtime_provenance_fails():
    with pytest.raises(characterization.CalibrationError, match="provenance"):
        characterization.validate_candidate_provenance(
            _valid_provenance("4.8.0"), "4.9.0"
        )


def test_exact_search_scope_and_complete_budget_pass():
    result = characterization.validate_optimization_searches(
        _valid_search_rows(), "SVC"
    )
    assert result["outer_searches"] == 15
    assert result["final_searches"] == 1
    assert result["total_searches"] == 16
    assert result["e1_budget_accounting"] is True
    assert result["e2_trial_health"] is True


def test_wrong_search_row_count_or_scope_fails():
    rows = _valid_search_rows()[:-1]
    with pytest.raises(characterization.CalibrationError, match="15 outer"):
        characterization.validate_optimization_searches(rows, "SVC")


def test_e1_budget_mismatch_fails():
    rows = _valid_search_rows()
    rows[0]["n_trials_complete"] = "99"
    with pytest.raises(characterization.CalibrationError, match="E1"):
        characterization.validate_optimization_searches(rows, "SVC")


def test_e2_failed_trial_fails():
    rows = _valid_search_rows()
    rows[0]["n_trials_complete"] = "99"
    rows[0]["n_trials_failed"] = "1"
    with pytest.raises(characterization.CalibrationError, match="E2"):
        characterization.validate_optimization_searches(rows, "SVC")


def test_search_gate_does_not_require_n_trials_pruned():
    rows = _valid_search_rows()
    assert all("n_trials_pruned" not in row for row in rows)
    result = characterization.validate_optimization_searches(rows, "SVC")
    assert result["n_trials_pruned_required"] is False


def test_fold_means_require_exactly_five_scores_per_candidate(tmp_path):
    path = tmp_path / "evaluation_folds.csv"
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["classifier_name", "f1_macro"])
        writer.writeheader()
        for classifier_name in characterization.CLASSIFIER_NAMES.values():
            for score in (0.7, 0.75, 0.8, 0.85, 0.9):
                writer.writerow({"classifier_name": classifier_name, "f1_macro": score})
    assert characterization.fold_means_from_csv(
        path, tuple(characterization.CLASSIFIER_NAMES.values())
    ) == {
        "SVC": 0.8,
        "RandomForestClassifier": 0.8,
        "XGBClassifier": 0.8,
    }


def test_scientific_delta_equal_to_margin_passes():
    baseline = {name: 0.8 for name in characterization.CLASSIFIER_NAMES.values()}
    candidate = {name: 0.75 for name in characterization.CLASSIFIER_NAMES.values()}
    result = characterization.scientific_comparison(baseline, candidate, "SVC", "SVC")
    assert result["overall_pass"] is True


def test_scientific_delta_below_margin_fails():
    baseline = {name: 0.8 for name in characterization.CLASSIFIER_NAMES.values()}
    candidate = dict(baseline)
    candidate["XGBClassifier"] = 0.749
    result = characterization.scientific_comparison(baseline, candidate, "SVC", "SVC")
    assert result["classifiers"]["XGBClassifier"]["passed"] is False
    assert result["overall_pass"] is False


def test_changed_candidate_winner_is_informational():
    means = {name: 0.8 for name in characterization.CLASSIFIER_NAMES.values()}
    result = characterization.scientific_comparison(
        means, means, "SVC", "XGBClassifier"
    )
    assert result["selected_classifier_changed"] is True
    assert result["overall_pass"] is True


def test_worst_classifier_delta_is_minimum():
    baseline = {name: 0.8 for name in characterization.CLASSIFIER_NAMES.values()}
    candidate = {"SVC": 0.81, "RandomForestClassifier": 0.78, "XGBClassifier": 0.8}
    result = characterization.scientific_comparison(baseline, candidate, "SVC", "SVC")
    assert result["worst_classifier_delta"] == {
        "classifier": "RandomForestClassifier",
        "delta": pytest.approx(-0.02),
    }


def test_candidate_fit_count_is_4816():
    accounting = characterization.candidate_fit_count_accounting()
    assert accounting["per_search_fit_count"] == 301
    assert accounting["outer_search_fit_count"] == 4515
    assert accounting["final_search_fit_count"] == 301
    assert accounting["expected_total_fit_count"] == 4816


def test_tee_helper_captures_combined_console_without_melite(tmp_path, capsys):
    log_path = tmp_path / "candidate.log"
    characterization.tee_subprocess(
        [
            sys.executable,
            "-c",
            "import sys; print('out'); print('err', file=sys.stderr)",
        ],
        cwd=tmp_path,
        env=None,
        log_path=log_path,
    )
    terminal = capsys.readouterr().out
    log = log_path.read_text(encoding="utf-8")
    assert "out" in terminal and "err" in terminal
    assert "out" in log and "err" in log


def test_success_report_rejects_local_temp_paths(tmp_path):
    with pytest.raises(characterization.CalibrationError, match="absolute paths"):
        characterization.validate_portable_success_report(
            {"characterization_status": "passed", "path": str(tmp_path.resolve())}
        )
    characterization.validate_portable_success_report(
        {"characterization_status": "passed", "dataset_sha256": "abc"}
    )


def test_candidate_helpers_do_not_mutate_baseline_report(baseline_evidence):
    original_file = characterization.REPORT_PATH.read_bytes()
    original_object = deepcopy(baseline_evidence)
    characterization.validate_baseline_evidence(baseline_evidence)
    characterization.validate_candidate_dataset(baseline_evidence)
    characterization.scientific_comparison(
        baseline_evidence["stage_2"]["classifier_mean_outer_f1_macro"],
        baseline_evidence["stage_2"]["classifier_mean_outer_f1_macro"],
        baseline_evidence["stage_2"]["selected_classifier"],
        baseline_evidence["stage_2"]["selected_classifier"],
    )
    assert baseline_evidence == original_object
    assert characterization.REPORT_PATH.read_bytes() == original_file


def test_candidate_mocked_happy_path_writes_complete_success_report(
    monkeypatch, tmp_path, baseline_evidence
):
    baseline_bytes = characterization.REPORT_PATH.read_bytes()
    expected_common = baseline_evidence["environment"]["installed_packages"]
    evidence_identity = {
        "sha256": characterization.sha256_bytes(baseline_bytes),
        "git_blob": "baseline-git-blob",
        "git_commit": "baseline-evidence-commit",
        "passed": True,
    }
    candidate_root = tmp_path / "isolated-candidate-root"
    report_path = tmp_path / "B5_characterization.json"
    console_path = tmp_path / "B5_candidate_console.log"
    run_commands = []
    tee_commands = []

    monkeypatch.setattr(
        characterization,
        "committed_baseline_identity",
        lambda: dict(evidence_identity),
    )
    monkeypatch.setattr(
        characterization.platform,
        "python_version",
        lambda: baseline_evidence["environment"]["orchestrator_python_version"],
    )
    monkeypatch.setattr(
        characterization,
        "installed_distribution_versions",
        lambda: dict(expected_common),
    )

    def fake_mkdtemp(prefix):
        assert prefix == "melite-b5-candidate-"
        candidate_root.mkdir()
        return str(candidate_root)

    monkeypatch.setattr(characterization.tempfile, "mkdtemp", fake_mkdtemp)

    def fake_run(command, *, cwd, env=None, capture=False):
        parts = [str(part) for part in command]
        run_commands.append(parts)
        if parts[:3] == ["git", "rev-parse", "HEAD"]:
            assert capture is True
            return "candidate-source-commit"
        if "build" in parts:
            dist_dir = Path(parts[parts.index("--outdir") + 1])
            dist_dir.mkdir(parents=True)
            (dist_dir / "melite-0.3.0-py3-none-any.whl").write_bytes(b"wheel")
        return ""

    monkeypatch.setattr(characterization, "_run", fake_run)
    monkeypatch.setattr(
        characterization,
        "_candidate_environment_info",
        lambda python, cwd, env: {
            "python_version": baseline_evidence["environment"][
                "orchestrator_python_version"
            ],
            "packages": dict(expected_common),
            "optuna_version": characterization.expected_optuna_version(),
            "melite_version": "0.3.0",
        },
    )

    def fake_tee(command, *, cwd, env, log_path):
        tee_commands.append([str(part) for part in command])
        _write_candidate_artifacts(cwd / "output", fold_score=0.8)
        log_path.write_text("mocked verbose MELITE output\n", encoding="utf-8")

    monkeypatch.setattr(characterization, "tee_subprocess", fake_tee)
    monkeypatch.setattr(
        characterization.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )

    assert (
        characterization.candidate(
            report_path=report_path,
            console_log_path=console_path,
        )
        is True
    )

    report_text = report_path.read_text(encoding="utf-8")
    report = json.loads(report_text)
    assert report["characterization_status"] == "passed"
    assert report["overall_pass"] is True
    assert report["baseline_evidence"]["committed_evidence"] == evidence_identity
    assert report["provenance_gates"]["overall_pass"] is True
    assert report["optimization_search_gates"]["overall_pass"] is True
    assert report["scientific_comparison"]["overall_pass"] is True
    assert report["workload"]["candidate"]["expected_total_fit_count"] == 4816
    assert report_text == f"{characterization.serialize_report(report)}\n"
    assert str(candidate_root) not in report_text
    assert characterization.REPORT_PATH.read_bytes() == baseline_bytes
    assert console_path.read_text(encoding="utf-8") == "mocked verbose MELITE output\n"
    assert tee_commands == [
        [
            str(characterization._venv_melite(candidate_root / "candidate-venv")),
            "run",
            "--verbose",
            "--config",
            str(candidate_root / "candidate-run" / "config.toml"),
        ]
    ]
    assert any("build" in command for command in run_commands)
    assert any("venv" in command for command in run_commands)
    assert any("install" in command for command in run_commands)
