# SPDX-License-Identifier: LGPL-3.0-or-later
"""Calibration and candidate infrastructure for MELITE v0.3.0 characterization."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

import pandas as pd
from sklearn.datasets import make_classification

from melite.optimization_policy import OPTIMIZATION_POLICY
from melite.search_spaces import get_search_space


REPO_ROOT = Path(__file__).resolve().parents[1]
CONSTRAINTS_PATH = Path(__file__).with_name("characterize_v030_constraints.txt")
REPORT_PATH = REPO_ROOT / "B5_calibration.json"
CANDIDATE_REPORT_PATH = REPO_ROOT / "B5_characterization.json"
CANDIDATE_CONSOLE_LOG_PATH = REPO_ROOT / "B5_candidate_console.log"
BASELINE_TAG = "v0.2.5"
DATASET_ID = "b5_calibration"
FROZEN_CLASS_SEP = 0.70
FROZEN_DATASET_SHA256 = (
    "8fcf49be02395073e63014f6096d587897595c664cf08f3dddf55aac470a29bb"
)
CANDIDATE_VERSION = "0.3.0"
CANDIDATE_N_TRIALS = 100
SCIENTIFIC_MARGIN = -0.05
CANDIDATE_CLASSIFIERS = ("svc", "rf", "xgb")
CANDIDATE_ARTIFACT_NAMES = (
    "results.csv",
    "evaluations.csv",
    "evaluation_folds.csv",
    "optimization_searches.csv",
    "optimization_provenance.json",
)
CLASS_SEP_CANDIDATES = (0.60, 0.70, 0.80, 0.90, 1.00)
REOPEN_N_INFORMATIVE_CANDIDATES = (10, 14, 8, 16)
GENERATOR_PARAMETERS = {
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
CV_CONTRACT = {"n_splits": 5, "n_repeats": 1, "inner_n_splits": 3}
RANDOM_STATE = 42
SVC_ELIGIBILITY_BAND = (0.65, 0.95)
CLASSIFIER_ACCEPTANCE_BAND = (0.65, 0.95)
WINNER_ACCEPTANCE_BAND = (0.70, 0.90)
HISTORICAL_CANDIDATE_COUNTS = {"svc": 804, "rf": 120, "xgb": 1728}
CLASSIFIER_NAMES = {
    "svc": "SVC",
    "rf": "RandomForestClassifier",
    "xgb": "XGBClassifier",
}
COMMON_DISTRIBUTIONS = (
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "xgboost",
    "matplotlib",
    "joblib",
)
FIT_COUNT_DEFINITION = (
    "fit_count means estimated estimator.fit() invocations, including one "
    "best-model refit performed by GridSearchCV after candidate CV evaluation."
)


class CalibrationError(RuntimeError):
    """The fixed baseline calibration protocol could not produce a valid profile."""


def sha256_bytes(payload: bytes) -> str:
    """Return the lowercase SHA-256 digest of bytes."""
    return hashlib.sha256(payload).hexdigest()


def generate_dataset_bytes(class_sep: float) -> bytes:
    """Generate one deterministic calibration CSV payload."""
    X, y = make_classification(class_sep=class_sep, **GENERATOR_PARAMETERS)
    columns = [
        f"feature_{index:03d}" for index in range(GENERATOR_PARAMETERS["n_features"])
    ]
    frame = pd.DataFrame(X, columns=columns)
    frame["label"] = y
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def svc_profile_is_eligible(mean_f1_macro: float) -> bool:
    """Return whether an SVC profile is inside the inclusive calibration band."""
    return SVC_ELIGIBILITY_BAND[0] <= mean_f1_macro <= SVC_ELIGIBILITY_BAND[1]


def select_svc_profile(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the eligible profile closest to 0.80, breaking ties downward."""
    eligible = [entry for entry in entries if entry["eligible"]]
    if not eligible:
        raise CalibrationError("No SVC calibration profile met the eligibility band.")
    return min(
        eligible,
        key=lambda entry: (
            abs(entry["svc_mean_outer_f1_macro"] - 0.80),
            entry["class_sep"],
        ),
    )


def acceptance_result(
    classifier_means: dict[str, float], selected_classifier: str
) -> dict[str, Any]:
    """Evaluate the fixed Stage 2 classifier and winner acceptance bands."""
    expected = set(CLASSIFIER_NAMES.values())
    if set(classifier_means) != expected:
        raise CalibrationError(
            "Stage 2 fold evidence must contain exactly SVC, "
            "RandomForestClassifier, and XGBClassifier."
        )
    if selected_classifier not in classifier_means:
        raise CalibrationError(
            f"Selected classifier {selected_classifier!r} has no fold evidence."
        )
    classifier_checks = {
        name: CLASSIFIER_ACCEPTANCE_BAND[0] <= score <= CLASSIFIER_ACCEPTANCE_BAND[1]
        for name, score in classifier_means.items()
    }
    winner_score = classifier_means[selected_classifier]
    winner_check = (
        WINNER_ACCEPTANCE_BAND[0] <= winner_score <= WINNER_ACCEPTANCE_BAND[1]
    )
    return {
        "classifier_band": list(CLASSIFIER_ACCEPTANCE_BAND),
        "classifier_band_checks": classifier_checks,
        "winner_band": list(WINNER_ACCEPTANCE_BAND),
        "winner_band_check": winner_check,
        "overall_pass": all(classifier_checks.values()) and winner_check,
    }


def one_search_fit_count(classifier_key: str, inner_n_splits: int = 3) -> int:
    """Calculate fits for one historical GridSearchCV search."""
    return HISTORICAL_CANDIDATE_COUNTS[classifier_key] * inner_n_splits + 1


def fit_count_accounting(selected_classifier: str) -> dict[str, Any]:
    """Return fixed outer and final baseline fit-count accounting."""
    inverse_names = {name: key for key, name in CLASSIFIER_NAMES.items()}
    try:
        selected_key = inverse_names[selected_classifier]
    except KeyError as exc:
        raise CalibrationError(
            f"Unsupported selected classifier for fit-count accounting: {selected_classifier}."
        ) from exc
    outer = {
        key: CV_CONTRACT["n_splits"]
        * CV_CONTRACT["n_repeats"]
        * one_search_fit_count(key, CV_CONTRACT["inner_n_splits"])
        for key in HISTORICAL_CANDIDATE_COUNTS
    }
    final = {
        key: one_search_fit_count(key, CV_CONTRACT["inner_n_splits"])
        for key in HISTORICAL_CANDIDATE_COUNTS
    }
    outer_total = sum(outer.values())
    return {
        "definition": FIT_COUNT_DEFINITION,
        "historical_grid_candidate_counts": HISTORICAL_CANDIDATE_COUNTS,
        "outer_search_fit_count": outer,
        "outer_search_total_fit_count": outer_total,
        "possible_final_search_fit_count": final,
        "selected_final_search_fit_count": final[selected_key],
        "expected_total_fit_count": outer_total + final[selected_key],
    }


def fold_means_from_csv(
    folds_path: Path, expected_classifiers: tuple[str, ...]
) -> dict[str, float]:
    """Recompute classifier means from persisted outer-fold evidence."""
    grouped: dict[str, list[float]] = {name: [] for name in expected_classifiers}
    with open(folds_path, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            classifier_name = row["classifier_name"]
            if classifier_name in grouped:
                grouped[classifier_name].append(float(row["f1_macro"]))
    expected_fold_count = CV_CONTRACT["n_splits"] * CV_CONTRACT["n_repeats"]
    for classifier_name, scores in grouped.items():
        if len(scores) != expected_fold_count:
            raise CalibrationError(
                f"Expected {expected_fold_count} outer folds for {classifier_name}; "
                f"found {len(scores)}."
            )
    return {name: sum(scores) / len(scores) for name, scores in grouped.items()}


def predefined_reopen_axis() -> dict[str, Any]:
    """Return the documented, deliberately unapplied reopen axis."""
    return {
        "name": "n_informative",
        "candidates": list(REOPEN_N_INFORMATIVE_CANDIDATES),
        "applied": False,
    }


def require_effective_config(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    """Fail when installed baseline configuration differs from the protocol."""
    if actual != expected:
        raise CalibrationError(
            f"Effective v0.2.5 configuration mismatch: expected {expected!r}; "
            f"got {actual!r}."
        )


def serialize_report(report: dict[str, Any]) -> str:
    """Serialize a report deterministically and reject non-finite values."""
    return json.dumps(
        report,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )


def write_report(report: dict[str, Any], path: Path = REPORT_PATH) -> None:
    """Write one deterministic JSON report with one trailing newline."""
    payload = serialize_report(report)
    path.write_text(f"{payload}\n", encoding="utf-8", newline="\n")


def validate_baseline_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """Validate the frozen baseline evidence required by candidate mode."""
    try:
        status = evidence["calibration_status"]
        package_version = evidence["baseline"]["package_version"]
        selected_profile = evidence["selected_profile"]
        class_sep = selected_profile["class_sep"]
        dataset_sha256 = selected_profile["dataset_sha256"]
        stage_2 = evidence["stage_2"]
        means = stage_2["classifier_mean_outer_f1_macro"]
        selected_classifier = stage_2["selected_classifier"]
        cv = evidence["cv"]
    except (KeyError, TypeError) as exc:
        raise CalibrationError(
            f"Baseline evidence is missing required field {exc!s}."
        ) from exc

    if status != "passed":
        raise CalibrationError(
            f"Baseline calibration_status must be 'passed'; got {status!r}."
        )
    if package_version != "0.2.5":
        raise CalibrationError(
            f"Baseline package_version must be '0.2.5'; got {package_version!r}."
        )
    if class_sep != FROZEN_CLASS_SEP:
        raise CalibrationError(
            f"Baseline class_sep must be {FROZEN_CLASS_SEP}; got {class_sep!r}."
        )
    if dataset_sha256 != FROZEN_DATASET_SHA256:
        raise CalibrationError(
            "Baseline selected dataset SHA-256 mismatch: expected "
            f"{FROZEN_DATASET_SHA256}; got {dataset_sha256!r}."
        )
    expected_names = set(CLASSIFIER_NAMES.values())
    if not isinstance(means, dict) or set(means) != expected_names:
        raise CalibrationError(
            "Baseline Stage 2 means must contain exactly SVC, "
            "RandomForestClassifier, and XGBClassifier."
        )
    if selected_classifier not in means:
        raise CalibrationError(
            "Baseline selected classifier must exist in the Stage 2 means; "
            f"got {selected_classifier!r}."
        )
    if cv != CV_CONTRACT:
        raise CalibrationError(
            f"Baseline CV mismatch: expected {CV_CONTRACT!r}; got {cv!r}."
        )
    return evidence


def validate_committed_baseline_bytes(
    worktree_bytes: bytes,
    index_bytes: bytes,
    head_bytes: bytes,
    *,
    git_commit: str,
    git_blob: str,
) -> dict[str, Any]:
    """Require worktree, index, and HEAD to contain one baseline artifact."""
    if index_bytes != head_bytes:
        raise CalibrationError(
            "B5_calibration.json staged content differs from the evidence committed "
            "at HEAD."
        )
    if worktree_bytes != head_bytes:
        raise CalibrationError(
            "B5_calibration.json working-tree content differs from the evidence "
            "committed at HEAD."
        )
    return {
        "sha256": sha256_bytes(head_bytes),
        "git_blob": git_blob,
        "git_commit": git_commit,
        "passed": True,
    }


def _git_bytes(arguments: list[str], repo_root: Path) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        diagnostic = completed.stderr.decode("utf-8", errors="replace").strip()
        raise CalibrationError(
            f"Unable to verify committed B5_calibration.json: {diagnostic}."
        )
    return completed.stdout


def committed_baseline_identity(
    path: Path = REPORT_PATH, repo_root: Path = REPO_ROOT
) -> dict[str, Any]:
    """Prove that baseline evidence equals the index and the blob at HEAD."""
    if not path.is_file():
        raise CalibrationError("Committed B5_calibration.json is missing.")
    try:
        relative_path = path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise CalibrationError(
            "B5_calibration.json must be located inside the repository root."
        ) from exc

    head_bytes = _git_bytes(["show", f"HEAD:{relative_path}"], repo_root)
    index_bytes = _git_bytes(["show", f":{relative_path}"], repo_root)
    git_commit = _git_bytes(["rev-parse", "HEAD"], repo_root).decode().strip()
    git_blob = (
        _git_bytes(["rev-parse", f"HEAD:{relative_path}"], repo_root).decode().strip()
    )
    return validate_committed_baseline_bytes(
        path.read_bytes(),
        index_bytes,
        head_bytes,
        git_commit=git_commit,
        git_blob=git_blob,
    )


def load_baseline_evidence(path: Path = REPORT_PATH) -> dict[str, Any]:
    """Load and validate repository baseline evidence without repairing it."""
    if not path.is_file():
        raise CalibrationError(f"Baseline evidence file does not exist: {path}.")
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CalibrationError(f"Baseline evidence is invalid JSON: {exc}.") from exc
    if not isinstance(evidence, dict):
        raise CalibrationError("Baseline evidence must be a JSON object.")
    return validate_baseline_evidence(evidence)


def validate_python_version(
    expected: str,
    actual: str | None = None,
    executable: str | None = None,
) -> dict[str, Any]:
    """Require an exact orchestrator Python-version match."""
    actual = platform.python_version() if actual is None else actual
    executable = sys.executable if executable is None else executable
    if actual != expected:
        raise CalibrationError(
            "Orchestrator Python version mismatch: "
            f"expected {expected}; actual {actual}; executable {executable}."
        )
    return {"expected": expected, "actual": actual, "passed": True}


def installed_distribution_versions(
    distributions: tuple[str, ...] = COMMON_DISTRIBUTIONS,
) -> dict[str, str]:
    """Read installed distribution versions in the orchestrator environment."""
    return {name: importlib.metadata.version(name) for name in distributions}


def validate_dependency_versions(
    expected: dict[str, str], actual: dict[str, str], environment: str
) -> dict[str, Any]:
    """Require exact versions for the seven common scientific distributions."""
    expected_common = {name: expected.get(name) for name in COMMON_DISTRIBUTIONS}
    actual_common = {name: actual.get(name) for name in COMMON_DISTRIBUTIONS}
    if expected_common != actual_common:
        mismatches = {
            name: {"expected": expected_common[name], "actual": actual_common[name]}
            for name in COMMON_DISTRIBUTIONS
            if expected_common[name] != actual_common[name]
        }
        raise CalibrationError(
            f"{environment} common dependency mismatch: {mismatches!r}."
        )
    return {
        "environment": environment,
        "versions": actual_common,
        "passed": True,
    }


def expected_optuna_version() -> str:
    """Read the candidate-only Optuna pin independently of common dependencies."""
    try:
        return _constraint_versions()["optuna"]
    except KeyError as exc:
        raise CalibrationError("Candidate constraints do not pin Optuna.") from exc


def validate_optuna_version(expected: str, actual: str) -> dict[str, Any]:
    """Require the candidate Optuna runtime to match its exact pin."""
    if actual != expected:
        raise CalibrationError(
            f"Candidate Optuna version mismatch: expected {expected}; got {actual}."
        )
    return {"expected": expected, "actual": actual, "passed": True}


def validate_candidate_dataset(
    baseline: dict[str, Any], payload: bytes | None = None
) -> tuple[bytes, dict[str, Any]]:
    """Regenerate and verify only the frozen selected candidate dataset."""
    selected_profile = baseline["selected_profile"]
    expected_generator = {**GENERATOR_PARAMETERS, "class_sep": FROZEN_CLASS_SEP}
    recorded_generator = selected_profile.get("generator_parameters")
    if recorded_generator != expected_generator:
        raise CalibrationError(
            "Baseline selected generator parameters drifted from the fixed contract: "
            f"expected {expected_generator!r}; got {recorded_generator!r}."
        )
    if payload is None:
        payload = generate_dataset_bytes(FROZEN_CLASS_SEP)
    expected_sha256 = selected_profile["dataset_sha256"]
    actual_sha256 = sha256_bytes(payload)
    if actual_sha256 != expected_sha256:
        raise CalibrationError(
            "Candidate dataset SHA-256 mismatch before MELITE execution: "
            f"expected {expected_sha256}; actual {actual_sha256}."
        )
    return payload, {
        "expected_sha256": expected_sha256,
        "actual_sha256": actual_sha256,
        "passed": True,
    }


def require_fresh_output(output_dir: Path) -> dict[str, Any]:
    """Reject any pre-existing candidate output directory."""
    if output_dir.exists():
        raise CalibrationError(
            f"Candidate output directory must not exist before execution: {output_dir}."
        )
    return {"fresh_before_execution": True}


def candidate_artifact_paths(output_dir: Path) -> dict[str, Path]:
    """Resolve the five candidate artifacts only below the fresh output root."""
    return {name: output_dir / name for name in CANDIDATE_ARTIFACT_NAMES}


def expected_search_spaces() -> dict[str, Any]:
    """Return the current B1 contracts for the three candidate classifiers."""
    spaces = {}
    for key in CANDIDATE_CLASSIFIERS:
        search_space = get_search_space(key)
        if search_space is None:
            raise CalibrationError(f"Candidate classifier {key!r} is not tunable.")
        spaces[key] = search_space.to_dict()
    return spaces


def validate_candidate_provenance(
    provenance: dict[str, Any], optuna_version: str
) -> dict[str, Any]:
    """Validate the complete B4 provenance contract for candidate execution."""
    expected = {
        "melite_version": CANDIDATE_VERSION,
        "optimization_backend": {"name": "optuna", "version": optuna_version},
        "smoke": False,
        "random_state": RANDOM_STATE,
        "active_classifiers": list(CANDIDATE_CLASSIFIERS),
        "cv": CV_CONTRACT,
        "optimization": {
            "effective_n_trials": CANDIDATE_N_TRIALS,
            "policy": asdict(OPTIMIZATION_POLICY),
        },
        "search_spaces": expected_search_spaces(),
    }
    if provenance != expected:
        raise CalibrationError(
            f"Candidate optimization provenance mismatch: expected {expected!r}; "
            f"got {provenance!r}."
        )
    return {
        "melite_version": True,
        "backend": True,
        "smoke": True,
        "random_state": True,
        "active_classifiers": True,
        "cv": True,
        "optimization": True,
        "search_spaces": True,
        "overall_pass": True,
    }


def _csv_int(row: dict[str, str], field: str) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationError(
            f"Optimization search field {field!r} must be an integer."
        ) from exc


def validate_optimization_searches(
    rows: list[dict[str, str]], selected_classifier: str
) -> dict[str, Any]:
    """Validate B4 search scope, split coverage, and E1/E2 trial gates."""
    outer_rows = [row for row in rows if row.get("search_scope") == "outer"]
    final_rows = [row for row in rows if row.get("search_scope") == "final"]
    if len(rows) != 16 or len(outer_rows) != 15 or len(final_rows) != 1:
        raise CalibrationError(
            "Candidate optimization searches must contain exactly 15 outer rows "
            f"and 1 final row; found {len(outer_rows)} outer, "
            f"{len(final_rows)} final, {len(rows)} total."
        )
    if any(row.get("search_scope") not in {"outer", "final"} for row in rows):
        raise CalibrationError("Candidate optimization search_scope is invalid.")

    for classifier_name in CLASSIFIER_NAMES.values():
        classifier_rows = [
            row for row in outer_rows if row.get("classifier_name") == classifier_name
        ]
        splits = sorted(_csv_int(row, "outer_split") for row in classifier_rows)
        if splits != list(range(CV_CONTRACT["n_splits"])):
            raise CalibrationError(
                f"Incomplete outer split coverage for {classifier_name}: {splits!r}."
            )
        for row in classifier_rows:
            outer_split = _csv_int(row, "outer_split")
            if _csv_int(row, "outer_repeat") != outer_split // CV_CONTRACT["n_splits"]:
                raise CalibrationError("Outer repeat indexing is inconsistent.")
            if _csv_int(row, "outer_fold") != outer_split % CV_CONTRACT["n_splits"]:
                raise CalibrationError("Outer fold indexing is inconsistent.")
            if row.get("selected") not in {"True", "False"}:
                raise CalibrationError("Outer selected must be True or False.")
            expected_selected = str(classifier_name == selected_classifier)
            if row["selected"] != expected_selected:
                raise CalibrationError(
                    "Outer selected flags must match the dataset-level winner."
                )
            if row.get("smoke") != "False":
                raise CalibrationError("Candidate optimization rows must be non-smoke.")

    final_row = final_rows[0]
    if final_row.get("classifier_name") != selected_classifier:
        raise CalibrationError(
            "Final optimization search must belong to the selected classifier."
        )
    if any(
        final_row.get(field) not in {None, ""}
        for field in (
            "outer_split",
            "outer_repeat",
            "outer_fold",
            "selected",
        )
    ):
        raise CalibrationError("Final optimization search outer fields must be empty.")
    if final_row.get("smoke") != "False":
        raise CalibrationError("Candidate optimization rows must be non-smoke.")

    for row in rows:
        requested = _csv_int(row, "n_trials_requested")
        complete = _csv_int(row, "n_trials_complete")
        failed = _csv_int(row, "n_trials_failed")
        if requested != CANDIDATE_N_TRIALS or complete + failed != requested:
            raise CalibrationError(
                "E1 trial-budget accounting failed: complete + failed must equal "
                f"requested={CANDIDATE_N_TRIALS}."
            )
        if failed != 0:
            raise CalibrationError(
                f"E2 trial-health gate failed: n_trials_failed={failed}."
            )
    return {
        "outer_searches": len(outer_rows),
        "final_searches": len(final_rows),
        "total_searches": len(rows),
        "complete_outer_split_coverage": True,
        "e1_budget_accounting": True,
        "e2_trial_health": True,
        "n_trials_pruned_required": False,
        "overall_pass": True,
    }


def candidate_fit_count_accounting() -> dict[str, Any]:
    """Return the fixed candidate estimator-fit workload estimate."""
    per_search = CANDIDATE_N_TRIALS * CV_CONTRACT["inner_n_splits"] + 1
    outer_searches = len(CANDIDATE_CLASSIFIERS) * CV_CONTRACT["n_splits"]
    outer_total = outer_searches * per_search
    return {
        "definition": (
            "fit_count means estimated estimator.fit() invocations, including one "
            "best-model refit per Optuna search."
        ),
        "per_search_fit_count": per_search,
        "outer_search_fit_count": outer_total,
        "final_search_fit_count": per_search,
        "expected_total_fit_count": outer_total + per_search,
    }


def scientific_comparison(
    baseline_means: dict[str, float],
    candidate_means: dict[str, float],
    baseline_selected_classifier: str,
    candidate_selected_classifier: str,
) -> dict[str, Any]:
    """Compare candidate means with the inclusive fixed per-classifier margin."""
    expected_names = set(CLASSIFIER_NAMES.values())
    if set(baseline_means) != expected_names or set(candidate_means) != expected_names:
        raise CalibrationError(
            "Scientific comparison requires exactly three classifiers."
        )
    classifiers = {}
    for classifier_name in CLASSIFIER_NAMES.values():
        baseline_mean = baseline_means[classifier_name]
        candidate_mean = candidate_means[classifier_name]
        delta = candidate_mean - baseline_mean
        passed = delta >= SCIENTIFIC_MARGIN or math.isclose(
            delta, SCIENTIFIC_MARGIN, rel_tol=0.0, abs_tol=1e-12
        )
        classifiers[classifier_name] = {
            "baseline_mean_outer_f1_macro": baseline_mean,
            "candidate_mean_outer_f1_macro": candidate_mean,
            "delta": delta,
            "passed": passed,
        }
    worst_name = min(
        CLASSIFIER_NAMES.values(), key=lambda name: classifiers[name]["delta"]
    )
    return {
        "margin": SCIENTIFIC_MARGIN,
        "classifiers": classifiers,
        "baseline_selected_classifier": baseline_selected_classifier,
        "candidate_selected_classifier": candidate_selected_classifier,
        "selected_classifier_changed": (
            baseline_selected_classifier != candidate_selected_classifier
        ),
        "worst_classifier_delta": {
            "classifier": worst_name,
            "delta": classifiers[worst_name]["delta"],
        },
        "overall_pass": all(item["passed"] for item in classifiers.values()),
    }


def validate_portable_success_report(report: dict[str, Any]) -> None:
    """Reject local paths and diagnostic-only identities in a success report."""
    forbidden_keys = {"hostname", "timestamp", "study", "trial", "raw_environment"}

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            bad_keys = forbidden_keys.intersection(value)
            if bad_keys:
                raise CalibrationError(
                    f"Success report contains forbidden keys: {sorted(bad_keys)}."
                )
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str) and (
            PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute()
        ):
            raise CalibrationError(
                "Success report must not contain local absolute paths."
            )

    walk(report)


def tee_subprocess(
    command: list[str | Path],
    *,
    cwd: Path,
    env: dict[str, str] | None,
    log_path: Path,
) -> None:
    """Tee combined subprocess output to the terminal and a diagnostic log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8", newline="\n") as log_file:
        process = subprocess.Popen(
            [str(part) for part in command],
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise CalibrationError("Candidate subprocess output pipe was unavailable.")
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(
            return_code, [str(part) for part in command]
        )


def _run(
    command: list[str | Path],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    capture: bool = False,
) -> str:
    printable = " ".join(str(part) for part in command)
    print(f"[B5 calibration] {printable}", flush=True)
    completed = subprocess.run(
        [str(part) for part in command],
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        capture_output=capture,
    )
    return completed.stdout.strip() if capture else ""


def _venv_python(venv_dir: Path) -> Path:
    return (
        venv_dir / "Scripts" / "python.exe"
        if os.name == "nt"
        else venv_dir / "bin" / "python"
    )


def _venv_melite(venv_dir: Path) -> Path:
    return (
        venv_dir / "Scripts" / "melite.exe"
        if os.name == "nt"
        else venv_dir / "bin" / "melite"
    )


def _constraint_versions() -> dict[str, str]:
    versions = {}
    for raw_line in CONSTRAINTS_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            name, version = line.split("==", 1)
            versions[name] = version
    return versions


def _environment_info(python: Path, cwd: Path, env: dict[str, str]) -> dict[str, Any]:
    code = f"""
import importlib.metadata
import json
import platform
import sys

names = {list(COMMON_DISTRIBUTIONS)!r}
print(json.dumps({{
    "python_version": platform.python_version(),
    "python_implementation": platform.python_implementation(),
    "platform": platform.platform(),
    "packages": {{name: importlib.metadata.version(name) for name in names}},
    "melite_version": importlib.metadata.version("melite"),
}}, sort_keys=True))
"""
    return json.loads(_run([python, "-c", code], cwd=cwd, env=env, capture=True))


def _candidate_environment_info(
    python: Path, cwd: Path, env: dict[str, str]
) -> dict[str, Any]:
    code = f"""
import importlib.metadata
import json
import platform

common = {list(COMMON_DISTRIBUTIONS)!r}
print(json.dumps({{
    "python_version": platform.python_version(),
    "packages": {{name: importlib.metadata.version(name) for name in common}},
    "optuna_version": importlib.metadata.version("optuna"),
    "melite_version": importlib.metadata.version("melite"),
}}, sort_keys=True))
"""
    return json.loads(_run([python, "-c", code], cwd=cwd, env=env, capture=True))


def _write_config(
    run_dir: Path, dataset_path: Path, active_classifiers: tuple[str, ...]
) -> Path:
    output_dir = run_dir / "output"
    input_dir = run_dir / "raw"
    data_dir = run_dir / "data"
    active = json.dumps(list(active_classifiers), separators=(",", ":"))
    config = f"""
[paths]
input = {json.dumps(input_dir.as_posix())}
dataset = {json.dumps(data_dir.as_posix())}
output = {json.dumps(output_dir.as_posix())}

[benchmark]
random_state = {RANDOM_STATE}

[cv]
n_splits = {CV_CONTRACT["n_splits"]}
n_repeats = {CV_CONTRACT["n_repeats"]}
inner_n_splits = {CV_CONTRACT["inner_n_splits"]}

[classifiers]
active = {active}

[datasets.{DATASET_ID}]
path = {json.dumps(dataset_path.as_posix())}
label_column = "label"
family = "calibration"
method = "synthetic"
description = "B5 baseline calibration dataset"
""".lstrip()
    run_dir.mkdir(parents=True)
    config_path = run_dir / "config.toml"
    config_path.write_text(config, encoding="utf-8", newline="\n")
    return config_path


def _write_candidate_config(run_dir: Path, dataset_path: Path) -> Path:
    output_dir = run_dir / "output"
    input_dir = run_dir / "raw"
    data_dir = run_dir / "data"
    active = json.dumps(list(CANDIDATE_CLASSIFIERS), separators=(",", ":"))
    config = f"""
[paths]
input = {json.dumps(input_dir.as_posix())}
dataset = {json.dumps(data_dir.as_posix())}
output = {json.dumps(output_dir.as_posix())}

[benchmark]
random_state = {RANDOM_STATE}

[cv]
n_splits = {CV_CONTRACT["n_splits"]}
n_repeats = {CV_CONTRACT["n_repeats"]}
inner_n_splits = {CV_CONTRACT["inner_n_splits"]}

[optimization]
n_trials = {CANDIDATE_N_TRIALS}

[classifiers]
active = {active}

[datasets.{DATASET_ID}]
path = {json.dumps(dataset_path.as_posix())}
label_column = "label"
family = "characterization"
method = "synthetic"
description = "B5 candidate characterization dataset"
""".lstrip()
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.toml"
    config_path.write_text(config, encoding="utf-8", newline="\n")
    return config_path


def _effective_config(
    python: Path,
    config_path: Path,
    active_classifiers: tuple[str, ...],
    dataset_path: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    code = f"""
import json
from pathlib import Path
from melite.config import Config

cfg = Config(user_config=Path({str(config_path)!r}))
dataset = cfg.DATASETS[{DATASET_ID!r}]
print(json.dumps({{
    "active_classifiers": cfg.ACTIVE_CLASSIFIERS,
    "random_state": cfg.RANDOM_STATE,
    "cv": cfg.CV_CONFIG,
    "dataset": {{
        "id": {DATASET_ID!r},
        "path": str(Path(dataset["path"]).resolve()),
        "label_column": dataset["label_column"],
    }},
}}, sort_keys=True))
"""
    actual = json.loads(
        _run([python, "-c", code], cwd=config_path.parent, env=env, capture=True)
    )
    expected = {
        "active_classifiers": list(active_classifiers),
        "random_state": RANDOM_STATE,
        "cv": CV_CONTRACT,
        "dataset": {
            "id": DATASET_ID,
            "path": str(dataset_path.resolve()),
            "label_column": "label",
        },
    }
    require_effective_config(actual, expected)
    return {
        **actual,
        "dataset": {
            "id": actual["dataset"]["id"],
            "label_column": actual["dataset"]["label_column"],
            "path_verified": True,
        },
    }


def _baseline_run(
    melite: Path,
    python: Path,
    run_dir: Path,
    dataset_path: Path,
    active_classifiers: tuple[str, ...],
    env: dict[str, str],
) -> tuple[float, dict[str, Any]]:
    config_path = _write_config(run_dir, dataset_path, active_classifiers)
    effective = _effective_config(
        python, config_path, active_classifiers, dataset_path, env
    )
    started = time.perf_counter()
    _run([melite, "run", "--config", config_path], cwd=run_dir, env=env)
    elapsed = time.perf_counter() - started
    return elapsed, effective


def _selected_classifier(results_path: Path) -> str:
    with open(results_path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 1:
        raise CalibrationError(f"Expected one selected result row; found {len(rows)}.")
    if rows[0]["smoke"] != "False":
        raise CalibrationError("Baseline calibration unexpectedly ran in smoke mode.")
    return rows[0]["classifier_name"]


def load_candidate_outputs(
    output_dir: Path,
    baseline: dict[str, Any],
    optuna_version: str,
) -> dict[str, Any]:
    """Parse and validate artifacts only from the exact fresh output directory."""
    paths = candidate_artifact_paths(output_dir)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise CalibrationError(f"Candidate output is missing artifacts: {missing!r}.")

    candidate_selected = _selected_classifier(paths["results.csv"])
    expected_names = tuple(CLASSIFIER_NAMES.values())
    candidate_means = fold_means_from_csv(paths["evaluation_folds.csv"], expected_names)
    provenance = json.loads(
        paths["optimization_provenance.json"].read_text(encoding="utf-8")
    )
    provenance_gates = validate_candidate_provenance(provenance, optuna_version)
    with open(paths["optimization_searches.csv"], newline="", encoding="utf-8") as file:
        search_rows = list(csv.DictReader(file))
    search_gates = validate_optimization_searches(search_rows, candidate_selected)
    baseline_stage_2 = baseline["stage_2"]
    comparison = scientific_comparison(
        baseline_stage_2["classifier_mean_outer_f1_macro"],
        candidate_means,
        baseline_stage_2["selected_classifier"],
        candidate_selected,
    )
    return {
        "candidate_means": candidate_means,
        "candidate_selected_classifier": candidate_selected,
        "provenance_gates": provenance_gates,
        "search_gates": search_gates,
        "scientific_comparison": comparison,
    }


def _initial_report(source_commit: str) -> dict[str, Any]:
    constraints_bytes = CONSTRAINTS_PATH.read_bytes()
    return {
        "calibration_status": "failed",
        "baseline": {
            "tag": BASELINE_TAG,
            "source_commit": source_commit,
            "package_version": None,
        },
        "comparison_scope": (
            "Historical MELITE v0.2.5 GridSearchCV engine on the current pinned "
            "characterization stack; this is not a reproduction of the historical "
            "v0.2.5 software environment."
        ),
        "environment": {
            "orchestrator_python_version": platform.python_version(),
            "orchestrator_platform": platform.platform(),
        },
        "constraints": {
            "file": "scripts/characterize_v030_constraints.txt",
            "sha256": sha256_bytes(constraints_bytes),
        },
        "generator": GENERATOR_PARAMETERS,
        "class_sep_candidates": list(CLASS_SEP_CANDIDATES),
        "cv": CV_CONTRACT,
        "random_state": RANDOM_STATE,
        "stage_1": [],
        "selected_profile": None,
        "stage_2": None,
        "acceptance": None,
        "predefined_reopen_axis": predefined_reopen_axis(),
    }


def _initial_candidate_report() -> dict[str, Any]:
    return {
        "characterization_status": "failed",
        "overall_pass": False,
        "candidate": {"source_commit": None, "package_version": None},
        "baseline_evidence": None,
        "python_check": None,
        "common_dependency_checks": None,
        "candidate_optuna": None,
        "dataset": None,
        "class_sep": FROZEN_CLASS_SEP,
        "cv": CV_CONTRACT,
        "active_classifiers": list(CANDIDATE_CLASSIFIERS),
        "effective_n_trials": CANDIDATE_N_TRIALS,
        "provenance_gates": None,
        "optimization_search_gates": None,
        "workload": None,
        "wall_clock": None,
        "scientific_comparison": None,
    }


def calibrate() -> bool:
    """Run the fixed baseline-only v0.2.5 calibration protocol."""
    source_commit = _run(
        ["git", "rev-list", "-n", "1", BASELINE_TAG],
        cwd=REPO_ROOT,
        capture=True,
    )
    report = _initial_report(source_commit)
    temp_root = Path(tempfile.mkdtemp(prefix="melite-b5-calibration-")).resolve()
    worktree_dir = temp_root / "baseline-source"
    worktree_added = False
    try:
        _run(
            ["git", "worktree", "add", "--detach", worktree_dir, BASELINE_TAG],
            cwd=REPO_ROOT,
        )
        worktree_added = True
        dist_dir = temp_root / "dist"
        _run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                dist_dir,
            ],
            cwd=worktree_dir,
        )
        wheels = sorted(dist_dir.glob("melite-*.whl"))
        if len(wheels) != 1:
            raise CalibrationError(f"Expected one baseline wheel; found {wheels}.")

        venv_dir = temp_root / "baseline-venv"
        _run([sys.executable, "-m", "venv", venv_dir], cwd=temp_root)
        python = _venv_python(venv_dir)
        melite = _venv_melite(venv_dir)
        _run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--constraint",
                CONSTRAINTS_PATH,
                wheels[0],
            ],
            cwd=temp_root,
        )
        isolated_env = os.environ.copy()
        isolated_env.pop("PYTHONPATH", None)

        installed = _environment_info(python, temp_root, isolated_env)
        expected_versions = _constraint_versions()
        actual_common = installed["packages"]
        expected_common = {
            name: expected_versions[name] for name in COMMON_DISTRIBUTIONS
        }
        if actual_common != expected_common:
            raise CalibrationError(
                f"Installed dependency mismatch: expected {expected_common!r}; "
                f"got {actual_common!r}."
            )
        if installed["melite_version"] != "0.2.5":
            raise CalibrationError(
                f"Expected installed MELITE 0.2.5; got {installed['melite_version']}."
            )
        report["baseline"]["package_version"] = installed["melite_version"]
        report["environment"].update(
            {
                "python_version": installed["python_version"],
                "python_implementation": installed["python_implementation"],
                "platform": installed["platform"],
                "installed_packages": actual_common,
            }
        )

        profiles_dir = temp_root / "profiles"
        profiles_dir.mkdir()
        profile_paths = {}
        for index, class_sep in enumerate(CLASS_SEP_CANDIDATES):
            payload = generate_dataset_bytes(class_sep)
            dataset_path = profiles_dir / f"class_sep_{class_sep:.2f}.csv"
            dataset_path.write_bytes(payload)
            profile_paths[class_sep] = dataset_path
            run_dir = temp_root / "runs" / f"stage_1_{index}"
            print(
                f"[B5 calibration] Stage 1 class_sep={class_sep:.2f}",
                flush=True,
            )
            wall_seconds, effective = _baseline_run(
                melite,
                python,
                run_dir,
                dataset_path,
                ("svc",),
                isolated_env,
            )
            mean = fold_means_from_csv(
                run_dir / "output" / "evaluation_folds.csv",
                (CLASSIFIER_NAMES["svc"],),
            )[CLASSIFIER_NAMES["svc"]]
            report["stage_1"].append(
                {
                    "class_sep": class_sep,
                    "dataset_sha256": sha256_bytes(payload),
                    "svc_mean_outer_f1_macro": mean,
                    "end_to_end_wall_seconds": wall_seconds,
                    "eligible": svc_profile_is_eligible(mean),
                    "verified_effective_config": effective,
                }
            )

        selected = select_svc_profile(report["stage_1"])
        selected_class_sep = selected["class_sep"]
        selected_dataset_path = profile_paths[selected_class_sep]
        selected_generator = {**GENERATOR_PARAMETERS, "class_sep": selected_class_sep}
        report["selected_profile"] = {
            "class_sep": selected_class_sep,
            "generator_parameters": selected_generator,
            "dataset_sha256": selected["dataset_sha256"],
            "selection_rationale": {
                "rule": "eligible SVC mean closest to 0.80; exact tie uses lower class_sep",
                "target_svc_mean_outer_f1_macro": 0.80,
                "selected_svc_mean_outer_f1_macro": selected["svc_mean_outer_f1_macro"],
                "absolute_distance": abs(selected["svc_mean_outer_f1_macro"] - 0.80),
            },
        }

        stage_2_dir = temp_root / "runs" / "stage_2"
        print(
            f"[B5 calibration] Stage 2 selected class_sep={selected_class_sep:.2f}",
            flush=True,
        )
        wall_seconds, effective = _baseline_run(
            melite,
            python,
            stage_2_dir,
            selected_dataset_path,
            ("svc", "rf", "xgb"),
            isolated_env,
        )
        expected_names = tuple(CLASSIFIER_NAMES.values())
        classifier_means = fold_means_from_csv(
            stage_2_dir / "output" / "evaluation_folds.csv",
            expected_names,
        )
        selected_classifier = _selected_classifier(
            stage_2_dir / "output" / "results.csv"
        )
        acceptance = acceptance_result(classifier_means, selected_classifier)
        report["stage_2"] = {
            "classifier_mean_outer_f1_macro": classifier_means,
            "selected_classifier": selected_classifier,
            "selected_classifier_mean_outer_f1_macro": classifier_means[
                selected_classifier
            ],
            "end_to_end_wall_seconds": wall_seconds,
            "verified_effective_config": effective,
            "fit_count_accounting": fit_count_accounting(selected_classifier),
        }
        report["acceptance"] = acceptance
        if not acceptance["overall_pass"]:
            raise CalibrationError(
                "Selected Stage 2 profile failed the fixed calibration acceptance."
            )
        report["calibration_status"] = "passed"
        write_report(report)
        return True
    except Exception as exc:  # noqa: BLE001 - preserve calibration failure report
        report["calibration_status"] = "failed"
        report["failure_reason"] = f"{type(exc).__name__}: {exc}"
        write_report(report)
        print(f"[B5 calibration] FAILED: {exc}", file=sys.stderr, flush=True)
        return False
    finally:
        if worktree_added:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree_dir)],
                cwd=REPO_ROOT,
                check=False,
            )
        shutil.rmtree(temp_root, ignore_errors=True)


def candidate(
    report_path: Path = CANDIDATE_REPORT_PATH,
    console_log_path: Path = CANDIDATE_CONSOLE_LOG_PATH,
) -> bool:
    """Run the frozen v0.3.0 candidate characterization protocol."""
    report = _initial_candidate_report()
    baseline_bytes: bytes | None = None
    temp_root: Path | None = None
    worktree_dir: Path | None = None
    worktree_added = False
    try:
        evidence_identity = committed_baseline_identity()
        baseline_bytes = REPORT_PATH.read_bytes()
        baseline = load_baseline_evidence()
        report["baseline_evidence"] = {
            "tag": baseline["baseline"].get("tag"),
            "calibration_source_commit": baseline["baseline"].get("source_commit"),
            "package_version": baseline["baseline"]["package_version"],
            "dataset_sha256": baseline["selected_profile"]["dataset_sha256"],
            "committed_evidence": evidence_identity,
        }

        expected_python = baseline["environment"]["orchestrator_python_version"]
        orchestrator_python = validate_python_version(expected_python)
        report["python_check"] = {"orchestrator": orchestrator_python}
        expected_common = baseline["environment"]["installed_packages"]
        orchestrator_common = validate_dependency_versions(
            expected_common,
            installed_distribution_versions(),
            "orchestrator",
        )

        dataset_payload, dataset_gate = validate_candidate_dataset(baseline)
        report["dataset"] = dataset_gate
        pinned_optuna = expected_optuna_version()

        source_commit = _run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture=True)
        report["candidate"]["source_commit"] = source_commit

        temp_root = Path(tempfile.mkdtemp(prefix="melite-b5-candidate-")).resolve()
        worktree_dir = temp_root / "candidate-source"
        _run(
            ["git", "worktree", "add", "--detach", worktree_dir, source_commit],
            cwd=REPO_ROOT,
        )
        worktree_added = True

        dist_dir = temp_root / "dist"
        _run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                dist_dir,
            ],
            cwd=worktree_dir,
        )
        wheels = sorted(dist_dir.glob("melite-*.whl"))
        if len(wheels) != 1:
            raise CalibrationError(f"Expected one candidate wheel; found {wheels}.")

        venv_dir = temp_root / "candidate-venv"
        _run([sys.executable, "-m", "venv", venv_dir], cwd=temp_root)
        python = _venv_python(venv_dir)
        melite = _venv_melite(venv_dir)
        _run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--constraint",
                CONSTRAINTS_PATH,
                wheels[0],
            ],
            cwd=temp_root,
        )
        isolated_env = os.environ.copy()
        isolated_env.pop("PYTHONPATH", None)
        installed = _candidate_environment_info(python, temp_root, isolated_env)
        candidate_python = validate_python_version(
            expected_python, installed["python_version"], str(python)
        )
        report["python_check"]["candidate_venv"] = candidate_python
        candidate_common = validate_dependency_versions(
            expected_common, installed["packages"], "candidate_venv"
        )
        optuna_check = validate_optuna_version(
            pinned_optuna, installed["optuna_version"]
        )
        if installed["melite_version"] != CANDIDATE_VERSION:
            raise CalibrationError(
                f"Expected installed MELITE {CANDIDATE_VERSION}; "
                f"got {installed['melite_version']}."
            )
        report["candidate"]["package_version"] = installed["melite_version"]
        report["candidate_optuna"] = optuna_check
        report["common_dependency_checks"] = {
            "orchestrator": orchestrator_common,
            "candidate_venv": candidate_common,
        }

        run_dir = temp_root / "candidate-run"
        dataset_path = run_dir / "dataset" / "b5_calibration.csv"
        dataset_path.parent.mkdir(parents=True)
        dataset_path.write_bytes(dataset_payload)
        config_path = _write_candidate_config(run_dir, dataset_path)
        output_dir = run_dir / "output"
        require_fresh_output(output_dir)

        started = time.perf_counter()
        tee_subprocess(
            [melite, "run", "--verbose", "--config", config_path],
            cwd=run_dir,
            env=isolated_env,
            log_path=console_log_path,
        )
        candidate_wall_seconds = time.perf_counter() - started
        parsed = load_candidate_outputs(output_dir, baseline, pinned_optuna)

        baseline_wall_seconds = baseline["stage_2"]["end_to_end_wall_seconds"]
        report["provenance_gates"] = parsed["provenance_gates"]
        report["optimization_search_gates"] = parsed["search_gates"]
        report["workload"] = {
            "baseline": {
                "expected_total_fit_count": baseline["stage_2"]["fit_count_accounting"][
                    "expected_total_fit_count"
                ],
                "definition": baseline["stage_2"]["fit_count_accounting"]["definition"],
            },
            "candidate": candidate_fit_count_accounting(),
        }
        report["wall_clock"] = {
            "definition": "complete MELITE subprocess; descriptive, not a gate",
            "baseline_end_to_end_wall_seconds": baseline_wall_seconds,
            "candidate_end_to_end_wall_seconds": candidate_wall_seconds,
            "candidate_to_baseline_ratio": (
                candidate_wall_seconds / baseline_wall_seconds
            ),
        }
        report["scientific_comparison"] = parsed["scientific_comparison"]
        report["overall_pass"] = parsed["scientific_comparison"]["overall_pass"]
        report["characterization_status"] = (
            "passed" if report["overall_pass"] else "failed"
        )
        if baseline_bytes is None or REPORT_PATH.read_bytes() != baseline_bytes:
            raise CalibrationError("Candidate mode modified B5_calibration.json.")
        if report["overall_pass"]:
            validate_portable_success_report(report)
        write_report(report, report_path)
        return bool(report["overall_pass"])
    except Exception as exc:  # noqa: BLE001 - preserve characterization failure report
        report["characterization_status"] = "failed"
        report["overall_pass"] = False
        report["failure_reason"] = f"{type(exc).__name__}: {exc}"
        write_report(report, report_path)
        print(f"[B5 candidate] FAILED: {exc}", file=sys.stderr, flush=True)
        return False
    finally:
        if worktree_added and worktree_dir is not None:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree_dir)],
                cwd=REPO_ROOT,
                check=False,
            )
        if temp_root is not None:
            shutil.rmtree(temp_root, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the development-only characterization command parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("calibrate", "candidate"))
    return parser


def main() -> None:
    """Run the requested calibration or candidate characterization mode."""
    args = build_parser().parse_args()
    succeeded = calibrate() if args.mode == "calibrate" else candidate()
    raise SystemExit(0 if succeeded else 1)


if __name__ == "__main__":
    main()
