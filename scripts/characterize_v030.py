# SPDX-License-Identifier: LGPL-3.0-or-later
"""Baseline-only calibration infrastructure for MELITE v0.3.0 characterization."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.datasets import make_classification


REPO_ROOT = Path(__file__).resolve().parents[1]
CONSTRAINTS_PATH = Path(__file__).with_name("characterize_v030_constraints.txt")
REPORT_PATH = REPO_ROOT / "B5_calibration.json"
BASELINE_TAG = "v0.2.5"
DATASET_ID = "b5_calibration"
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


def build_parser() -> argparse.ArgumentParser:
    """Build the development-only calibration command parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("calibrate",))
    return parser


def main() -> None:
    """Run the requested baseline-only calibration mode."""
    args = build_parser().parse_args()
    if args.mode != "calibrate":
        raise SystemExit(2)
    raise SystemExit(0 if calibrate() else 1)


if __name__ == "__main__":
    main()
