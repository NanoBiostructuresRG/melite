# SPDX-License-Identifier: LGPL-3.0-or-later
"""Installed-wheel smoke test for the public MELITE example workflow.

The script builds a wheel from the current checkout, installs it into a
temporary virtual environment, runs ``melite example`` followed by the public
smoke command, and verifies the copied resources and generated evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_API = [
    "Config",
    "load_datasets",
    "plot_f1_macro_evidence",
    "predict",
    "__version__",
]
REGRESSION_IDENTITY_FIELDS = [
    "dataset",
    "task",
    "family",
    "method",
    "variant",
    "level",
    "description",
    "reduction_type",
    "regressor_key",
    "regressor_name",
]
REGRESSION_CSV_SCHEMAS = {
    "regression_results.csv": [
        *REGRESSION_IDENTITY_FIELDS,
        "parameters",
        "rmse",
        "rmse_std",
        "mae",
        "mae_std",
        "r2",
        "r2_std",
        "smoke",
    ],
    "regression_evaluations.csv": [
        *REGRESSION_IDENTITY_FIELDS,
        "rmse",
        "rmse_std",
        "mae",
        "mae_std",
        "r2",
        "r2_std",
        "selected",
        "smoke",
    ],
    "regression_evaluation_folds.csv": [
        *REGRESSION_IDENTITY_FIELDS,
        "outer_split",
        "outer_repeat",
        "outer_fold",
        "rmse",
        "mae",
        "r2",
        "selected",
        "smoke",
    ],
    "regression_optimization_searches.csv": [
        *REGRESSION_IDENTITY_FIELDS,
        "search_scope",
        "outer_split",
        "outer_repeat",
        "outer_fold",
        "best_inner_rmse",
        "best_params",
        "n_trials_requested",
        "n_trials_complete",
        "n_trials_failed",
        "selected",
        "smoke",
    ],
}
REGRESSION_ARTIFACTS = {
    "regression_results.txt",
    *REGRESSION_CSV_SCHEMAS,
    "regression_optimization_provenance.json",
}


def _run(
    cmd: list[str | Path], *, cwd: Path, env: dict[str, str] | None = None
) -> None:
    display = " ".join(str(part) for part in cmd)
    print(f"[smoke] {display}")
    subprocess.run([str(part) for part in cmd], cwd=cwd, env=env, check=True)


def _run_expect_smoke_block(cmd: list[str | Path], *, cwd: Path) -> None:
    display = " ".join(str(part) for part in cmd)
    print(f"[smoke] expecting failure: {display}")
    completed = subprocess.run(
        [str(part) for part in cmd],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    combined_output = f"{completed.stdout}\n{completed.stderr}".lower()
    if completed.returncode != 1:
        raise AssertionError(
            "Expected regression smoke export to exit with code 1, "
            f"got {completed.returncode}"
        )
    if "smoke mode" not in combined_output or "--force" not in combined_output:
        raise AssertionError(
            "Regression export failed without the expected smoke-mode guard message"
        )


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _venv_melite(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "melite.exe"
    return venv_dir / "bin" / "melite"


def _build_wheel(dist_dir: Path) -> Path:
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
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
        cwd=REPO_ROOT,
    )
    wheels = sorted(dist_dir.glob("melite-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(
            f"Expected exactly one MELITE wheel in {dist_dir}, got {wheels}"
        )
    return wheels[0]


def _check_imports(python: Path) -> None:
    code = f"""
import pathlib
import melite
expected = {EXPECTED_API!r}
assert melite.__all__ == expected, melite.__all__
for name in expected:
    assert hasattr(melite, name), f"{{name}} missing"
assert 'load_dataset' not in melite.__all__
assert 'ResultManager' not in melite.__all__
repo = pathlib.Path({str(REPO_ROOT)!r}).resolve()
module_path = pathlib.Path(melite.__file__).resolve()
assert repo not in module_path.parents, module_path
print(melite.__version__, module_path)
"""
    _run([python, "-c", code], cwd=REPO_ROOT.parent)


def _verify_example_tree(work_dir: Path) -> None:
    example_dir = work_dir / "melite_example"
    expected_tree = {
        "config.toml",
        "data",
        "data/sample_tabular.csv",
    }
    actual_tree = {
        path.relative_to(example_dir).as_posix() for path in example_dir.rglob("*")
    }
    if actual_tree != expected_tree:
        raise AssertionError(
            f"Expected example tree {sorted(expected_tree)}, got {sorted(actual_tree)}"
        )


def _verify_outputs(work_dir: Path) -> None:
    example_dir = work_dir / "melite_example"
    output_dir = example_dir / "output"
    results_csv = output_dir / "results.csv"
    evaluations_csv = output_dir / "evaluations.csv"
    folds_csv = output_dir / "evaluation_folds.csv"
    optimization_csv = output_dir / "optimization_searches.csv"
    provenance_json = output_dir / "optimization_provenance.json"
    model_path = output_dir / "Model_SVC_sample_tabular.pkl"
    figure_path = output_dir / "figures" / "evaluation_f1_macro_sample_tabular.png"

    expected_paths = (
        example_dir / "config.toml",
        example_dir / "data" / "sample_tabular.csv",
        output_dir / "results.txt",
        results_csv,
        evaluations_csv,
        folds_csv,
        optimization_csv,
        provenance_json,
        model_path,
        figure_path,
    )
    for path in expected_paths:
        if not path.exists():
            raise AssertionError(f"Expected artifact was not created: {path}")

    with open(results_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise AssertionError(f"Expected one result row, got {len(rows)}")
    row = rows[0]
    if row["dataset"] != "sample_tabular":
        raise AssertionError(
            f"Expected dataset 'sample_tabular', got {row['dataset']!r}"
        )
    if row["classifier_name"] != "SVC":
        raise AssertionError(
            f"Expected classifier 'SVC', got {row['classifier_name']!r}"
        )

    for csv_path in (results_csv, evaluations_csv, folds_csv):
        with open(csv_path, newline="", encoding="utf-8") as f:
            evidence_rows = list(csv.DictReader(f))
        if not evidence_rows:
            raise AssertionError(f"Expected evidence rows in {csv_path}")
        smoke_values = {evidence_row["smoke"] for evidence_row in evidence_rows}
        if smoke_values != {"True"}:
            raise AssertionError(
                f"Expected smoke=True rows in {csv_path}, got {smoke_values}"
            )

    with open(optimization_csv, newline="", encoding="utf-8") as f:
        optimization_rows = list(csv.DictReader(f))
    if not optimization_rows:
        raise AssertionError("Expected optimization rows for bundled SVC example")
    if not any(row["search_scope"] == "outer" for row in optimization_rows):
        raise AssertionError("Expected at least one outer optimization row")
    final_rows = [row for row in optimization_rows if row["search_scope"] == "final"]
    if len(final_rows) != 1:
        raise AssertionError(f"Expected one final optimization row, got {final_rows}")
    final_row = final_rows[0]
    for field in ("outer_split", "outer_repeat", "outer_fold", "selected"):
        if final_row[field] != "":
            raise AssertionError(
                f"Expected empty final {field}, got {final_row[field]!r}"
            )
    if final_row["smoke"] != "True":
        raise AssertionError(
            f"Expected smoke=True final optimization row, got {final_row['smoke']!r}"
        )

    with open(provenance_json, encoding="utf-8") as f:
        provenance = json.load(f)
    if provenance["smoke"] is not True:
        raise AssertionError("Expected smoke=true optimization provenance")
    if provenance["active_classifiers"] != ["svc"]:
        raise AssertionError(
            "Expected active_classifiers=['svc'] in optimization provenance"
        )
    if provenance["optimization"]["effective_n_trials"] != 5:
        raise AssertionError("Expected effective_n_trials=5 in smoke provenance")
    if provenance["optimization_backend"]["name"] != "optuna":
        raise AssertionError("Expected Optuna optimization backend provenance")


def _write_regression_workspace(work_dir: Path) -> tuple[Path, Path, Path]:
    regression_dir = work_dir / "melite_regression_smoke"
    data_dir = regression_dir / "data"
    output_dir = regression_dir / "output"
    data_dir.mkdir(parents=True)
    dataset_path = data_dir / "ridge_smoke.csv"
    with open(dataset_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_1", "feature_2", "feature_3", "target"])
        for index in range(12):
            feature_1 = (index - 5.5) / 3.0
            feature_2 = ((index * 3) % 7 - 3) / 2.0
            feature_3 = ((index * index) % 11 - 5) / 4.0
            target = (
                4.5 * feature_1
                - 2.0 * feature_2
                + 0.75 * feature_3
                + 1.2
                + ((index % 3) - 1) * 0.05
            )
            writer.writerow([feature_1, feature_2, feature_3, target])

    config_path = regression_dir / "config.toml"
    config_path.write_text(
        '''[paths]
input = "melite_regression_smoke/data/"
dataset = "melite_regression_smoke/data/"
output = "melite_regression_smoke/output/"

[benchmark]
random_state = 42

[datasets.ridge_smoke]
path = "melite_regression_smoke/data/ridge_smoke.csv"
label_column = "target"
task = "regression"
family = "tabular"
description = "Deterministic installed-wheel Ridge regression smoke dataset"

[cv]
n_splits = 2
n_repeats = 1
inner_n_splits = 2

[cv_smoke]
n_splits = 2
n_repeats = 1
inner_n_splits = 2

[classifiers]
active = []

[regressors]
active = ["ridge"]

[optimization]
n_trials = 1
''',
        encoding="utf-8",
    )
    return regression_dir, config_path, output_dir


def _read_regression_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return list(reader.fieldnames or []), rows


def _verify_regression_outputs(output_dir: Path) -> None:
    actual_artifacts = {path.name for path in output_dir.iterdir()}
    if actual_artifacts != REGRESSION_ARTIFACTS:
        raise AssertionError(
            "Regression-only smoke run produced an unexpected artifact set: "
            f"expected {sorted(REGRESSION_ARTIFACTS)}, "
            f"got {sorted(actual_artifacts)}"
        )

    csv_rows = {}
    for filename, expected_schema in REGRESSION_CSV_SCHEMAS.items():
        actual_schema, rows = _read_regression_csv(output_dir / filename)
        if actual_schema != expected_schema:
            raise AssertionError(
                f"Unexpected schema for {filename}: {actual_schema}"
            )
        csv_rows[filename] = rows
        if not rows:
            raise AssertionError(f"Expected evidence rows in {filename}")
        smoke_values = {row["smoke"] for row in rows}
        if smoke_values != {"True"}:
            raise AssertionError(
                f"Expected smoke=True rows in {filename}, got {smoke_values}"
            )

    result_rows = csv_rows["regression_results.csv"]
    if len(result_rows) != 1:
        raise AssertionError(f"Expected one regression result row, got {result_rows}")
    result = result_rows[0]
    expected_identity = {
        "dataset": "ridge_smoke",
        "task": "regression",
        "regressor_key": "ridge",
        "regressor_name": "Ridge",
    }
    for field, expected in expected_identity.items():
        if result[field] != expected:
            raise AssertionError(
                f"Expected regression {field}={expected!r}, got {result[field]!r}"
            )
    parameters = json.loads(result["parameters"])
    if set(parameters) != {"ridge__alpha"}:
        raise AssertionError(f"Unexpected Ridge parameters: {parameters}")
    if not 1e-4 <= parameters["ridge__alpha"] <= 1e2:
        raise AssertionError(f"Ridge alpha is outside its search domain: {parameters}")
    for metric in ("rmse", "mae", "r2"):
        if not math.isfinite(float(result[metric])):
            raise AssertionError(f"Expected finite regression {metric}: {result[metric]}")
    if float(result["rmse"]) < 0.0 or float(result["mae"]) < 0.0:
        raise AssertionError(f"Expected nonnegative regression errors: {result}")

    evaluation_rows = csv_rows["regression_evaluations.csv"]
    if len(evaluation_rows) != 1 or evaluation_rows[0]["selected"] != "True":
        raise AssertionError(
            f"Expected one selected Ridge evaluation, got {evaluation_rows}"
        )
    fold_rows = csv_rows["regression_evaluation_folds.csv"]
    if len(fold_rows) != 2 or {row["outer_fold"] for row in fold_rows} != {"0", "1"}:
        raise AssertionError(f"Expected two outer regression folds, got {fold_rows}")
    for row in (*evaluation_rows, *fold_rows):
        for metric in ("rmse", "mae", "r2"):
            if not math.isfinite(float(row[metric])):
                raise AssertionError(
                    f"Expected finite {metric} in regression evidence: {row}"
                )

    optimization_rows = csv_rows["regression_optimization_searches.csv"]
    if [row["search_scope"] for row in optimization_rows] != [
        "outer",
        "outer",
        "final",
    ]:
        raise AssertionError(
            f"Expected two outer and one final Ridge searches, got {optimization_rows}"
        )
    if any(row["n_trials_requested"] != "5" for row in optimization_rows):
        raise AssertionError(
            "Expected MELITE's five-trial smoke optimization budget for Ridge"
        )
    if json.loads(optimization_rows[-1]["best_params"]) != parameters:
        raise AssertionError(
            "Regression result parameters do not match the final Ridge search"
        )

    with open(
        output_dir / "regression_optimization_provenance.json",
        encoding="utf-8",
    ) as f:
        provenance = json.load(f)
    expected_provenance_fields = {
        "melite_version",
        "task",
        "optimization_backend",
        "smoke",
        "random_state",
        "active_regressors",
        "cv",
        "optimization",
        "search_spaces",
    }
    if set(provenance) != expected_provenance_fields:
        raise AssertionError(
            f"Unexpected regression provenance schema: {sorted(provenance)}"
        )
    if provenance["task"] != "regression" or provenance["smoke"] is not True:
        raise AssertionError(f"Unexpected regression provenance: {provenance}")
    if provenance["active_regressors"] != ["ridge"]:
        raise AssertionError("Expected active_regressors=['ridge'] in provenance")
    if provenance["optimization"]["direction"] != "minimize":
        raise AssertionError("Expected regression optimization to minimize RMSE")
    if provenance["optimization"]["effective_n_trials"] != 5:
        raise AssertionError("Expected effective_n_trials=5 in smoke provenance")

    report = (output_dir / "regression_results.txt").read_text(encoding="utf-8")
    if "Regressor selected: Ridge" not in report:
        raise AssertionError("Expected Ridge identity in regression_results.txt")
    if "Classifier selected:" in report:
        raise AssertionError("Regression report contains classification output")


def _check_regression_prediction(python: Path, model_path: Path) -> None:
    code = f"""
import pathlib
import joblib
import numpy as np
import melite

repo = pathlib.Path({str(REPO_ROOT)!r}).resolve()
module_path = pathlib.Path(melite.__file__).resolve()
assert repo not in module_path.parents, module_path
model_path = pathlib.Path({str(model_path)!r}).resolve()
model = joblib.load(model_path)
X_new = np.array([
    [-1.0, 0.5, -0.25],
    [0.75, -1.0, 1.25],
], dtype=float)
loaded_predictions = model.predict(X_new)
result = melite.predict(model_path, X_new)
assert loaded_predictions.shape == (2,), loaded_predictions.shape
assert np.all(np.isfinite(loaded_predictions)), loaded_predictions
assert result['predictions'].shape == (2,), result['predictions'].shape
assert np.all(np.isfinite(result['predictions'])), result['predictions']
assert np.allclose(result['predictions'], loaded_predictions)
assert result['probabilities'] is None, result['probabilities']
assert result['n_samples'] == 2, result['n_samples']
assert result['model_path'] == str(model_path), result['model_path']
print(type(model).__name__, result['predictions'])
"""
    _run([python, "-c", code], cwd=model_path.parent)


def _run_regression_workflow(melite: Path, python: Path, work_dir: Path) -> None:
    _, config_path, output_dir = _write_regression_workspace(work_dir)
    results_path = output_dir / "regression_results.csv"
    model_path = output_dir / "Model_Ridge_ridge_smoke.pkl"
    _run(
        [melite, "run", "--smoke", "--config", config_path],
        cwd=work_dir,
    )
    _verify_regression_outputs(output_dir)

    export_command = [
        melite,
        "export",
        "--config",
        config_path,
        "--row",
        "0",
        "--csv",
        results_path,
        "--outdir",
        output_dir,
    ]
    _run_expect_smoke_block(export_command, cwd=work_dir)
    if model_path.exists():
        raise AssertionError("Blocked regression smoke export created a model artifact")
    _run([*export_command, "--force"], cwd=work_dir)
    if not model_path.exists():
        raise AssertionError(f"Expected exported Ridge artifact: {model_path}")
    _check_regression_prediction(python, model_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the temporary smoke directory for debugging.",
    )
    args = parser.parse_args()

    temp_root = Path(tempfile.mkdtemp(prefix="melite-wheel-smoke-")).resolve()
    print(f"[smoke] temp root: {temp_root}")
    try:
        wheel = _build_wheel(temp_root / "dist")
        venv_dir = temp_root / "venv"
        work_dir = temp_root / "work"
        work_dir.mkdir()

        _run(
            [sys.executable, "-m", "venv", "--system-site-packages", venv_dir],
            cwd=temp_root,
        )
        python = _venv_python(venv_dir)
        melite = _venv_melite(venv_dir)

        _run([python, "-m", "pip", "install", "--no-deps", wheel], cwd=temp_root)
        _check_imports(python)

        _run([melite, "example"], cwd=work_dir)
        _verify_example_tree(work_dir)
        _run(
            [
                melite,
                "run",
                "--smoke",
                "--config",
                "melite_example/config.toml",
            ],
            cwd=work_dir,
        )
        _run(
            [
                melite,
                "export",
                "--config",
                "melite_example/config.toml",
                "--row",
                "0",
                "--csv",
                "melite_example/output/results.csv",
                "--outdir",
                "melite_example/output",
                "--force",
            ],
            cwd=work_dir,
        )
        _verify_outputs(work_dir)
        print("[smoke] installed-wheel public example workflow passed")
        _run_regression_workflow(melite, python, work_dir)
        print("[smoke] installed-wheel regression workflow passed")
    finally:
        if args.keep_temp:
            print(f"[smoke] kept temp root: {temp_root}")
        else:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
