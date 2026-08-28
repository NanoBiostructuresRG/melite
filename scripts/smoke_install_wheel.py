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


def _run(
    cmd: list[str | Path], *, cwd: Path, env: dict[str, str] | None = None
) -> None:
    display = " ".join(str(part) for part in cmd)
    print(f"[smoke] {display}")
    subprocess.run([str(part) for part in cmd], cwd=cwd, env=env, check=True)


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
    finally:
        if args.keep_temp:
            print(f"[smoke] kept temp root: {temp_root}")
        else:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
