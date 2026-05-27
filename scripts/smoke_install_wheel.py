# SPDX-License-Identifier: LGPL-3.0-or-later
"""Installed-wheel smoke test for the MELITE toy dataset workflow.

The script builds a wheel from the current checkout, installs it into a
temporary virtual environment, creates a tiny strict ``[datasets.toy]``
configuration outside the repository, runs ``melite run --smoke``, exports row
0 non-interactively, and verifies the expected artifacts.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_API = [
    "Config",
    "load_datasets",
    "plot_cv_distributions",
    "predict",
    "__version__",
]


def _run(cmd: list[str | Path], *, cwd: Path, env: dict[str, str] | None = None) -> None:
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
        [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", dist_dir],
        cwd=REPO_ROOT,
    )
    wheels = sorted(dist_dir.glob("melite-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected exactly one MELITE wheel in {dist_dir}, got {wheels}")
    return wheels[0]


def _write_toy_project(work_dir: Path) -> Path:
    raw_dir = work_dir / "raw"
    data_dir = work_dir / "data"
    output_dir = work_dir / "output"
    raw_dir.mkdir()
    data_dir.mkdir()
    output_dir.mkdir()

    X = np.array([
        [0.0, 0.1, 1.0],
        [0.1, 0.0, 0.9],
        [0.2, 0.1, 1.1],
        [0.1, 0.2, 1.0],
        [0.2, 0.0, 0.8],
        [0.0, 0.2, 1.2],
        [1.0, 1.1, 0.0],
        [1.1, 1.0, 0.1],
        [0.9, 1.2, 0.0],
        [1.2, 0.9, 0.2],
        [1.0, 1.0, 0.0],
        [1.1, 1.2, 0.1],
    ], dtype=float)
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.int64)

    label_path = raw_dir / "labels.npy"
    dataset_path = data_dir / "toy.npz"
    np.save(label_path, y)
    np.savez(dataset_path, X=X, y=y)

    config_path = work_dir / "toy_config.toml"
    config_path.write_text(
        f"""
[paths]
input = "{raw_dir.as_posix()}/"
dataset = "{data_dir.as_posix()}/"
output = "{output_dir.as_posix()}/"

[benchmark]
random_state = 42

[cv]
n_splits = 3
n_repeats = 1

[cv_smoke]
n_splits = 3
n_repeats = 1

[models]
active = ["svc"]

[datasets.toy]
path = "{dataset_path.as_posix()}"
label_path = "{label_path.as_posix()}"
family = "smoke"
method = "toy"
description = "Installed wheel smoke dataset"
""".lstrip(),
        encoding="utf-8",
    )
    return config_path


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


def _verify_outputs(work_dir: Path) -> None:
    output_dir = work_dir / "output"
    results_csv = output_dir / "results.csv"
    model_path = output_dir / "Model_SVC_toy.pkl"
    figure_path = output_dir / "figures" / "SVC_toy.png"

    for path in (results_csv, model_path, figure_path):
        if not path.exists():
            raise AssertionError(f"Expected artifact was not created: {path}")

    with open(results_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise AssertionError(f"Expected one result row, got {len(rows)}")
    row = rows[0]
    if row["dataset"] != "toy":
        raise AssertionError(f"Expected dataset 'toy', got {row['dataset']!r}")
    if row["model_name"] != "SVC":
        raise AssertionError(f"Expected model 'SVC', got {row['model_name']!r}")
    if row["smoke"] != "True":
        raise AssertionError(f"Expected smoke=True row, got {row['smoke']!r}")


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

        _run([sys.executable, "-m", "venv", "--system-site-packages", venv_dir], cwd=temp_root)
        python = _venv_python(venv_dir)
        melite = _venv_melite(venv_dir)

        _run([python, "-m", "pip", "install", "--no-deps", wheel], cwd=temp_root)
        _check_imports(python)

        config_path = _write_toy_project(work_dir)
        _run([melite, "run", "--config", config_path, "--smoke"], cwd=work_dir)
        _run(
            [
                melite,
                "export",
                "--config",
                config_path,
                "--row",
                "0",
                "--csv",
                work_dir / "output" / "results.csv",
                "--outdir",
                work_dir / "output",
                "--force",
            ],
            cwd=work_dir,
        )
        _verify_outputs(work_dir)
        print("[smoke] installed-wheel toy workflow passed")
    finally:
        if args.keep_temp:
            print(f"[smoke] kept temp root: {temp_root}")
        else:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
