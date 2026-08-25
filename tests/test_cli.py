# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for melite CLI entry points."""

import subprocess
import sys
from types import SimpleNamespace

import melite.cli as cli
import numpy as np
import pytest
from melite.config import Config
from melite.load_dataset import load_datasets
from melite.version import __version__


EXPECTED_EXAMPLE_TREE = {
    "config.toml",
    "data",
    "data/sample_tabular.npz",
    "raw",
    "raw/labels.npy",
}


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "melite.cli"] + args,
        capture_output=True,
        text=True,
    )


def test_help_exits_zero():
    result = _run(["--help"])
    assert result.returncode == 0


def test_help_output_contains_run_export_and_example():
    result = _run(["--help"])
    assert "run" in result.stdout
    assert "export" in result.stdout
    assert "example" in result.stdout


def test_run_help_exits_zero():
    result = _run(["run", "--help"])
    assert result.returncode == 0


def test_run_help_mentions_smoke():
    result = _run(["run", "--help"])
    assert "--smoke" in result.stdout


def test_export_help_exits_zero():
    result = _run(["export", "--help"])
    assert result.returncode == 0


def test_export_help_mentions_row():
    result = _run(["export", "--help"])
    assert "--row" in result.stdout


def test_example_help_exits_zero():
    result = _run(["example", "--help"])
    assert result.returncode == 0


def test_version_exits_zero():
    result = _run(["--version"])
    assert result.returncode == 0


def test_version_output_contains_version_string():
    result = _run(["--version"])
    assert __version__ in result.stdout + result.stderr


def test_run_passes_config_to_main(monkeypatch, tmp_path):
    import melite.main as main_module

    calls = {}
    config_path = tmp_path / "custom.toml"

    class DummyMain:
        def __init__(self, smoke=False, user_config=None):
            calls["smoke"] = smoke
            calls["user_config"] = user_config

        def run(self):
            calls["ran"] = True

    monkeypatch.setattr(main_module, "Main", DummyMain)

    cli._run(SimpleNamespace(smoke=True, config=config_path))

    assert calls == {
        "smoke": True,
        "user_config": config_path,
        "ran": True,
    }


def test_export_passes_config_to_config_loader(monkeypatch, tmp_path):
    import melite.config as config_module
    import melite.export_best_model as export_module

    calls = {}
    config_path = tmp_path / "custom.toml"
    output_dir = tmp_path / "output"

    class DummyConfig:
        def __init__(self, user_config=None):
            calls["user_config"] = user_config
            self.PATHS = {"OUTPUT": str(output_dir)}

    class DummyFinalizer:
        def __init__(self, csv_path, outdir, config, row_index=None, force=False):
            calls["csv_path"] = csv_path
            calls["outdir"] = outdir
            calls["config"] = config
            calls["row_index"] = row_index
            calls["force"] = force

        def run(self):
            calls["ran"] = True

    monkeypatch.setattr(config_module, "Config", DummyConfig)
    monkeypatch.setattr(export_module, "Finalizer", DummyFinalizer)

    cli._export(
        SimpleNamespace(
            config=config_path,
            csv=None,
            outdir=None,
            row=2,
            force=True,
        )
    )

    assert calls["user_config"] == config_path
    assert calls["csv_path"] == output_dir / "results.csv"
    assert calls["outdir"] == output_dir
    assert isinstance(calls["config"], DummyConfig)
    assert calls["row_index"] == 2
    assert calls["force"] is True
    assert calls["ran"] is True


def test_example_dispatches(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "_example", lambda args: calls.append(args.command))
    monkeypatch.setattr(sys, "argv", ["melite", "example"])

    cli.main()

    assert calls == ["example"]


def _copy_example(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    cli._example(SimpleNamespace())
    return tmp_path / "melite_example"


def test_example_creates_exact_expected_tree(monkeypatch, tmp_path):
    destination = _copy_example(monkeypatch, tmp_path)

    copied_tree = {
        path.relative_to(destination).as_posix() for path in destination.rglob("*")
    }
    assert copied_tree == EXPECTED_EXAMPLE_TREE


def test_example_resources_are_valid_balanced_numeric_data(monkeypatch, tmp_path):
    destination = _copy_example(monkeypatch, tmp_path)
    labels = np.load(destination / "raw" / "labels.npy")
    with np.load(destination / "data" / "sample_tabular.npz") as dataset:
        X = dataset["X"]
        embedded_labels = dataset["y"]

    assert X.shape == (120, 12)
    assert labels.shape == (120,)
    assert np.issubdtype(X.dtype, np.number)
    assert np.isfinite(X).all()
    assert np.array_equal(embedded_labels, labels)
    assert set(labels.tolist()) == {0, 1}
    assert np.bincount(labels).tolist() == [60, 60]

    class_zero = X[labels == 0, 0]
    class_one = X[labels == 1, 0]
    assert class_zero.mean() < class_one.mean()
    assert class_zero.max() > class_one.min()
    assert class_one.max() > class_zero.min()


def test_example_config_loads_from_parent_directory(monkeypatch, tmp_path):
    destination = _copy_example(monkeypatch, tmp_path)

    config = Config(user_config=destination / "config.toml")

    assert config.ACTIVE_CLASSIFIERS == ["svc"]
    assert set(config.DATASETS) == {"sample_tabular"}
    assert config.PATHS["OUTPUT"] == "melite_example/output/"
    assert config.DATASETS["sample_tabular"]["path"] == (
        "melite_example/data/sample_tabular.npz"
    )
    assert config.DATASETS["sample_tabular"]["label_path"] == (
        "melite_example/raw/labels.npy"
    )


def test_example_dataset_passes_strict_loading(monkeypatch, tmp_path):
    destination = _copy_example(monkeypatch, tmp_path)
    config = Config(user_config=destination / "config.toml")

    datasets = load_datasets(config)

    assert set(datasets) == {"sample_tabular"}
    assert datasets["sample_tabular"]["X"].shape == (120, 12)
    assert np.bincount(datasets["sample_tabular"]["y"]).tolist() == [60, 60]


def test_example_refuses_existing_directory_without_modifying_it(
    monkeypatch, tmp_path, capsys
):
    destination = _copy_example(monkeypatch, tmp_path)
    sentinel = destination / "keep.txt"
    sentinel.write_text("preserve me", encoding="utf-8")
    before = {
        path.relative_to(destination).as_posix(): path.read_bytes()
        for path in destination.rglob("*")
        if path.is_file()
    }

    with pytest.raises(SystemExit) as exc_info:
        cli._example(SimpleNamespace())

    after = {
        path.relative_to(destination).as_posix(): path.read_bytes()
        for path in destination.rglob("*")
        if path.is_file()
    }
    assert exc_info.value.code == 1
    assert after == before
    assert "already exists" in capsys.readouterr().err
