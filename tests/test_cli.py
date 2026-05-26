# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for melite CLI entry points."""

import subprocess
import sys
from types import SimpleNamespace

import melite.cli as cli
from melite.version import __version__


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "melite.cli"] + args,
        capture_output=True,
        text=True,
    )


def test_help_exits_zero():
    result = _run(["--help"])
    assert result.returncode == 0


def test_help_output_contains_run_and_export():
    result = _run(["--help"])
    assert "run" in result.stdout
    assert "export" in result.stdout


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
