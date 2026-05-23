# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for mosaic CLI entry points."""

import subprocess
import sys
from mosaic.version import __version__


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "mosaic.cli"] + args,
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
