# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for MELITE's internal optimization policy contract."""

import ast
import tomllib
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path

import pytest

import melite.config as config_module
import melite.optimization_policy as policy_module
from melite.optimization_policy import OPTIMIZATION_POLICY


_REPOSITORY_ROOT = Path(__file__).parents[1]


def _imported_modules(module_path):
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)
    return imported_modules


def test_optimization_policy_contract_is_exact():
    """Protect the fixed methodological policy against accidental drift.

    Accidental changes must be reverted. A deliberate methodological revision
    requires explicit policy review and corresponding updates to this expected
    contract and the release/decision documentation.
    """
    assert asdict(OPTIMIZATION_POLICY) == {
        "sampler": "tpe",
        "n_startup_trials": 20,
        "smoke_n_trials": 5,
        "multivariate": False,
        "group": False,
        "constant_liar": False,
        "pruning": False,
        "storage": "in_memory",
        "n_jobs": 1,
        "direction": "maximize",
        "objective": "f1_macro",
    }


def test_optimization_policy_is_immutable():
    with pytest.raises(FrozenInstanceError):
        setattr(OPTIMIZATION_POLICY, "n_jobs", 2)


def test_b2_production_modules_do_not_import_optuna():
    for module in (policy_module, config_module):
        imported_modules = _imported_modules(Path(module.__file__))
        assert all(
            imported != "optuna" and not imported.startswith("optuna.")
            for imported in imported_modules
        )


def test_optuna_is_an_exact_base_dependency_only():
    """Protect the deliberate Optuna major-version compatibility boundary.

    Changing this dependency, especially to Optuna 5+, requires the review in
    ``docs/decisions.md`` covering sampler behavior, trial/error semantics, and
    compatibility with MELITE's declared optimization policy.
    """
    with open(_REPOSITORY_ROOT / "pyproject.toml", "rb") as file:
        pyproject = tomllib.load(file)

    base_optuna = [
        dependency
        for dependency in pyproject["project"]["dependencies"]
        if dependency.lower().startswith("optuna")
    ]
    optional_dependencies = pyproject["project"]["optional-dependencies"]
    optional_optuna = [
        dependency
        for dependencies in optional_dependencies.values()
        for dependency in dependencies
        if dependency.lower().startswith("optuna")
    ]

    assert base_optuna == ["optuna>=4,<5"]
    assert optional_optuna == []


def test_default_config_has_no_optimization_smoke_section():
    with open(_REPOSITORY_ROOT / "melite" / "config_default.toml", "rb") as file:
        default_config = tomllib.load(file)

    assert "optimization_smoke" not in default_config
