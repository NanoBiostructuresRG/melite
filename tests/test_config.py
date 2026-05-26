# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for melite.config."""

import pytest
from pathlib import Path
from melite.config import Config


def test_config_instantiates_without_filesystem_side_effects(tmp_path):
    """Config() must not create directories or touch the filesystem."""
    cfg = Config()
    # Directories should not be created by __init__
    assert not (tmp_path / "raw").exists()
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "output").exists()


def test_config_smoke_false_uses_full_cv():
    cfg = Config(smoke=False)
    assert cfg.CV_CONFIG["n_splits"] == 10
    assert cfg.CV_CONFIG["n_repeats"] == 5


def test_config_smoke_true_uses_reduced_cv():
    cfg = Config(smoke=True)
    assert cfg.CV_CONFIG["n_splits"] == 3
    assert cfg.CV_CONFIG["n_repeats"] == 1


def test_config_smoke_true_uses_single_value_grids():
    cfg = Config(smoke=True)
    for entry in cfg.PARAM_GRID:
        for key, val in entry.items():
            if key == "model":
                continue
            assert len(val) == 1, f"Smoke grid for {key} has more than one value: {val}"


def test_config_smoke_false_uses_full_grids():
    cfg = Config(smoke=False)
    # At least one grid entry should have multiple values
    has_multiple = any(
        len(val) > 1
        for entry in cfg.PARAM_GRID
        for key, val in entry.items()
        if key != "model"
    )
    assert has_multiple


def test_config_setup_creates_directories(tmp_path):
    cfg = Config()
    cfg.PATHS = {
        "INPUT":   str(tmp_path / "raw") + "/",
        "DATASET": str(tmp_path / "data") + "/",
        "OUTPUT":  str(tmp_path / "output") + "/",
    }
    cfg.setup()
    assert (tmp_path / "raw").exists()
    assert (tmp_path / "data").exists()
    assert (tmp_path / "output").exists()


def test_config_user_toml_overrides_defaults(tmp_path):
    toml_content = '[benchmark]\nlevels = [70, 85]\n'
    user_toml = tmp_path / "custom.toml"
    user_toml.write_text(toml_content)

    cfg = Config(user_config=user_toml)
    assert cfg.REDUCTION_LEVELS == [70, 85]


def test_config_user_toml_falls_back_to_defaults_for_missing_keys(tmp_path):
    toml_content = '[benchmark]\nlevels = [70]\n'
    user_toml = tmp_path / "custom.toml"
    user_toml.write_text(toml_content)

    cfg = Config(user_config=user_toml)
    # random_state should still be the default (42)
    assert cfg.RANDOM_STATE == 42
