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
    assert cfg.CV_CONFIG == {
        "n_splits": 5,
        "n_repeats": 3,
        "inner_n_splits": 3,
        "random_state": 42,
    }


def test_config_smoke_true_uses_reduced_cv():
    cfg = Config(smoke=True)
    assert cfg.CV_CONFIG == {
        "n_splits": 3,
        "n_repeats": 1,
        "inner_n_splits": 2,
        "random_state": 42,
    }


def test_config_user_cv_override_inherits_default_inner_splits(tmp_path):
    user_toml = tmp_path / "custom.toml"
    user_toml.write_text("[cv]\nn_splits = 4\nn_repeats = 2\n")

    cfg = Config(user_config=user_toml)

    assert cfg.CV_CONFIG["n_splits"] == 4
    assert cfg.CV_CONFIG["n_repeats"] == 2
    assert cfg.CV_CONFIG["inner_n_splits"] == 3


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


def test_config_svc_grid_uses_pipeline_parameter_names():
    cfg = Config()
    svc_entries = [
        entry for entry in cfg.PARAM_GRID
        if entry["model"] == ["svc"]
    ]

    assert svc_entries
    for entry in svc_entries:
        assert "C" not in entry
        assert "kernel" not in entry
        assert "gamma" not in entry
        assert "svc__C" in entry
        assert "svc__kernel" in entry


def test_config_full_svc_grid_includes_linear_kernel_without_unused_params():
    cfg = Config()

    linear_entries = [
        entry for entry in cfg.PARAM_GRID
        if entry["model"] == ["svc"] and entry["svc__kernel"] == ["linear"]
    ]

    assert linear_entries == [{
        "model": ["svc"],
        "svc__kernel": ["linear"],
        "svc__C": [0.01, 0.1, 1, 10],
    }]
    assert "svc__gamma" not in linear_entries[0]
    assert "svc__degree" not in linear_entries[0]
    assert "svc__coef0" not in linear_entries[0]


def test_config_smoke_svc_grid_uses_pipeline_parameter_names():
    cfg = Config(smoke=True)
    svc_entry = next(
        entry for entry in cfg.PARAM_GRID
        if entry["model"] == ["svc"]
    )

    assert svc_entry["svc__kernel"] == ["linear"]
    assert svc_entry["svc__C"] == [1]


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


def test_config_synthesizes_legacy_dataset_registry(tmp_path):
    toml_content = (
        '[paths]\ninput = "raw/"\ndataset = "data/"\noutput = "output/"\n'
        '[benchmark]\nreduction_types = ["PCA", "UMAP"]\nlevels = [70, 75]\n'
    )
    user_toml = tmp_path / "custom.toml"
    user_toml.write_text(toml_content)

    cfg = Config(user_config=user_toml)

    assert set(cfg.DATASETS) == {"PCA70", "PCA75", "UMAP70", "UMAP75"}
    assert Path(cfg.DATASETS["PCA70"]["path"]) == Path("data/PCA70.npz")
    assert Path(cfg.DATASETS["PCA70"]["label_path"]) == Path("raw/labels.npy")
    assert cfg.DATASETS["PCA70"]["metadata"] == {
        "family": "dimensionality",
        "method": "PCA",
        "level": 70,
    }


def test_config_uses_user_defined_dataset_registry(tmp_path):
    toml_content = '''
[datasets.morgan_r2_2048]
path = "data/morgan_r2_2048.npz"
label_path = "raw/labels.npy"
family = "fingerprints"
method = "Morgan"
variant = "r2_2048"
description = "Morgan radius 2 fingerprint"

[datasets.rdkit_descriptors]
path = "data/rdkit_descriptors.npz"
label_path = "raw/labels.npy"
family = "descriptors"
'''
    user_toml = tmp_path / "custom.toml"
    user_toml.write_text(toml_content)

    cfg = Config(user_config=user_toml)

    assert set(cfg.DATASETS) == {"morgan_r2_2048", "rdkit_descriptors"}
    assert cfg.DATASETS["morgan_r2_2048"] == {
        "path": "data/morgan_r2_2048.npz",
        "label_path": "raw/labels.npy",
        "metadata": {
            "family": "fingerprints",
            "method": "Morgan",
            "variant": "r2_2048",
            "description": "Morgan radius 2 fingerprint",
        },
    }


def test_config_user_dataset_requires_path(tmp_path):
    toml_content = '''
[datasets.maccs]
label_path = "raw/labels.npy"
'''
    user_toml = tmp_path / "custom.toml"
    user_toml.write_text(toml_content)

    with pytest.raises(ValueError, match="path"):
        Config(user_config=user_toml)


def test_config_user_dataset_requires_label_path(tmp_path):
    toml_content = '''
[datasets.maccs]
path = "data/maccs.npz"
'''
    user_toml = tmp_path / "custom.toml"
    user_toml.write_text(toml_content)

    with pytest.raises(ValueError, match="label_path"):
        Config(user_config=user_toml)
