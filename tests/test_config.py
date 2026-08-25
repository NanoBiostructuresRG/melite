# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for melite.config."""

import doctest
import random
from pathlib import Path

import numpy as np
import pytest

import melite.config as config_module
from melite.config import Config


def test_config_instantiates_without_filesystem_side_effects(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    Config()

    assert not (tmp_path / "raw").exists()
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "output").exists()


def test_config_default_construction_does_not_discover_cwd_config(
    monkeypatch, tmp_path
):
    (tmp_path / "config.toml").write_text(
        '[classifiers]\nactive = ["stack"]\n', encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    cfg = Config(user_config=None)

    assert cfg.ACTIVE_CLASSIFIERS == ["svc", "rf", "xgb"]


def test_config_construction_does_not_mutate_global_rng_state():
    random.seed(1729)
    np.random.seed(1729)
    python_state = random.getstate()
    numpy_state = np.random.get_state()

    Config()

    assert random.getstate() == python_state
    actual_numpy_state = np.random.get_state()
    assert actual_numpy_state[0] == numpy_state[0]
    assert np.array_equal(actual_numpy_state[1], numpy_state[1])
    assert actual_numpy_state[2:] == numpy_state[2:]


def test_config_exposes_only_the_intended_public_runtime_values():
    cfg = Config()

    public_instance_values = {name for name in vars(cfg) if not name.startswith("_")}

    assert public_instance_values == {
        "SMOKE",
        "PATHS",
        "RESULTS_FILE",
        "RANDOM_STATE",
        "DATASETS",
        "ACTIVE_CLASSIFIERS",
        "CV_CONFIG",
    }


def test_config_smoke_false_uses_full_cv():
    cfg = Config(smoke=False)
    assert cfg.CV_CONFIG == {
        "n_splits": 5,
        "n_repeats": 3,
        "inner_n_splits": 3,
    }
    assert "random_state" not in cfg.CV_CONFIG


def test_config_smoke_true_uses_reduced_cv():
    cfg = Config(smoke=True)
    assert cfg.CV_CONFIG == {
        "n_splits": 3,
        "n_repeats": 1,
        "inner_n_splits": 2,
    }
    assert "random_state" not in cfg.CV_CONFIG


def test_config_user_cv_override_inherits_default_inner_splits(tmp_path):
    user_toml = tmp_path / "custom.toml"
    user_toml.write_text("[cv]\nn_splits = 4\nn_repeats = 2\n")

    cfg = Config(user_config=user_toml)

    assert cfg.CV_CONFIG["n_splits"] == 4
    assert cfg.CV_CONFIG["n_repeats"] == 2
    assert cfg.CV_CONFIG["inner_n_splits"] == 3


def test_config_smoke_true_uses_single_value_grids():
    cfg = Config(smoke=True)
    for entry in cfg._param_grid:
        for key, val in entry.items():
            if key == "model":
                continue
            assert len(val) == 1, f"Smoke grid for {key} has more than one value: {val}"


def test_config_smoke_false_uses_full_grids():
    cfg = Config(smoke=False)
    # At least one grid entry should have multiple values
    has_multiple = any(
        len(val) > 1
        for entry in cfg._param_grid
        for key, val in entry.items()
        if key != "model"
    )
    assert has_multiple


def test_config_svc_grid_uses_pipeline_parameter_names():
    cfg = Config()
    svc_entries = [entry for entry in cfg._param_grid if entry["model"] == ["svc"]]

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
        entry
        for entry in cfg._param_grid
        if entry["model"] == ["svc"] and entry["svc__kernel"] == ["linear"]
    ]

    assert linear_entries == [
        {
            "model": ["svc"],
            "svc__kernel": ["linear"],
            "svc__C": [0.01, 0.1, 1, 10],
        }
    ]
    assert "svc__gamma" not in linear_entries[0]
    assert "svc__degree" not in linear_entries[0]
    assert "svc__coef0" not in linear_entries[0]


def test_config_smoke_svc_grid_uses_pipeline_parameter_names():
    cfg = Config(smoke=True)
    svc_entry = next(entry for entry in cfg._param_grid if entry["model"] == ["svc"])

    assert svc_entry["svc__kernel"] == ["linear"]
    assert svc_entry["svc__C"] == [1]


def test_config_private_setup_creates_directories_and_seeds_rngs(tmp_path):
    cfg = Config()
    cfg.RANDOM_STATE = 17
    cfg.PATHS = {
        "INPUT": str(tmp_path / "raw") + "/",
        "DATASET": str(tmp_path / "data") + "/",
        "OUTPUT": str(tmp_path / "output") + "/",
    }
    cfg._setup()
    assert (tmp_path / "raw").exists()
    assert (tmp_path / "data").exists()
    assert (tmp_path / "output").exists()
    assert random.random() == random.Random(17).random()
    assert np.random.random() == np.random.RandomState(17).random_sample()


def test_config_user_toml_overrides_defaults(tmp_path):
    toml_content = "[benchmark]\nlevels = [70, 85]\n"
    user_toml = tmp_path / "custom.toml"
    user_toml.write_text(toml_content)

    cfg = Config(user_config=user_toml)
    assert set(cfg.DATASETS) == {"PCA70", "PCA85", "UMAP70", "UMAP85"}


def test_config_user_toml_falls_back_to_defaults_for_missing_keys(tmp_path):
    toml_content = "[benchmark]\nlevels = [70]\n"
    user_toml = tmp_path / "custom.toml"
    user_toml.write_text(toml_content)

    cfg = Config(user_config=user_toml)
    # random_state should still be the default (42)
    assert cfg.RANDOM_STATE == 42


def test_benchmark_random_state_is_the_canonical_public_seed(tmp_path):
    user_toml = tmp_path / "seed.toml"
    user_toml.write_text("[benchmark]\nrandom_state = 17\n", encoding="utf-8")

    cfg = Config(user_config=user_toml)

    assert cfg.RANDOM_STATE == 17
    assert set(cfg.CV_CONFIG) == {
        "n_splits",
        "n_repeats",
        "inner_n_splits",
    }


@pytest.mark.parametrize("section", ["cv", "cv_smoke"])
def test_random_state_in_cv_section_fails_with_canonical_location(tmp_path, section):
    user_toml = tmp_path / f"{section}.toml"
    user_toml.write_text(f"[{section}]\nrandom_state = 17\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=rf"\[{section}\]\.random_state.*\[benchmark\]\.random_state",
    ):
        Config(user_config=user_toml)


def test_classifiers_is_the_active_public_configuration_section(tmp_path):
    user_toml = tmp_path / "classifiers.toml"
    user_toml.write_text('[classifiers]\nactive = ["svc"]\n')

    cfg = Config(user_config=user_toml)

    assert cfg.ACTIVE_CLASSIFIERS == ["svc"]
    assert not hasattr(cfg, "ACTIVE_MODELS")


def test_legacy_models_section_fails_with_migration_message(tmp_path):
    user_toml = tmp_path / "models.toml"
    user_toml.write_text('[models]\nactive = ["stack"]\n')

    with pytest.raises(
        ValueError,
        match=(
            r"\[models\] configuration section was renamed to "
            r"\[classifiers\] in MELITE v0\.2\.4"
        ),
    ):
        Config(user_config=user_toml)


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
    assert cfg.ACTIVE_CLASSIFIERS == ["svc", "rf", "xgb"]


def test_removed_config_public_members_are_absent():
    cfg = Config()

    for name in (
        "REDUCTION_TYPES",
        "REDUCTION_LEVELS",
        "PARAM_GRID",
        "PARAM_GRID_BY_MODEL",
        "get_param_grid",
        "get_cv_config",
        "setup",
    ):
        assert not hasattr(cfg, name)


def test_actual_config_doctests_execute_successfully():
    result = doctest.testmod(config_module, raise_on_error=True)

    assert result.attempted > 0
    assert result.failed == 0


def test_config_uses_user_defined_dataset_registry(tmp_path):
    toml_content = """
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
"""
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
    toml_content = """
[datasets.maccs]
label_path = "raw/labels.npy"
"""
    user_toml = tmp_path / "custom.toml"
    user_toml.write_text(toml_content)

    with pytest.raises(ValueError, match="path"):
        Config(user_config=user_toml)


def test_config_user_dataset_requires_label_path(tmp_path):
    toml_content = """
[datasets.maccs]
path = "data/maccs.npz"
"""
    user_toml = tmp_path / "custom.toml"
    user_toml.write_text(toml_content)

    with pytest.raises(ValueError, match="label_path"):
        Config(user_config=user_toml)
