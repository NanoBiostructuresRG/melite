# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for melite.load_dataset."""

import doctest

import melite.load_dataset as load_dataset_module
import numpy as np
import pytest
from melite.load_dataset import load_datasets, _load_dataset_legacy


def _make_config(tmp_path):
    from melite.config import Config

    cfg = Config()
    cfg.PATHS = {
        "INPUT": str(tmp_path / "raw") + "/",
        "DATASET": str(tmp_path / "data") + "/",
        "OUTPUT": str(tmp_path / "output") + "/",
    }
    return cfg


def _write_labels(tmp_path, n_samples=20):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(exist_ok=True)
    y = np.array([0, 1] * (n_samples // 2), dtype=np.int64)
    if n_samples % 2:
        y = np.append(y, 0)
    label_path = raw_dir / "labels.npy"
    np.save(label_path, y)
    return label_path, y


def _write_dataset(tmp_path, name, X, y=None, embedded_y=None):
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    path = data_dir / f"{name}.npz"
    if embedded_y is None:
        if y is None:
            np.savez(path, X=X)
        else:
            np.savez(path, X=X, y=y)
    else:
        np.savez(path, X=X, y=embedded_y)
    return path


def _registry_config(tmp_path, datasets):
    cfg = _make_config(tmp_path)
    cfg.DATASETS = datasets
    return cfg


def test_valid_npz_with_matching_y_loads(tmp_path, tmp_labels, tmp_npz_valid):
    cfg = _make_config(tmp_path)
    result = _load_dataset_legacy(cfg, "PCA", [70])
    assert "PCA70" in result
    X, y = result["PCA70"]
    assert X.shape == (20, 5)
    assert y.shape == (20,)


def test_valid_npz_without_y_loads(tmp_path, tmp_labels, tmp_npz_no_y):
    cfg = _make_config(tmp_path)
    result = _load_dataset_legacy(cfg, "PCA", [70])
    assert "PCA70" in result


def test_missing_file_warns_and_skips(tmp_path, tmp_labels, caplog):
    import logging

    cfg = _make_config(tmp_path)
    with caplog.at_level(logging.WARNING, logger="melite.load_dataset"):
        result = _load_dataset_legacy(cfg, "PCA", [70])
    assert result == {}
    assert any("not found" in msg.lower() for msg in caplog.messages)


def test_missing_X_key_raises_value_error(tmp_path, tmp_labels, tmp_npz_missing_X):
    cfg = _make_config(tmp_path)
    with pytest.raises(ValueError, match="Required key 'X' not found"):
        _load_dataset_legacy(cfg, "PCA", [70])


def test_missing_X_error_includes_available_keys(
    tmp_path, tmp_labels, tmp_npz_missing_X
):
    cfg = _make_config(tmp_path)
    with pytest.raises(ValueError, match="Available keys"):
        _load_dataset_legacy(cfg, "PCA", [70])


def test_mismatched_y_raises_value_error(tmp_path, tmp_labels, tmp_npz_mismatched_y):
    cfg = _make_config(tmp_path)
    with pytest.raises(ValueError, match="Label mismatch"):
        _load_dataset_legacy(cfg, "PCA", [70])


def test_mismatched_y_error_includes_shapes(tmp_path, tmp_labels, tmp_npz_mismatched_y):
    cfg = _make_config(tmp_path)
    with pytest.raises(ValueError, match=r"shape=\(20,\)"):
        _load_dataset_legacy(cfg, "PCA", [70])


def test_mismatched_y_error_includes_diff_count(
    tmp_path, tmp_labels, tmp_npz_mismatched_y
):
    cfg = _make_config(tmp_path)
    with pytest.raises(ValueError, match="Differing elements"):
        _load_dataset_legacy(cfg, "PCA", [70])


def test_load_datasets_loads_arbitrary_dataset_ids_and_metadata(tmp_path):
    label_path, y = _write_labels(tmp_path, n_samples=20)
    datasets = {}
    specs = {
        "morgan_r2_2048": ("fingerprints", "Morgan", None),
        "maccs": ("fingerprints", "MACCS", None),
        "rdkit_descriptors": ("descriptors", "RDKit", None),
        "pca85": ("dimensionality", "PCA", 85),
        "umap90": ("dimensionality", "UMAP", 90),
    }
    for index, (dataset_id, (family, method, level)) in enumerate(specs.items()):
        path = _write_dataset(
            tmp_path,
            dataset_id,
            np.full((20, 3 + index), index, dtype=np.float32),
            y=y,
        )
        metadata = {"family": family, "method": method}
        if level is not None:
            metadata["level"] = level
        datasets[dataset_id] = {
            "path": str(path),
            "label_path": str(label_path),
            "metadata": metadata,
        }

    result = load_datasets(_registry_config(tmp_path, datasets))

    assert list(result) == [
        "morgan_r2_2048",
        "maccs",
        "rdkit_descriptors",
        "pca85",
        "umap90",
    ]
    assert result["maccs"]["X"].shape == (20, 4)
    assert np.array_equal(result["rdkit_descriptors"]["y"], y)
    assert result["pca85"]["metadata"] == {
        "family": "dimensionality",
        "method": "PCA",
        "level": 85,
    }


def test_load_datasets_arbitrary_id_is_not_treated_as_method_name(tmp_path):
    label_path, y = _write_labels(tmp_path, n_samples=20)
    path = _write_dataset(tmp_path, "anything_user_wants", np.ones((20, 2)), y=y)
    cfg = _registry_config(
        tmp_path,
        {
            "not_a_method_name": {
                "path": str(path),
                "label_path": str(label_path),
                "metadata": {"family": "custom"},
            }
        },
    )

    result = load_datasets(cfg)

    assert set(result) == {"not_a_method_name"}
    assert result["not_a_method_name"]["metadata"] == {"family": "custom"}


def test_load_datasets_missing_X_key_raises_value_error(tmp_path):
    label_path, y = _write_labels(tmp_path, n_samples=20)
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    path = data_dir / "maccs.npz"
    np.savez(path, y=y)
    cfg = _registry_config(
        tmp_path,
        {"maccs": {"path": str(path), "label_path": str(label_path), "metadata": {}}},
    )

    with pytest.raises(ValueError, match="Required key 'X' not found"):
        load_datasets(cfg)


def test_load_datasets_missing_npz_raises_file_not_found_error(tmp_path):
    label_path, _ = _write_labels(tmp_path, n_samples=20)
    cfg = _registry_config(
        tmp_path,
        {
            "maccs": {
                "path": str(tmp_path / "data" / "missing.npz"),
                "label_path": str(label_path),
                "metadata": {},
            }
        },
    )

    with pytest.raises(FileNotFoundError, match="file not found"):
        load_datasets(cfg)


def test_load_datasets_missing_label_path_raises_file_not_found_error(tmp_path):
    path = _write_dataset(tmp_path, "maccs", np.ones((20, 2)), y=None)
    cfg = _registry_config(
        tmp_path,
        {
            "maccs": {
                "path": str(path),
                "label_path": str(tmp_path / "raw" / "missing.npy"),
                "metadata": {},
            }
        },
    )

    with pytest.raises(FileNotFoundError, match="label_path not found"):
        load_datasets(cfg)


def test_load_datasets_non_2d_X_raises_value_error(tmp_path):
    label_path, y = _write_labels(tmp_path, n_samples=20)
    path = _write_dataset(tmp_path, "maccs", np.ones(20), y=y)
    cfg = _registry_config(
        tmp_path,
        {"maccs": {"path": str(path), "label_path": str(label_path), "metadata": {}}},
    )

    with pytest.raises(ValueError, match="2D"):
        load_datasets(cfg)


def test_load_datasets_non_numeric_X_raises_value_error(tmp_path):
    label_path, y = _write_labels(tmp_path, n_samples=20)
    X = np.array([["a", "b"]] * 20)
    path = _write_dataset(tmp_path, "maccs", X, y=y)
    cfg = _registry_config(
        tmp_path,
        {"maccs": {"path": str(path), "label_path": str(label_path), "metadata": {}}},
    )

    with pytest.raises(ValueError, match="numeric"):
        load_datasets(cfg)


def test_load_datasets_X_y_length_mismatch_raises_value_error(tmp_path):
    label_path, y = _write_labels(tmp_path, n_samples=20)
    path = _write_dataset(tmp_path, "maccs", np.ones((19, 2)), y=None)
    cfg = _registry_config(
        tmp_path,
        {"maccs": {"path": str(path), "label_path": str(label_path), "metadata": {}}},
    )

    with pytest.raises(ValueError, match="length mismatch"):
        load_datasets(cfg)


def test_load_datasets_embedded_y_mismatch_raises_value_error(tmp_path):
    label_path, y = _write_labels(tmp_path, n_samples=20)
    bad_y = np.ones_like(y)
    path = _write_dataset(tmp_path, "maccs", np.ones((20, 2)), embedded_y=bad_y)
    cfg = _registry_config(
        tmp_path,
        {"maccs": {"path": str(path), "label_path": str(label_path), "metadata": {}}},
    )

    with pytest.raises(ValueError, match="Label mismatch"):
        load_datasets(cfg)


def test_load_datasets_accepts_categorical_authoritative_labels(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    y = np.array(["class_a", "class_b", "class_a", "class_b"])
    label_path = raw_dir / "labels.npy"
    np.save(label_path, y)
    path = _write_dataset(
        tmp_path,
        "sample_tabular",
        np.ones((4, 2)),
        embedded_y=y,
    )
    cfg = _registry_config(
        tmp_path,
        {
            "sample_tabular": {
                "path": str(path),
                "label_path": str(label_path),
                "metadata": {},
            }
        },
    )

    result = load_datasets(cfg)

    assert np.array_equal(result["sample_tabular"]["y"], y)


def test_load_datasets_scalar_authoritative_y_raises_specific_error(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    label_path = raw_dir / "labels.npy"
    np.save(label_path, np.array(1))
    path = _write_dataset(tmp_path, "sample_tabular", np.ones((1, 2)))
    cfg = _registry_config(
        tmp_path,
        {
            "sample_tabular": {
                "path": str(path),
                "label_path": str(label_path),
                "metadata": {},
            }
        },
    )

    with pytest.raises(
        ValueError,
        match=r"sample_tabular.*authoritative y must be 1D.*shape \(\)",
    ):
        load_datasets(cfg)


def test_load_datasets_2d_authoritative_y_raises_specific_error(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    label_path = raw_dir / "labels.npy"
    np.save(label_path, np.ones((20, 1), dtype=np.int64))
    path = _write_dataset(tmp_path, "sample_tabular", np.ones((20, 2)))
    cfg = _registry_config(
        tmp_path,
        {
            "sample_tabular": {
                "path": str(path),
                "label_path": str(label_path),
                "metadata": {},
            }
        },
    )

    with pytest.raises(
        ValueError,
        match=r"authoritative y must be 1D.*shape \(20, 1\)",
    ):
        load_datasets(cfg)


def test_load_datasets_non_1d_embedded_y_raises_specific_error(tmp_path):
    label_path, _ = _write_labels(tmp_path, n_samples=20)
    path = _write_dataset(
        tmp_path,
        "sample_tabular",
        np.ones((20, 2)),
        embedded_y=np.ones((20, 1), dtype=np.int64),
    )
    cfg = _registry_config(
        tmp_path,
        {
            "sample_tabular": {
                "path": str(path),
                "label_path": str(label_path),
                "metadata": {},
            }
        },
    )

    with pytest.raises(
        ValueError,
        match=r"embedded y must be 1D.*shape \(20, 1\)",
    ):
        load_datasets(cfg)


def test_load_datasets_embedded_y_shape_mismatch_is_structural_error(tmp_path):
    label_path, _ = _write_labels(tmp_path, n_samples=20)
    path = _write_dataset(
        tmp_path,
        "sample_tabular",
        np.ones((20, 2)),
        embedded_y=np.ones(19, dtype=np.int64),
    )
    cfg = _registry_config(
        tmp_path,
        {
            "sample_tabular": {
                "path": str(path),
                "label_path": str(label_path),
                "metadata": {},
            }
        },
    )

    with pytest.raises(
        ValueError,
        match=(
            r"embedded y shape \(19,\) does not match "
            r"authoritative y shape \(20,\)"
        ),
    ):
        load_datasets(cfg)


def test_load_datasets_embedded_y_value_mismatch_keeps_diagnostics(tmp_path):
    label_path, y = _write_labels(tmp_path, n_samples=20)
    path = _write_dataset(
        tmp_path,
        "sample_tabular",
        np.ones((20, 2)),
        embedded_y=1 - y,
    )
    cfg = _registry_config(
        tmp_path,
        {
            "sample_tabular": {
                "path": str(path),
                "label_path": str(label_path),
                "metadata": {},
            }
        },
    )

    with pytest.raises(ValueError) as exc_info:
        load_datasets(cfg)

    assert "Label mismatch" in str(exc_info.value)
    assert "Differing elements" in str(exc_info.value)


def test_load_datasets_metadata_is_shallow_copied_and_uninterpreted(tmp_path):
    label_path, y = _write_labels(tmp_path, n_samples=20)
    path = _write_dataset(
        tmp_path,
        "opaque_dataset_17",
        np.ones((20, 2)),
        embedded_y=y,
    )
    nested_value = {"labels": ["alpha", "beta"]}
    metadata = {"opaque_key": nested_value, "description": "Opaque metadata"}
    cfg = _registry_config(
        tmp_path,
        {
            "opaque_dataset_17": {
                "path": str(path),
                "label_path": str(label_path),
                "metadata": metadata,
            }
        },
    )

    result = load_datasets(cfg)
    loaded_metadata = result["opaque_dataset_17"]["metadata"]

    assert loaded_metadata == metadata
    assert loaded_metadata is not metadata
    assert loaded_metadata["opaque_key"] is nested_value


def test_load_dataset_module_doctest_executes_public_example():
    result = doctest.testmod(load_dataset_module)

    assert result.attempted > 0
    assert result.failed == 0


def test_load_dataset_legacy_private_wrapper_remains_tuple_mapping(tmp_path):
    label_path, y = _write_labels(tmp_path, n_samples=20)
    _write_dataset(tmp_path, "PCA70", np.ones((20, 2)), y=y)
    cfg = _make_config(tmp_path)

    result = _load_dataset_legacy(cfg, "PCA", [70])

    assert set(result) == {"PCA70"}
    X_loaded, y_loaded = result["PCA70"]
    assert X_loaded.shape == (20, 2)
    assert np.array_equal(y_loaded, np.load(label_path))
