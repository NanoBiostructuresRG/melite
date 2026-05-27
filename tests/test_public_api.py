# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for melite public API."""

import melite


def test_config_importable_from_melite():
    from melite import Config
    assert Config is not None


def test_load_datasets_importable_from_melite():
    from melite import load_datasets
    assert callable(load_datasets)


def test_plot_cv_distributions_importable_from_melite():
    from melite import plot_cv_distributions
    assert callable(plot_cv_distributions)


def test_predict_importable_from_melite():
    from melite import predict
    assert callable(predict)


def test_version_importable_from_melite():
    from melite import __version__
    assert isinstance(__version__, str)


def test_dunder_all_contains_expected_symbols():
    expected = {
        "Config",
        "load_datasets",
        "plot_cv_distributions",
        "predict",
        "__version__",
    }
    assert set(melite.__all__) == expected


def test_private_helpers_not_in_dunder_all():
    assert "_load_toml" not in melite.__all__
    assert "_deep_merge" not in melite.__all__
    assert "_scatter_with_jitter" not in melite.__all__
    assert "load_dataset" not in melite.__all__
    assert "ResultManager" not in melite.__all__
    assert "Pipeline" not in melite.__all__


def test_removed_top_level_symbols_not_exposed():
    assert not callable(getattr(melite, "load_dataset", None))
    assert not hasattr(melite, "ResultManager")
