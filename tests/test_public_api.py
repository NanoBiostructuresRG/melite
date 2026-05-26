# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for melite public API."""

import melite


def test_config_importable_from_melite():
    from melite import Config
    assert Config is not None


def test_load_dataset_importable_from_melite():
    from melite import load_dataset
    assert callable(load_dataset)


def test_result_manager_importable_from_melite():
    from melite import ResultManager
    assert ResultManager is not None


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
        "load_dataset",
        "ResultManager",
        "plot_cv_distributions",
        "predict",
        "__version__",
    }
    assert set(melite.__all__) == expected


def test_private_helpers_not_in_dunder_all():
    assert "_load_toml" not in melite.__all__
    assert "_deep_merge" not in melite.__all__
    assert "_scatter_with_jitter" not in melite.__all__
    assert "Pipeline" not in melite.__all__
