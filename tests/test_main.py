# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for melite.main orchestration."""

import ast
import csv
import json
import logging
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

import melite.main as main_module
from melite.config import Config
from melite.main import Main, Pipeline
from melite.optimization import OptimizationResult
from melite.search_spaces import get_search_space
from melite.version import __version__


class SVC:
    pass


class DummyPipeline:
    calls = []

    def __init__(self, config):
        self.config = config

    def run(self, X_train, y_train, reduction_type, level):
        self.calls.append(
            {
                "shape": X_train.shape,
                "n_labels": len(y_train),
                "reduction_type": reduction_type,
                "level": level,
            }
        )
        return (
            SVC(),
            {"C": 1.0, "kernel": "linear"},
            0.8,
            0.01,
            0.82,
            0.02,
            0.9,
            0.03,
        )

    def run_with_evaluations(self, X_train, y_train, reduction_type, level):
        selected_result = self.run(X_train, y_train, reduction_type, level)
        evaluations = [
            {
                "classifier_key": "svc",
                "f1_macro": 0.8123456789,
                "f1_std": 0.0123456789,
                "accuracy": 0.8234567891,
                "acc_std": 0.0234567891,
                "auc_roc": 0.9345678912,
                "auc_std": 0.0345678912,
                "outer_scores": [
                    {
                        "outer_split": 0,
                        "outer_repeat": 0,
                        "outer_fold": 0,
                        "f1_macro": 0.8012345678,
                        "accuracy": 0.8123456789,
                        "auc_roc": 0.9234567891,
                    },
                    {
                        "outer_split": 1,
                        "outer_repeat": 0,
                        "outer_fold": 1,
                        "f1_macro": 0.8234567891,
                        "accuracy": 0.8345678912,
                        "auc_roc": 0.9456789123,
                    },
                ],
                "optimization_searches": [
                    {
                        "outer_split": 0,
                        "outer_repeat": 0,
                        "outer_fold": 0,
                        "best_params": {"C": 0.123456789012345},
                        "best_inner_f1_macro": 0.7111111111111111,
                        "n_trials_requested": 100,
                        "n_trials_complete": 99,
                        "n_trials_failed": 1,
                    },
                    {
                        "outer_split": 1,
                        "outer_repeat": 0,
                        "outer_fold": 1,
                        "best_params": {"C": 1.23456789012345},
                        "best_inner_f1_macro": 0.7222222222222222,
                        "n_trials_requested": 100,
                        "n_trials_complete": 100,
                        "n_trials_failed": 0,
                    },
                ],
                "final_optimization_search": OptimizationResult(
                    best_params={"C": 1.0, "kernel": "linear"},
                    best_inner_f1_macro=0.8333333333333333,
                    n_trials_requested=100,
                    n_trials_complete=98,
                    n_trials_failed=2,
                ),
                "selected": True,
            },
            {
                "classifier_key": "rf",
                "f1_macro": 0.7123456789,
                "f1_std": 0.0456789123,
                "accuracy": 0.7234567891,
                "acc_std": 0.0567891234,
                "auc_roc": None,
                "auc_std": None,
                "outer_scores": [
                    {
                        "outer_split": 0,
                        "outer_repeat": 0,
                        "outer_fold": 0,
                        "f1_macro": 0.7123456789,
                        "accuracy": 0.7234567891,
                        "auc_roc": None,
                    }
                ],
                "optimization_searches": [
                    {
                        "outer_split": 0,
                        "outer_repeat": 0,
                        "outer_fold": 0,
                        "best_params": {"max_depth": 7},
                        "best_inner_f1_macro": 0.6111111111111111,
                        "n_trials_requested": 100,
                        "n_trials_complete": 97,
                        "n_trials_failed": 3,
                    }
                ],
                "selected": False,
            },
        ]
        return selected_result, evaluations


class DummyTrainer:
    def __init__(self):
        self.legacy_result = object()
        self.rich_result = (object(), [{"classifier_key": "svc", "selected": True}])
        self.calls = []

    def train_and_select_best_model(self, *args):
        self.calls.append(("legacy", args))
        return self.legacy_result

    def evaluate_and_select_models(self, *args):
        self.calls.append(("rich", args))
        return self.rich_result


def _write_labels(path, n_samples=8):
    path.mkdir(parents=True, exist_ok=True)
    y = np.array([0, 1] * (n_samples // 2), dtype=np.int64)
    label_path = path / "labels.npy"
    np.save(label_path, y)
    return label_path, y


def _write_npz(path, name, X, y):
    path.mkdir(parents=True, exist_ok=True)
    data_path = path / f"{name}.npz"
    np.savez(data_path, X=X, y=y)
    return data_path


def _rows(csv_path):
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_pipeline_run_preserves_legacy_trainer_result():
    trainer = DummyTrainer()
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.model_trainer = trainer

    result = pipeline.run("X", "y", "PCA", 70)

    assert result is trainer.legacy_result
    assert trainer.calls == [("legacy", ("X", "y", "PCA", 70))]


def test_pipeline_run_with_evaluations_returns_rich_result_unchanged():
    trainer = DummyTrainer()
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.model_trainer = trainer

    result = pipeline.run_with_evaluations("X", "y", "PCA", 70)

    assert result is trainer.rich_result
    assert trainer.calls == [("rich", ("X", "y", "PCA", 70))]


@pytest.mark.parametrize(
    ("smoke", "n_trials", "active", "expected_warnings"),
    [
        (False, 20, ["svc", "rf"], 1),
        (False, 21, ["svc"], 0),
        (True, 5, ["svc"], 0),
        (False, 5, ["stack"], 0),
    ],
)
def test_low_budget_warning_conditions(
    caplog, smoke, n_trials, active, expected_warnings
):
    main = Main.__new__(Main)
    main.config = SimpleNamespace(
        SMOKE=smoke,
        N_TRIALS=n_trials,
        ACTIVE_CLASSIFIERS=active,
    )

    with caplog.at_level(logging.WARNING, logger="melite.main"):
        main._warn_for_low_optimization_budget()

    warnings = [
        record for record in caplog.records if "startup sampling" in record.message
    ]
    assert len(warnings) == expected_warnings


def test_main_run_enters_optuna_logging_scope_once(monkeypatch):
    events = []

    @contextmanager
    def fake_scope(*, verbose):
        events.append(("enter", verbose))
        try:
            yield
        finally:
            events.append(("exit", verbose))

    main = Main.__new__(Main)
    main._run_evaluation = lambda: events.append(("run", None))
    monkeypatch.setattr(main_module, "optuna_logging_scope", fake_scope)
    monkeypatch.setattr(main_module.logger, "level", logging.INFO)

    main.run()

    assert events == [("enter", True), ("run", None), ("exit", True)]


@pytest.mark.parametrize(
    ("evaluations", "count"),
    [
        ([], 0),
        ([{"selected": True}, {"selected": True}], 2),
    ],
)
def test_selected_evaluation_requires_exactly_one(evaluations, count):
    with pytest.raises(
        RuntimeError,
        match=rf"Expected exactly one selected classifier evaluation; found {count}",
    ):
        Main._selected_evaluation(evaluations)


def test_main_run_uses_arbitrary_dataset_ids(monkeypatch, tmp_path):
    DummyPipeline.calls = []
    monkeypatch.setattr(main_module, "Pipeline", DummyPipeline)
    figure_calls = []

    def fake_write_evaluation_figures(self, rows, smoke=False):
        figure_calls.append({"rows": rows, "smoke": smoke})

    monkeypatch.setattr(
        main_module.ResultManager,
        "write_evaluation_figures",
        fake_write_evaluation_figures,
    )

    raw_dir = tmp_path / "raw"
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    label_path, y = _write_labels(raw_dir)
    morgan_path = _write_npz(data_dir, "morgan_r2_2048", np.ones((8, 4)), y)
    desc_path = _write_npz(data_dir, "rdkit_descriptors", np.ones((8, 3)), y)
    pca_path = _write_npz(data_dir, "PCA85", np.ones((8, 2)), y)
    config_path = tmp_path / "config.toml"
    config_path.write_text(f'''
[paths]
input = "{raw_dir.as_posix()}/"
dataset = "{data_dir.as_posix()}/"
output = "{output_dir.as_posix()}/"

[datasets.morgan_r2_2048]
path = "{morgan_path.as_posix()}"
label_path = "{label_path.as_posix()}"
family = "fingerprints"
method = "Morgan"
variant = "r2_2048"
description = "Morgan radius 2 fingerprint"

[datasets.rdkit_descriptors]
path = "{desc_path.as_posix()}"
label_path = "{label_path.as_posix()}"
family = "descriptors"
method = "RDKit"

[datasets.pca85]
path = "{pca_path.as_posix()}"
label_path = "{label_path.as_posix()}"
family = "dimensionality"
method = "PCA"
level = 85
''')

    main = Main(user_config=config_path)
    main.run()

    rows = _rows(output_dir / "results.csv")
    report = (output_dir / "results.txt").read_text(encoding="utf-8")
    assert "Classifiers: SVC, RandomForest, XGBoost, Stacking (opt-in)" in report
    assert "Classifier selected: SVC" in report
    assert "Best classifier parameters:" in report
    assert "Model Selected:" not in report
    assert "Best ML-model Parameters:" not in report
    assert [row["dataset"] for row in rows] == [
        "morgan_r2_2048",
        "rdkit_descriptors",
        "pca85",
    ]
    assert rows[0]["family"] == "fingerprints"
    assert rows[0]["method"] == "Morgan"
    assert rows[0]["variant"] == "r2_2048"
    assert rows[0]["description"] == "Morgan radius 2 fingerprint"
    assert rows[0]["reduction_type"] == ""
    assert rows[1]["method"] == "RDKit"
    assert rows[1]["reduction_type"] == ""
    assert rows[2]["method"] == "PCA"
    assert rows[2]["reduction_type"] == "PCA"
    assert rows[2]["level"] == "85"
    assert DummyPipeline.calls[0]["reduction_type"] == "morgan_r2_2048"
    assert DummyPipeline.calls[1]["reduction_type"] == "rdkit_descriptors"
    assert DummyPipeline.calls[2]["reduction_type"] == "PCA"
    assert DummyPipeline.calls[2]["level"] == 85
    assert list(main.evaluations_by_dataset) == [
        "morgan_r2_2048",
        "rdkit_descriptors",
        "pca85",
    ]
    retained = main.evaluations_by_dataset["morgan_r2_2048"]
    assert [evaluation["classifier_key"] for evaluation in retained] == ["svc", "rf"]
    assert [evaluation["selected"] for evaluation in retained] == [True, False]
    assert "outer_scores" not in rows[0]
    assert "selected" not in rows[0]

    evaluation_rows = _rows(output_dir / "evaluations.csv")
    fold_rows = _rows(output_dir / "evaluation_folds.csv")
    optimization_rows = _rows(output_dir / "optimization_searches.csv")
    assert len(evaluation_rows) == 6
    assert len(fold_rows) == 9
    assert list(evaluation_rows[0]) == [
        "dataset",
        "family",
        "method",
        "variant",
        "level",
        "description",
        "reduction_type",
        "classifier_name",
        "f1_macro",
        "f1_std",
        "accuracy",
        "acc_std",
        "auc_roc",
        "auc_std",
        "selected",
        "smoke",
    ]
    assert list(fold_rows[0]) == [
        "dataset",
        "family",
        "method",
        "variant",
        "level",
        "description",
        "reduction_type",
        "classifier_name",
        "outer_split",
        "outer_repeat",
        "outer_fold",
        "f1_macro",
        "accuracy",
        "auc_roc",
        "selected",
        "smoke",
    ]
    assert evaluation_rows[0]["dataset"] == "morgan_r2_2048"
    assert evaluation_rows[0]["family"] == "fingerprints"
    assert evaluation_rows[0]["method"] == "Morgan"
    assert evaluation_rows[0]["variant"] == "r2_2048"
    assert evaluation_rows[0]["description"] == "Morgan radius 2 fingerprint"
    assert evaluation_rows[0]["classifier_name"] == "SVC"
    assert evaluation_rows[0]["f1_macro"] == "0.8123456789"
    assert evaluation_rows[0]["selected"] == "True"
    assert evaluation_rows[0]["smoke"] == "False"
    assert evaluation_rows[1]["classifier_name"] == "RandomForestClassifier"
    assert evaluation_rows[1]["auc_roc"] == ""
    assert fold_rows[1]["outer_split"] == "1"
    assert fold_rows[1]["outer_fold"] == "1"
    assert fold_rows[1]["f1_macro"] == "0.8234567891"
    assert fold_rows[1]["selected"] == "True"
    assert len(optimization_rows) == 12
    assert [
        (
            row["classifier_name"],
            row["search_scope"],
            row["outer_split"],
        )
        for row in optimization_rows[:4]
    ] == [
        ("SVC", "outer", "0"),
        ("SVC", "outer", "1"),
        ("RandomForestClassifier", "outer", "0"),
        ("SVC", "final", ""),
    ]
    assert optimization_rows[0]["selected"] == "True"
    assert optimization_rows[2]["selected"] == "False"
    assert optimization_rows[3]["outer_repeat"] == ""
    assert optimization_rows[3]["outer_fold"] == ""
    assert optimization_rows[3]["selected"] == ""
    assert optimization_rows[3]["best_inner_f1_macro"] == "0.8333333333333333"
    assert optimization_rows[3]["n_trials_requested"] == "100"
    assert optimization_rows[3]["n_trials_complete"] == "98"
    assert optimization_rows[3]["n_trials_failed"] == "2"
    assert optimization_rows[3]["smoke"] == "False"
    assert json.loads(optimization_rows[3]["best_params"]) == {
        "C": 1.0,
        "kernel": "linear",
    }
    assert main.optimization_rows[-1]["search_scope"] == "final"
    assert len(figure_calls) == 1
    assert figure_calls[0]["rows"] is main.evaluation_fold_rows
    assert figure_calls[0]["smoke"] is False


def test_main_persists_exact_final_parameters(monkeypatch, tmp_path):
    final_params = {
        "learning_rate": 0.123456789012345,
        "max_depth": 7,
        "gamma": 0.0123456789012345,
    }
    final_inner_f1_macro = 0.8567890123456789

    class XGBClassifier:
        pass

    class ExactParamsPipeline:
        def __init__(self, config):
            self.config = config

        def run_with_evaluations(self, *args):
            selected_result = (
                XGBClassifier(),
                final_params,
                0.8123456789,
                0.0123456789,
                0.8234567891,
                0.0234567891,
                0.9345678912,
                0.0345678912,
            )
            evaluation = {
                "classifier_key": "xgb",
                "f1_macro": 0.8123456789,
                "f1_std": 0.0123456789,
                "accuracy": 0.8234567891,
                "acc_std": 0.0234567891,
                "auc_roc": 0.9345678912,
                "auc_std": 0.0345678912,
                "outer_scores": [],
                "optimization_searches": [
                    {
                        "outer_split": 0,
                        "outer_repeat": 0,
                        "outer_fold": 0,
                        "best_params": final_params,
                        "best_inner_f1_macro": 0.8456789012345678,
                        "n_trials_requested": 37,
                        "n_trials_complete": 35,
                        "n_trials_failed": 2,
                    }
                ],
                "final_optimization_search": OptimizationResult(
                    best_params=final_params,
                    best_inner_f1_macro=final_inner_f1_macro,
                    n_trials_requested=37,
                    n_trials_complete=36,
                    n_trials_failed=1,
                ),
                "selected": True,
            }
            return selected_result, [evaluation]

    monkeypatch.setattr(main_module, "Pipeline", ExactParamsPipeline)
    raw_dir = tmp_path / "raw"
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    label_path, y = _write_labels(raw_dir)
    dataset_path = _write_npz(data_dir, "sample_tabular", np.ones((8, 3)), y)
    config_path = tmp_path / "config.toml"
    config_path.write_text(f'''
[paths]
input = "{raw_dir.as_posix()}/"
dataset = "{data_dir.as_posix()}/"
output = "{output_dir.as_posix()}/"

[datasets.sample_tabular]
path = "{dataset_path.as_posix()}"
label_path = "{label_path.as_posix()}"

[classifiers]
active = ["xgb"]

[optimization]
n_trials = 37
''')

    Main(user_config=config_path).run()

    row = _rows(output_dir / "results.csv")[0]
    persisted_params = ast.literal_eval(row["parameters"])
    assert persisted_params == final_params
    optimization_rows = _rows(output_dir / "optimization_searches.csv")
    assert [row["search_scope"] for row in optimization_rows] == ["outer", "final"]
    assert json.loads(optimization_rows[-1]["best_params"]) == persisted_params
    assert optimization_rows[-1]["best_inner_f1_macro"] == str(final_inner_f1_macro)
    assert optimization_rows[-1]["outer_split"] == ""
    assert optimization_rows[-1]["outer_repeat"] == ""
    assert optimization_rows[-1]["outer_fold"] == ""
    assert optimization_rows[-1]["selected"] == ""
    assert row["f1_macro"] == "0.8123"
    assert row["f1_std"] == "0.0123"
    assert row["accuracy"] == "0.8235"
    assert row["acc_std"] == "0.0235"
    assert row["auc_roc"] == "0.9346"
    assert row["auc_std"] == "0.0346"
    report = (output_dir / "results.txt").read_text(encoding="utf-8")
    assert f"Best classifier parameters: {final_params}" in report

    provenance = json.loads(
        (output_dir / "optimization_provenance.json").read_text(encoding="utf-8")
    )
    assert set(provenance) == {
        "melite_version",
        "optimization_backend",
        "smoke",
        "random_state",
        "active_classifiers",
        "cv",
        "optimization",
        "search_spaces",
    }
    assert provenance["melite_version"] == __version__
    assert provenance["optimization_backend"]["name"] == "optuna"
    assert provenance["optimization_backend"]["version"]
    assert provenance["smoke"] is False
    assert provenance["random_state"] == 42
    assert provenance["active_classifiers"] == ["xgb"]
    assert provenance["cv"] == {
        "n_splits": 5,
        "n_repeats": 3,
        "inner_n_splits": 3,
    }
    assert provenance["optimization"]["effective_n_trials"] == 37
    assert set(provenance["optimization"]["policy"]) == {
        "sampler",
        "n_startup_trials",
        "smoke_n_trials",
        "multivariate",
        "group",
        "constant_liar",
        "pruning",
        "storage",
        "n_jobs",
        "direction",
        "objective",
    }
    assert provenance["search_spaces"] == {"xgb": get_search_space("xgb").to_dict()}
    assert "schema_version" not in provenance
    assert str(tmp_path) not in json.dumps(provenance)


def test_clean_params_normalizes_numpy_scalars_without_loss():
    params = {
        "learning_rate": np.float64(0.123456789012345),
        "max_depth": np.int64(7),
        "enabled": np.bool_(True),
    }

    normalized = Main._clean_params(params)

    assert normalized == {
        "learning_rate": 0.123456789012345,
        "max_depth": 7,
        "enabled": True,
    }
    assert ast.literal_eval(str(normalized)) == normalized
    assert type(normalized["learning_rate"]) is float
    assert type(normalized["max_depth"]) is int
    assert type(normalized["enabled"]) is bool


def test_smoke_provenance_uses_effective_budget_and_canonical_seed():
    main = Main.__new__(Main)
    main.config = Config(smoke=True)

    provenance = main._optimization_provenance()

    assert provenance["smoke"] is True
    assert provenance["optimization"]["effective_n_trials"] == 5
    assert provenance["random_state"] == main.config.RANDOM_STATE


def test_stack_only_writes_header_only_optimization_artifact(monkeypatch, tmp_path):
    class StackingClassifier:
        def fit(self, X, y):
            return self

    class StackPipeline:
        def __init__(self, config):
            self.config = config

        def run_with_evaluations(self, *args):
            selected_result = (
                StackingClassifier(),
                {},
                0.8,
                0.01,
                0.82,
                0.02,
                0.9,
                0.03,
            )
            evaluation = {
                "classifier_key": "stack",
                "f1_macro": 0.8,
                "f1_std": 0.01,
                "accuracy": 0.82,
                "acc_std": 0.02,
                "auc_roc": 0.9,
                "auc_std": 0.03,
                "outer_scores": [],
                "optimization_searches": [],
                "final_optimization_search": None,
                "selected": True,
            }
            return selected_result, [evaluation]

    monkeypatch.setattr(main_module, "Pipeline", StackPipeline)
    monkeypatch.setattr(
        main_module.ResultManager,
        "write_evaluation_figures",
        lambda self, rows, smoke=False: None,
    )
    raw_dir = tmp_path / "raw"
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    label_path, y = _write_labels(raw_dir)
    dataset_path = _write_npz(data_dir, "sample_tabular", np.ones((8, 3)), y)
    config_path = tmp_path / "config.toml"
    config_path.write_text(f'''
[paths]
input = "{raw_dir.as_posix()}/"
dataset = "{data_dir.as_posix()}/"
output = "{output_dir.as_posix()}/"

[datasets.sample_tabular]
path = "{dataset_path.as_posix()}"
label_path = "{label_path.as_posix()}"

[classifiers]
active = ["stack"]
''')

    Main(user_config=config_path).run()

    optimization_path = output_dir / "optimization_searches.csv"
    with open(optimization_path, encoding="utf-8") as f:
        assert list(csv.DictReader(f)) == []
    result_row = _rows(output_dir / "results.csv")[0]
    assert ast.literal_eval(result_row["parameters"]) == {}
    provenance = json.loads(
        (output_dir / "optimization_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["active_classifiers"] == ["stack"]
    assert provenance["search_spaces"] == {"stack": None}


def test_main_run_uses_legacy_registry_metadata(monkeypatch, tmp_path):
    DummyPipeline.calls = []
    monkeypatch.setattr(main_module, "Pipeline", DummyPipeline)

    raw_dir = tmp_path / "raw"
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    label_path, y = _write_labels(raw_dir)
    _write_npz(data_dir, "PCA70", np.ones((8, 2)), y)
    config_path = tmp_path / "legacy.toml"
    config_path.write_text(f'''
[paths]
input = "{raw_dir.as_posix()}/"
dataset = "{data_dir.as_posix()}/"
output = "{output_dir.as_posix()}/"

[benchmark]
reduction_types = ["PCA"]
levels = [70]
''')

    Main(user_config=config_path).run()

    rows = _rows(output_dir / "results.csv")
    assert rows[0]["dataset"] == "PCA70"
    assert rows[0]["family"] == "dimensionality"
    assert rows[0]["method"] == "PCA"
    assert rows[0]["reduction_type"] == "PCA"
    assert rows[0]["level"] == "70"
    assert DummyPipeline.calls == [
        {
            "shape": (8, 2),
            "n_labels": 8,
            "reduction_type": "PCA",
            "level": 70,
        }
    ]


def test_main_run_missing_registered_dataset_fails(monkeypatch, tmp_path):
    DummyPipeline.calls = []
    monkeypatch.setattr(main_module, "Pipeline", DummyPipeline)

    raw_dir = tmp_path / "raw"
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    label_path, _ = _write_labels(raw_dir)
    missing_path = data_dir / "missing.npz"
    config_path = tmp_path / "missing.toml"
    config_path.write_text(f'''
[paths]
input = "{raw_dir.as_posix()}/"
dataset = "{data_dir.as_posix()}/"
output = "{output_dir.as_posix()}/"

[datasets.maccs]
path = "{missing_path.as_posix()}"
label_path = "{label_path.as_posix()}"
family = "fingerprints"
''')

    with pytest.raises(FileNotFoundError, match="maccs"):
        Main(user_config=config_path).run()
    assert DummyPipeline.calls == []
