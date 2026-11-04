# SPDX-License-Identifier: LGPL-3.0-or-later
"""Failure characterization for MELITE's three-class classification contract."""

import csv
import json
import subprocess
import sys
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from melite.classification_metrics import classification_metric_policy
from melite.export_best_model import Finalizer
from melite.load_dataset import load_datasets
from melite.main import Main
from melite.model_training import MultiModelTrainer
from melite.predict import predict


@pytest.fixture(scope="module")
def three_class_data():
    """Return deterministic numerical data with original labels 1, 2, and 3."""
    X, encoded_y = make_classification(
        n_samples=60,
        n_features=6,
        n_informative=5,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=1.5,
        flip_y=0.0,
        random_state=42,
    )
    return X, encoded_y + 1


def _config(active_classifier):
    return SimpleNamespace(
        ACTIVE_CLASSIFIERS=[active_classifier],
        RANDOM_STATE=42,
        N_TRIALS=1,
        CV_CONFIG={"n_splits": 2, "n_repeats": 1, "inner_n_splits": 2},
    )


def _quick_stacking_model(builder):
    return builder().set_params(
        rf__n_estimators=3,
        xgb__n_estimators=3,
        xgb__max_depth=2,
    )


@pytest.fixture(scope="module")
def conventional_outer_evaluation(three_class_data):
    X, y = three_class_data
    trainer = MultiModelTrainer(_config("lr"))
    model = trainer.model_builders["lr"]().set_params(lr__C=1.0)
    return trainer._cross_validate_model_with_scores(model, X, y)


def test_registered_dataset_preserves_original_multiclass_labels(
    tmp_path, three_class_data
):
    X, y = three_class_data
    data_path = tmp_path / "three_class.npz"
    label_path = tmp_path / "labels.npy"
    np.savez(data_path, X=X, y=y)
    np.save(label_path, y)
    config = SimpleNamespace(
        DATASETS={
            "three_class": {
                "path": str(data_path),
                "label_path": str(label_path),
                "task": "classification",
                "metadata": {},
            }
        }
    )

    loaded = load_datasets(config)["three_class"]

    assert loaded["task"] == "classification"
    assert np.array_equal(loaded["y"], y)
    assert set(np.unique(loaded["y"])) == {1, 2, 3}


def test_conventional_outer_evaluation_reports_complete_multiclass_metrics(
    conventional_outer_evaluation,
):
    evaluation = conventional_outer_evaluation

    assert np.isfinite(evaluation["f1_macro"])
    assert np.isfinite(evaluation["accuracy"])
    assert np.isfinite(evaluation["auc_roc"])
    assert all(np.isfinite(fold["auc_roc"]) for fold in evaluation["outer_scores"])


def test_binary_roc_auc_preserves_existing_definition():
    X, y = make_classification(
        n_samples=60,
        n_features=6,
        n_informative=5,
        n_redundant=0,
        random_state=42,
    )
    trainer = MultiModelTrainer(_config("lr"))
    model = trainer.model_builders["lr"]().set_params(lr__C=1.0)

    evaluation = trainer._cross_validate_model_with_scores(model, X, y)
    expected_fold_auc = []
    for train_index, validation_index in trainer._build_outer_cv().split(X, y):
        fitted = clone(model).fit(X[train_index], y[train_index])
        expected_fold_auc.append(
            roc_auc_score(
                y[validation_index],
                fitted.decision_function(X[validation_index]),
            )
        )

    assert evaluation["auc_roc"] == pytest.approx(np.mean(expected_fold_auc))
    assert [fold["auc_roc"] for fold in evaluation["outer_scores"]] == pytest.approx(
        expected_fold_auc
    )


def test_classification_metric_policy_is_explicit_and_serializable():
    policy = classification_metric_policy()

    assert policy["optimization_and_selection"] == {
        "metric": "f1_macro",
        "direction": "maximize",
    }
    assert policy["reported"]["roc_auc"] == {
        "binary": {"scorer": "roc_auc"},
        "multiclass": {
            "strategy": "ovo",
            "average": "macro",
            "response_method": "predict_proba",
        },
    }
    assert policy["per_class"]["undefined_metric_convention"] == {
        "zero_division": 0,
        "description": (
            "Undefined precision, recall, and F1-score values are reported as 0."
        ),
    }
    assert policy["per_class"]["summary_aggregation"] == {
        "unit": "outer_fold",
        "mean": "arithmetic",
        "standard_deviation": {"ddof": 0},
        "support": "mean_and_standard_deviation_across_outer_folds",
    }


def test_multiclass_roc_auc_is_ovo_macro_probability_score(
    three_class_data, conventional_outer_evaluation
):
    X, y = three_class_data
    trainer = MultiModelTrainer(_config("lr"))
    model = trainer.model_builders["lr"]().set_params(lr__C=1.0)
    expected_fold_auc = []
    for train_index, validation_index in trainer._build_outer_cv().split(X, y):
        fitted = clone(model).fit(X[train_index], y[train_index])
        expected_fold_auc.append(
            roc_auc_score(
                y[validation_index],
                fitted.predict_proba(X[validation_index]),
                labels=fitted.classes_,
                multi_class="ovo",
                average="macro",
            )
        )

    assert conventional_outer_evaluation["auc_roc"] == pytest.approx(
        np.mean(expected_fold_auc)
    )
    assert [
        fold["auc_roc"] for fold in conventional_outer_evaluation["outer_scores"]
    ] == pytest.approx(expected_fold_auc)


def test_multiclass_outer_folds_retain_per_class_evidence(
    conventional_outer_evaluation,
):
    expected_labels = {1, 2, 3}

    for fold in conventional_outer_evaluation["outer_scores"]:
        assert "per_class_metrics" in fold
        assert set(fold["per_class_metrics"]) == expected_labels
        assert all(
            set(metrics) == {"precision", "recall", "f1", "support"}
            for metrics in fold["per_class_metrics"].values()
        )


def test_per_class_outer_evidence_matches_independent_reference_calculation(
    three_class_data, conventional_outer_evaluation
):
    X, y = three_class_data
    trainer = MultiModelTrainer(_config("lr"))
    model = trainer.model_builders["lr"]().set_params(lr__C=1.0)

    for fold, (train_indices, test_indices) in zip(
        conventional_outer_evaluation["outer_scores"],
        trainer._build_outer_cv().split(X, y),
    ):
        fitted = clone(model).fit(X[train_indices], y[train_indices])
        precision, recall, f1, support = precision_recall_fscore_support(
            y[test_indices],
            fitted.predict(X[test_indices]),
            labels=[1, 2, 3],
            zero_division=0,
        )

        assert list(fold["per_class_metrics"]) == [1, 2, 3]
        for index, class_label in enumerate((1, 2, 3)):
            metrics = fold["per_class_metrics"][class_label]
            assert metrics["precision"] == pytest.approx(precision[index])
            assert metrics["recall"] == pytest.approx(recall[index])
            assert metrics["f1"] == pytest.approx(f1[index])
            assert metrics["support"] == support[index]
        assert sum(
            metrics["support"] for metrics in fold["per_class_metrics"].values()
        ) == len(test_indices)
        assert fold["f1_macro"] == pytest.approx(np.mean(f1))


def test_four_class_outer_evaluation_retains_all_original_class_evidence():
    X, encoded_y = make_classification(
        n_samples=64,
        n_features=6,
        n_informative=5,
        n_redundant=0,
        n_classes=4,
        n_clusters_per_class=1,
        class_sep=1.5,
        flip_y=0.0,
        random_state=42,
    )
    y = encoded_y + 1
    trainer = MultiModelTrainer(_config("lr"))
    model = trainer.model_builders["lr"]().set_params(lr__C=1.0)

    evaluation = trainer._cross_validate_model_with_scores(model, X, y)

    for fold in evaluation["outer_scores"]:
        assert list(fold["per_class_metrics"]) == [1, 2, 3, 4]
        assert (
            sum(metrics["support"] for metrics in fold["per_class_metrics"].values())
            == 32
        )
        assert fold["f1_macro"] == pytest.approx(
            np.mean([metrics["f1"] for metrics in fold["per_class_metrics"].values()])
        )


def test_binary_outer_evaluation_retains_compatible_per_class_evidence():
    X, y = make_classification(
        n_samples=40,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        random_state=42,
    )
    trainer = MultiModelTrainer(_config("lr"))
    model = trainer.model_builders["lr"]().set_params(lr__C=1.0)

    evaluation = trainer._cross_validate_model_with_scores(model, X, y)

    for fold in evaluation["outer_scores"]:
        assert list(fold["per_class_metrics"]) == [0, 1]
        assert (
            sum(metrics["support"] for metrics in fold["per_class_metrics"].values())
            == 20
        )
        assert fold["f1_macro"] == pytest.approx(
            np.mean([metrics["f1"] for metrics in fold["per_class_metrics"].values()])
        )


def test_per_class_artifacts_and_metric_provenance_preserve_dataset_identities(
    tmp_path, monkeypatch
):
    output_path = tmp_path / "output"
    dataset_specs = {
        "three_classes": (np.repeat([1, 2, 3], 4), 3),
        "four_classes": (np.repeat([10, 20, 30, 40], 4), 4),
    }
    config_lines = [f'[paths]\noutput = "{output_path.as_posix()}/"\n']
    for dataset_id, (labels, _) in dataset_specs.items():
        dataset_path = tmp_path / f"{dataset_id}.csv"
        with open(dataset_path, "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["feature_a", "feature_b", "target"])
            for index, label in enumerate(labels):
                writer.writerow([float(index), float(index % 3), int(label)])
        config_lines.append(
            f"[datasets.{dataset_id}]\n"
            f'path = "{dataset_path.as_posix()}"\n'
            'label_column = "target"\n'
            'task = "classification"\n'
        )
    config_lines.append('[classifiers]\nactive = ["stack"]\n')
    config_path = tmp_path / "config.toml"
    config_path.write_text("\n".join(config_lines), encoding="utf-8")

    class EvidencePipeline:
        @staticmethod
        def run_with_evaluations(X, y, _dataset_id, _level):
            labels = np.unique(y)
            outer_scores = []
            for outer_fold in range(2):
                outer_scores.append(
                    {
                        "outer_split": outer_fold,
                        "outer_repeat": 0,
                        "outer_fold": outer_fold,
                        "f1_macro": 1.0,
                        "accuracy": 1.0,
                        "auc_roc": 1.0,
                        "per_class_metrics": {
                            label: {
                                "precision": 1.0,
                                "recall": 1.0,
                                "f1": 1.0,
                                "support": int(np.sum(y == label) // 2),
                            }
                            for label in labels
                        },
                    }
                )
            selected_result = (object(), {}, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0)
            return selected_result, [
                {
                    "classifier_key": "stack",
                    "f1_macro": 1.0,
                    "f1_std": 0.0,
                    "accuracy": 1.0,
                    "acc_std": 0.0,
                    "auc_roc": 1.0,
                    "auc_std": 0.0,
                    "outer_scores": outer_scores,
                    "optimization_searches": [],
                    "final_optimization_search": None,
                    "selected": True,
                }
            ]

    main = Main(user_config=config_path)
    main.pipeline = EvidencePipeline()
    monkeypatch.setattr(
        main.result_manager, "write_evaluation_figures", lambda *_args, **_kwargs: None
    )

    main.run()

    with open(output_path / "evaluation_class_folds.csv", encoding="utf-8") as stream:
        fold_reader = csv.DictReader(stream)
        fold_rows = list(fold_reader)
    with open(output_path / "evaluation_class_summary.csv", encoding="utf-8") as stream:
        summary_reader = csv.DictReader(stream)
        summary_rows = list(summary_reader)
    provenance = json.loads(
        (output_path / "classification_metric_provenance.json").read_text(
            encoding="utf-8"
        )
    )

    assert fold_reader.fieldnames == [
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
        "class_label",
        "precision",
        "recall",
        "f1_score",
        "support",
        "selected",
        "smoke",
    ]
    assert summary_reader.fieldnames == [
        "dataset",
        "family",
        "method",
        "variant",
        "level",
        "description",
        "reduction_type",
        "classifier_name",
        "class_label",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "f1_score_mean",
        "f1_score_std",
        "support_mean",
        "support_std",
        "n_outer_folds",
        "selected",
        "smoke",
    ]
    assert len(fold_rows) == 2 * (3 + 4)
    assert len(summary_rows) == 3 + 4
    assert {row["n_outer_folds"] for row in summary_rows} == {"2"}
    assert {row["support_mean"] for row in summary_rows} == {"2.0"}
    assert provenance["prediction_task"] == "classification"
    assert provenance["optimization_and_selection"] == {
        "metric": "f1_macro",
        "direction": "maximize",
    }
    assert provenance["reported"]["roc_auc"]["multiclass"] == {
        "strategy": "ovo",
        "average": "macro",
        "response_method": "predict_proba",
    }
    assert provenance["per_class"]["undefined_metric_convention"]["zero_division"] == 0
    assert provenance["datasets"] == [
        {
            "dataset": "three_classes",
            "number_of_classes": 3,
            "class_labels": [1, 2, 3],
            "evaluation_mode": "multiclass",
            "roc_auc_definition": "multiclass_ovo_macro_probability",
        },
        {
            "dataset": "four_classes",
            "number_of_classes": 4,
            "class_labels": [10, 20, 30, 40],
            "evaluation_mode": "multiclass",
            "roc_auc_definition": "multiclass_ovo_macro_probability",
        },
    ]
    with open(output_path / "evaluations.csv", encoding="utf-8") as stream:
        assert list(csv.DictReader(stream))[0]["f1_macro"] == "1.0"


def test_public_multiclass_workflow_run_export_predict_and_artifact_contract(
    tmp_path,
):
    X, encoded_y = make_classification(
        n_samples=36,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=1.5,
        flip_y=0.0,
        random_state=42,
    )
    y = encoded_y + 1
    dataset_path = tmp_path / "three_class.csv"
    output_path = tmp_path / "output"
    with open(dataset_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["feature_1", "feature_2", "feature_3", "feature_4", "label"])
        for features, label in zip(X, y):
            writer.writerow([*features, int(label)])

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f'''[paths]
input = "{tmp_path.as_posix()}/"
dataset = "{tmp_path.as_posix()}/"
output = "{output_path.as_posix()}/"

[benchmark]
random_state = 42

[datasets.three_class]
path = "{dataset_path.as_posix()}"
label_column = "label"
task = "classification"

[cv]
n_splits = 2
n_repeats = 1
inner_n_splits = 2

[classifiers]
active = ["lr", "xgb", "stack"]

[regressors]
active = []

[optimization]
n_trials = 1
''',
        encoding="utf-8",
    )

    run_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "melite.cli",
            "run",
            "--config",
            str(config_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr

    required_artifacts = {
        "results.csv",
        "evaluations.csv",
        "evaluation_folds.csv",
        "optimization_searches.csv",
        "optimization_provenance.json",
        "evaluation_class_folds.csv",
        "evaluation_class_summary.csv",
        "classification_metric_provenance.json",
    }
    assert required_artifacts.issubset({path.name for path in output_path.iterdir()})

    def read_rows(filename):
        with open(output_path / filename, newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    result_rows = read_rows("results.csv")
    evaluation_rows = read_rows("evaluations.csv")
    fold_rows = read_rows("evaluation_folds.csv")
    class_fold_rows = read_rows("evaluation_class_folds.csv")
    class_summary_rows = read_rows("evaluation_class_summary.csv")
    optimization_rows = read_rows("optimization_searches.csv")

    assert len(result_rows) == 1
    selected_name = result_rows[0]["classifier_name"]
    assert len(evaluation_rows) == 3
    assert len(fold_rows) == 6
    assert len(class_fold_rows) == 18
    assert len(class_summary_rows) == 9
    outer_optimization_rows = [
        row for row in optimization_rows if row["search_scope"] == "outer"
    ]
    final_optimization_rows = [
        row for row in optimization_rows if row["search_scope"] == "final"
    ]
    assert len(outer_optimization_rows) == 4
    assert {row["classifier_name"] for row in outer_optimization_rows} == {
        "LogisticRegression",
        "XGBClassifier",
    }
    assert len(final_optimization_rows) == (
        1 if selected_name in {"LogisticRegression", "XGBClassifier"} else 0
    )
    assert sum(row["selected"] == "True" for row in evaluation_rows) == 1

    for evaluation in evaluation_rows:
        classifier_folds = [
            row
            for row in fold_rows
            if row["classifier_name"] == evaluation["classifier_name"]
        ]
        assert len(classifier_folds) == 2
        for metric, aggregate, standard_deviation in (
            ("f1_macro", "f1_macro", "f1_std"),
            ("accuracy", "accuracy", "acc_std"),
            ("auc_roc", "auc_roc", "auc_std"),
        ):
            values = np.array([float(row[metric]) for row in classifier_folds])
            assert float(evaluation[aggregate]) == pytest.approx(values.mean())
            assert float(evaluation[standard_deviation]) == pytest.approx(
                values.std(ddof=0)
            )

        for fold in classifier_folds:
            matching_class_rows = [
                row
                for row in class_fold_rows
                if row["classifier_name"] == evaluation["classifier_name"]
                and row["outer_split"] == fold["outer_split"]
            ]
            assert [row["class_label"] for row in matching_class_rows] == [
                "1",
                "2",
                "3",
            ]
            assert sum(int(row["support"]) for row in matching_class_rows) == 18
            assert float(fold["f1_macro"]) == pytest.approx(
                np.mean([float(row["f1_score"]) for row in matching_class_rows])
            )

    for summary in class_summary_rows:
        matching_class_folds = [
            row
            for row in class_fold_rows
            if row["classifier_name"] == summary["classifier_name"]
            and row["class_label"] == summary["class_label"]
        ]
        assert len(matching_class_folds) == int(summary["n_outer_folds"]) == 2
        for metric in ("precision", "recall", "f1_score", "support"):
            values = np.array(
                [float(row[metric]) for row in matching_class_folds], dtype=float
            )
            assert float(summary[f"{metric}_mean"]) == pytest.approx(values.mean())
            assert float(summary[f"{metric}_std"]) == pytest.approx(values.std(ddof=0))

    optimization_provenance = json.loads(
        (output_path / "optimization_provenance.json").read_text(encoding="utf-8")
    )
    assert optimization_provenance["active_classifiers"] == ["lr", "xgb", "stack"]
    metric_provenance = json.loads(
        (output_path / "classification_metric_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert metric_provenance["datasets"] == [
        {
            "dataset": "three_class",
            "number_of_classes": 3,
            "class_labels": [1, 2, 3],
            "evaluation_mode": "multiclass",
            "roc_auc_definition": "multiclass_ovo_macro_probability",
        }
    ]

    export_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "melite.cli",
            "export",
            "--config",
            str(config_path),
            "--row",
            "0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert export_result.returncode == 0, export_result.stdout + export_result.stderr

    model_path = output_path / f"Model_{selected_name}_three_class.pkl"
    assert model_path.exists()
    fitted_model = joblib.load(model_path)
    prediction_result = predict(model_path, X[:6])

    assert np.array_equal(fitted_model.classes_, np.array([1, 2, 3]))
    assert set(prediction_result) == {
        "predictions",
        "probabilities",
        "model_path",
        "n_samples",
    }
    assert prediction_result["predictions"].shape == (6,)
    assert set(np.unique(prediction_result["predictions"])).issubset({1, 2, 3})
    probabilities = prediction_result["probabilities"]
    assert probabilities.shape == (6, 3)
    assert np.all(np.isfinite(probabilities))
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.allclose(probabilities, fitted_model.predict_proba(X[:6]))
    assert np.array_equal(
        prediction_result["predictions"],
        fitted_model.classes_[np.argmax(probabilities, axis=1)],
    )


def test_nested_conventional_multiclass_workflow_completes(three_class_data):
    X, y = three_class_data
    trainer = MultiModelTrainer(_config("lr"))

    selected, evaluations = trainer.evaluate_and_select_models(
        X, y, "three_class", None
    )

    assert len(evaluations[0]["outer_scores"]) == 2
    assert np.isfinite(selected[2])
    assert np.isfinite(selected[4])
    assert np.isfinite(selected[6])
    assert np.array_equal(selected[0].classes_, np.array([1, 2, 3]))


def test_xgboost_multiclass_workflow_preserves_original_class_identity(
    three_class_data,
):
    X, y = three_class_data
    trainer = MultiModelTrainer(_config("xgb"))

    selected, evaluations = trainer.evaluate_and_select_models(
        X, y, "three_class", None
    )
    model = selected[0]
    probabilities = model.predict_proba(X[:6])

    assert len(evaluations[0]["outer_scores"]) == 2
    assert np.array_equal(model.classes_, np.array([1, 2, 3]))
    assert probabilities.shape == (6, 3)
    assert np.array_equal(
        model.predict(X[:6]), model.classes_[np.argmax(probabilities, axis=1)]
    )


def test_multiclass_svc_uses_fold_local_probabilities_without_changing_predictions(
    three_class_data,
):
    X, y = three_class_data
    trainer = MultiModelTrainer(_config("svc"))

    selected, evaluations = trainer.evaluate_and_select_models(
        X, y, "three_class", None
    )
    model, parameters = selected[:2]
    probabilities = model.predict_proba(X[:6])

    assert np.isfinite(evaluations[0]["auc_roc"])
    assert np.array_equal(model.classes_, np.array([1, 2, 3]))
    assert probabilities.shape == (6, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert set(parameters).issubset({"svc__kernel", "svc__C", "svc__gamma"})
    calibrated_svc = model.named_steps["svc"]
    assert np.array_equal(calibrated_svc.predictor_.classes_, model.classes_)
    assert np.array_equal(calibrated_svc.calibrator_.classes_, model.classes_)
    assert set(np.unique(model.predict(X[:6]))).issubset({1, 2, 3})


def test_stacking_multiclass_workflow_preserves_labels_and_probability_alignment(
    three_class_data,
):
    X, y = three_class_data
    trainer = MultiModelTrainer(_config("stack"))
    stacking_builder = trainer.model_builders["stack"]
    trainer.model_builders["stack"] = lambda: _quick_stacking_model(stacking_builder)

    selected, evaluations = trainer.evaluate_and_select_models(
        X, y, "three_class", None
    )
    model = selected[0]
    evaluation = evaluations[0]
    probabilities = model.predict_proba(X[:6])

    assert np.isfinite(evaluation["f1_macro"])
    assert np.isfinite(evaluation["accuracy"])
    assert np.array_equal(model.classes_, np.array([1, 2, 3]))
    assert probabilities.shape == (6, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.array_equal(
        model.predict(X[:6]), model.classes_[np.argmax(probabilities, axis=1)]
    )
    assert np.isfinite(evaluation["auc_roc"])


@pytest.mark.parametrize(
    ("classifier_name", "parameters"),
    [
        ("LogisticRegression", {"lr__C": 1.0}),
        (
            "XGBClassifier",
            {"n_estimators": 3, "max_depth": 2, "learning_rate": 0.2},
        ),
        (
            "StackingClassifier",
            {
                "rf__n_estimators": 3,
                "xgb__n_estimators": 3,
                "xgb__max_depth": 2,
            },
        ),
    ],
)
def test_reconstructed_multiclass_model_inference_aligns_probability_columns(
    tmp_path,
    three_class_data,
    classifier_name,
    parameters,
):
    X, y = three_class_data
    model = Finalizer._build_model(
        classifier_name,
        str(parameters),
        random_state=42,
        cv_config={"n_splits": 2, "n_repeats": 1, "inner_n_splits": 2},
    )
    model.fit(X, y)
    artifact = tmp_path / f"Model_{classifier_name}_three_class.pkl"
    joblib.dump(model, artifact)

    result = predict(artifact, X[:6])
    fitted_model = joblib.load(artifact)

    assert np.array_equal(fitted_model.classes_, np.array([1, 2, 3]))
    assert set(np.unique(result["predictions"])).issubset({1, 2, 3})
    assert result["probabilities"].shape == (6, 3)
    assert np.allclose(result["probabilities"].sum(axis=1), 1.0)
    assert np.array_equal(
        result["predictions"],
        fitted_model.classes_[np.argmax(result["probabilities"], axis=1)],
    )


@pytest.mark.parametrize(
    ("y", "message"),
    [
        (np.array([1, 1, 1, 1, 2, 2, 2, 2, 3]), "outer cross-validation"),
        (np.repeat([1, 2, 3], 3), "inner cross-validation"),
    ],
)
def test_incompatible_multiclass_distributions_fail_explicitly(y, message):
    X = np.arange(len(y) * 3, dtype=float).reshape(len(y), 3)
    trainer = MultiModelTrainer(_config("lr"))

    with pytest.raises(ValueError, match=message):
        trainer.evaluate_and_select_models(X, y, "three_class", None)
