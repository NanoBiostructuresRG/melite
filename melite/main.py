# SPDX-License-Identifier: LGPL-3.0-or-later
"""Main evaluation pipeline for MELITE.

This module implements the end-to-end evaluation workflow: dataset loading,
multi-classifier evaluation and selection with nested cross-validation, and result
writing. It is invoked via ``melite run`` from the unified CLI.
"""

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from melite.classification_metrics import (
    analyze_classification_target,
    classification_metric_policy,
)
from melite.config import Config
from melite.load_dataset import load_datasets
from melite.model_catalog import MODEL_CATALOG
from melite.model_training import MultiModelTrainer
from melite.optimization import get_optimization_backend_info, optuna_logging_scope
from melite.optimization_policy import OPTIMIZATION_POLICY
from melite.regression_optimization import REGRESSION_OPTIMIZATION_DIRECTION
from melite.regression_search_spaces import get_regression_search_space
from melite.regression_training import RegressionModelTrainer
from melite.result_manager import ResultManager
from melite.search_spaces import get_search_space
from melite.version import __version__

logger = logging.getLogger(__name__)

_SMOKE_WARNING = (
    "\n[SMOKE TEST] Using reduced search and cross-validation settings. "
    "Results are not suitable for final classifier selection.\n"
)
_TASK_SMOKE_WARNING = (
    "\n[SMOKE TEST] Using reduced search and cross-validation settings. "
    "Results are not suitable for final model selection.\n"
)

_CLASSIFIER_NAMES = {
    model.key: model.public_name
    for model in MODEL_CATALOG
    if model.prediction_task == "classification" and model.availability == "implemented"
}
_REGRESSOR_NAMES = {
    model.key: model.public_name
    for model in MODEL_CATALOG
    if model.prediction_task == "regression"
}


class Pipeline:
    """Thin wrapper around :class:`~melite.model_training.MultiModelTrainer`.

    Parameters
    ----------
    config : melite.config.Config
        MELITE configuration object.
    """

    def __init__(self, config: Config):
        self.config = config
        self.model_trainer = MultiModelTrainer(config)

    def run(self, X_train, y_train, reduction_type: str, level: int | None):
        """Train all classifiers and return the best result for one dataset.

        Parameters
        ----------
        X_train : numpy.ndarray
            Feature matrix of shape ``(n_samples, n_features)``.
        y_train : numpy.ndarray
            Label vector of shape ``(n_samples,)``.
        reduction_type : str
            Legacy reduction method label when available; otherwise the
            dataset id is passed through for trace logging.
        level : int or None
            Legacy variance retention level when available.

        Returns
        -------
        tuple
            Eight-element tuple as returned by
            :meth:`~melite.model_training.MultiModelTrainer.train_and_select_best_model`.
        """
        return self.model_trainer.train_and_select_best_model(
            X_train, y_train, reduction_type, level
        )

    def run_with_evaluations(
        self, X_train, y_train, reduction_type: str, level: int | None
    ):
        """Return the selected result and all classifier evaluations."""
        return self.model_trainer.evaluate_and_select_models(
            X_train, y_train, reduction_type, level
        )


class Main:
    """Orchestrate the full MELITE evaluation pipeline.

    Parameters
    ----------
    smoke : bool, optional
        If ``True``, run in smoke mode with reduced search and cross-validation
        settings.
        Default is ``False``.
    user_config : pathlib.Path or None, optional
        Path to a user-supplied TOML configuration file. Default is ``None``.
    """

    def __init__(self, smoke: bool = False, user_config=None):
        self.config = Config(smoke=smoke, user_config=user_config)
        self.config._setup()
        dataset_tasks = {
            spec.get("task", "classification") for spec in self.config.DATASETS.values()
        }
        self.pipeline = (
            Pipeline(self.config) if "classification" in dataset_tasks else None
        )
        self.regression_model_trainer = (
            RegressionModelTrainer(self.config)
            if "regression" in dataset_tasks
            else None
        )
        self.result_manager = ResultManager(self.config.RESULTS_FILE)
        self.final_results: list[str] = []
        self.csv_rows: list[dict[str, Any]] = []
        self.evaluations_by_dataset: dict[str, list[dict[str, Any]]] = {}
        self.evaluation_rows: list[dict[str, Any]] = []
        self.evaluation_fold_rows: list[dict[str, Any]] = []
        self.evaluation_class_fold_rows: list[dict[str, Any]] = []
        self.evaluation_class_summary_rows: list[dict[str, Any]] = []
        self.classification_metric_datasets: list[dict[str, Any]] = []
        self.optimization_rows: list[dict[str, Any]] = []
        self.regression_final_results: list[str] = []
        self.regression_csv_rows: list[dict[str, Any]] = []
        self.regression_evaluations_by_dataset: dict[str, list[dict[str, Any]]] = {}
        self.regression_evaluation_rows: list[dict[str, Any]] = []
        self.regression_evaluation_fold_rows: list[dict[str, Any]] = []
        self.regression_optimization_rows: list[dict[str, Any]] = []

    @staticmethod
    def _clean_params(params):
        """Return parameter values that round-trip exactly through ast.literal_eval."""
        return {
            k: v.item() if isinstance(v, np.generic) else v for k, v in params.items()
        }

    @staticmethod
    def _legacy_reduction_type(metadata: dict):
        method = metadata.get("method")
        level = metadata.get("level")
        family = metadata.get("family")
        if (
            family == "dimensionality"
            and method in {"PCA", "UMAP"}
            and level is not None
        ):
            return method
        return None

    @staticmethod
    def _selected_evaluation(evaluations):
        selected = [
            evaluation for evaluation in evaluations if evaluation["selected"] is True
        ]
        if len(selected) != 1:
            raise RuntimeError(
                "Expected exactly one selected classifier evaluation; "
                f"found {len(selected)}."
            )
        return selected[0]

    @staticmethod
    def _selected_regression_evaluation(evaluations):
        selected = [
            evaluation for evaluation in evaluations if evaluation["selected"] is True
        ]
        if len(selected) != 1:
            raise RuntimeError(
                "Expected exactly one selected regressor evaluation; "
                f"found {len(selected)}."
            )
        return selected[0]

    def _optimization_provenance(self) -> dict[str, Any]:
        search_spaces = {}
        for classifier_key in self.config.ACTIVE_CLASSIFIERS:
            search_space = get_search_space(classifier_key)
            search_spaces[classifier_key] = (
                search_space.to_dict() if search_space is not None else None
            )

        return {
            "melite_version": __version__,
            "optimization_backend": get_optimization_backend_info(),
            "smoke": self.config.SMOKE,
            "random_state": self.config.RANDOM_STATE,
            "active_classifiers": list(self.config.ACTIVE_CLASSIFIERS),
            "cv": {
                key: self.config.CV_CONFIG[key]
                for key in ("n_splits", "n_repeats", "inner_n_splits")
            },
            "optimization": {
                "effective_n_trials": self.config.N_TRIALS,
                "policy": asdict(OPTIMIZATION_POLICY),
            },
            "search_spaces": search_spaces,
        }

    def _regression_optimization_provenance(self) -> dict[str, Any]:
        search_spaces = {}
        for regressor_key in self.config.ACTIVE_REGRESSORS:
            search_space = get_regression_search_space(regressor_key)
            search_spaces[regressor_key] = (
                {
                    "regressor": search_space.regressor,
                    "parameters": [
                        parameter.to_dict() for parameter in search_space.parameters
                    ],
                }
                if search_space is not None
                else None
            )

        cv_config = self.config.CV_CONFIG
        return {
            "melite_version": __version__,
            "task": "regression",
            "optimization_backend": get_optimization_backend_info(),
            "smoke": self.config.SMOKE,
            "random_state": self.config.RANDOM_STATE,
            "active_regressors": list(self.config.ACTIVE_REGRESSORS),
            "cv": {
                "outer": {
                    "strategy": "RepeatedKFold",
                    "n_splits": cv_config["n_splits"],
                    "n_repeats": cv_config["n_repeats"],
                },
                "inner": {
                    "strategy": "KFold",
                    "n_splits": cv_config["inner_n_splits"],
                    "shuffle": True,
                },
            },
            "optimization": {
                "objective": "rmse",
                "direction": REGRESSION_OPTIMIZATION_DIRECTION,
                "sampler": OPTIMIZATION_POLICY.sampler,
                "effective_n_trials": self.config.N_TRIALS,
                "n_startup_trials": OPTIMIZATION_POLICY.n_startup_trials,
                "pruning": OPTIMIZATION_POLICY.pruning,
                "n_jobs": OPTIMIZATION_POLICY.n_jobs,
            },
            "search_spaces": search_spaces,
        }

    def _classification_metric_provenance(self) -> dict[str, Any]:
        """Return classification metric definitions and per-dataset identities."""
        policy = classification_metric_policy()
        return {
            "melite_version": __version__,
            "prediction_task": "classification",
            "optimization_and_selection": policy["optimization_and_selection"],
            "reported": policy["reported"],
            "per_class": policy["per_class"],
            "datasets": self.classification_metric_datasets,
        }

    def run(self) -> None:
        """Execute the evaluation pipeline for all configured datasets.

        Iterates over the normalized ``config.DATASETS`` registry and dispatches
        each dataset to its configured prediction-task workflow.

        Notes
        -----
        When smoke mode is active, a visible banner is printed to stdout
        regardless of the logging level, to ensure the user is aware that
        results are not suitable for final model selection.
        """
        # The CLI configures MELITE logging before entering Main.run();
        # programmatic callers may configure the logger directly.
        verbose = logger.isEnabledFor(logging.INFO)
        with optuna_logging_scope(verbose=verbose):
            self._run_evaluation()

    def _warn_for_low_optimization_budget(self) -> None:
        datasets = getattr(self.config, "DATASETS", None)
        classification_configured = datasets is None or any(
            spec.get("task", "classification") == "classification"
            for spec in datasets.values()
        )
        regression_configured = datasets is not None and any(
            spec.get("task", "classification") == "regression"
            for spec in datasets.values()
        )
        tunable_classifier = classification_configured and any(
            get_search_space(classifier_key) is not None
            for classifier_key in self.config.ACTIVE_CLASSIFIERS
        )
        tunable_regressor = regression_configured and any(
            get_regression_search_space(regressor_key) is not None
            for regressor_key in getattr(self.config, "ACTIVE_REGRESSORS", [])
        )
        if (
            not self.config.SMOKE
            and self.config.N_TRIALS <= OPTIMIZATION_POLICY.n_startup_trials
            and (tunable_classifier or tunable_regressor)
        ):
            logger.warning(
                "The requested optimization budget remains within TPE startup "
                "sampling and does not reach model-based TPE."
            )

    def _dataset_metadata(self, dataset_id: str, metadata: dict) -> dict[str, Any]:
        return {
            "dataset": dataset_id,
            "family": metadata.get("family"),
            "method": metadata.get("method"),
            "variant": metadata.get("variant"),
            "level": metadata.get("level"),
            "description": metadata.get("description"),
            "reduction_type": self._legacy_reduction_type(metadata),
        }

    def _run_classification_dataset(
        self, dataset_id: str, dataset: dict, dataset_metadata: dict
    ) -> None:
        X_train = dataset["X"]
        y_train = dataset["y"]
        target = analyze_classification_target(y_train)
        self.classification_metric_datasets.append(
            {
                "dataset": dataset_id,
                "number_of_classes": len(target.classes),
                "class_labels": [
                    label.item() if isinstance(label, np.generic) else label
                    for label in target.classes
                ],
                "evaluation_mode": ("multiclass" if target.is_multiclass else "binary"),
                "roc_auc_definition": (
                    "multiclass_ovo_macro_probability"
                    if target.is_multiclass
                    else "binary"
                ),
            }
        )
        reduction_type = dataset_metadata["reduction_type"]
        level = dataset_metadata["level"]
        if self.pipeline is None:
            raise RuntimeError("Classification pipeline was not initialized.")

        selected_result, evaluations = self.pipeline.run_with_evaluations(
            X_train, y_train, reduction_type or dataset_id, level
        )
        self.evaluations_by_dataset[dataset_id] = evaluations

        for evaluation in evaluations:
            evaluation_metadata = {
                **dataset_metadata,
                "classifier_name": _CLASSIFIER_NAMES[evaluation["classifier_key"]],
            }
            self.evaluation_rows.append(
                {
                    **evaluation_metadata,
                    "f1_macro": evaluation["f1_macro"],
                    "f1_std": evaluation["f1_std"],
                    "accuracy": evaluation["accuracy"],
                    "acc_std": evaluation["acc_std"],
                    "auc_roc": evaluation["auc_roc"],
                    "auc_std": evaluation["auc_std"],
                    "selected": evaluation["selected"],
                }
            )
            for outer_score in evaluation["outer_scores"]:
                self.evaluation_fold_rows.append(
                    {
                        **evaluation_metadata,
                        "outer_split": outer_score["outer_split"],
                        "outer_repeat": outer_score["outer_repeat"],
                        "outer_fold": outer_score["outer_fold"],
                        "f1_macro": outer_score["f1_macro"],
                        "accuracy": outer_score["accuracy"],
                        "auc_roc": outer_score["auc_roc"],
                        "selected": evaluation["selected"],
                    }
                )
                for class_label, metrics in outer_score.get(
                    "per_class_metrics", {}
                ).items():
                    self.evaluation_class_fold_rows.append(
                        {
                            **evaluation_metadata,
                            "outer_split": outer_score["outer_split"],
                            "outer_repeat": outer_score["outer_repeat"],
                            "outer_fold": outer_score["outer_fold"],
                            "class_label": (
                                class_label.item()
                                if isinstance(class_label, np.generic)
                                else class_label
                            ),
                            "precision": metrics["precision"],
                            "recall": metrics["recall"],
                            "f1_score": metrics["f1"],
                            "support": metrics["support"],
                            "selected": evaluation["selected"],
                        }
                    )

            classifier_class_rows = [
                row
                for row in self.evaluation_class_fold_rows
                if row["dataset"] == dataset_id
                and row["classifier_name"] == evaluation_metadata["classifier_name"]
            ]
            for class_label in target.classes:
                native_label = (
                    class_label.item()
                    if isinstance(class_label, np.generic)
                    else class_label
                )
                class_rows = [
                    row
                    for row in classifier_class_rows
                    if row["class_label"] == native_label
                ]
                if not class_rows:
                    continue
                self.evaluation_class_summary_rows.append(
                    {
                        **evaluation_metadata,
                        "class_label": native_label,
                        "precision_mean": float(
                            np.mean([row["precision"] for row in class_rows])
                        ),
                        "precision_std": float(
                            np.std([row["precision"] for row in class_rows], ddof=0)
                        ),
                        "recall_mean": float(
                            np.mean([row["recall"] for row in class_rows])
                        ),
                        "recall_std": float(
                            np.std([row["recall"] for row in class_rows], ddof=0)
                        ),
                        "f1_score_mean": float(
                            np.mean([row["f1_score"] for row in class_rows])
                        ),
                        "f1_score_std": float(
                            np.std([row["f1_score"] for row in class_rows], ddof=0)
                        ),
                        "support_mean": float(
                            np.mean([row["support"] for row in class_rows])
                        ),
                        "support_std": float(
                            np.std([row["support"] for row in class_rows], ddof=0)
                        ),
                        "n_outer_folds": len(class_rows),
                        "selected": evaluation["selected"],
                    }
                )
            for optimization_search in evaluation.get("optimization_searches", []):
                self.optimization_rows.append(
                    {
                        **evaluation_metadata,
                        "search_scope": "outer",
                        **optimization_search,
                        "selected": evaluation["selected"],
                    }
                )

        selected_evaluation = self._selected_evaluation(evaluations)
        final_optimization_search = selected_evaluation.get("final_optimization_search")
        if final_optimization_search is not None:
            selected_metadata = {
                **dataset_metadata,
                "classifier_name": _CLASSIFIER_NAMES[
                    selected_evaluation["classifier_key"]
                ],
            }
            self.optimization_rows.append(
                {
                    **selected_metadata,
                    "search_scope": "final",
                    "outer_split": None,
                    "outer_repeat": None,
                    "outer_fold": None,
                    **asdict(final_optimization_search),
                    "selected": None,
                }
            )

        (
            best_model,
            best_params,
            best_f1,
            f1_std,
            best_acc,
            acc_std,
            best_auc,
            auc_std,
        ) = selected_result
        params = self._clean_params(best_params)
        classifier_name = _CLASSIFIER_NAMES[selected_evaluation["classifier_key"]]
        metadata_lines = [
            f"Family: {dataset_metadata['family']}"
            if dataset_metadata["family"] is not None
            else None,
            f"Method: {dataset_metadata['method']}"
            if dataset_metadata["method"] is not None
            else None,
            f"Variant: {dataset_metadata['variant']}"
            if dataset_metadata["variant"] is not None
            else None,
            f"Level: {level}" if level is not None else None,
            f"Description: {dataset_metadata['description']}"
            if dataset_metadata["description"] is not None
            else None,
        ]
        self.final_results.append(
            "\n".join(
                [
                    f"Results for dataset {dataset_id}:",
                    *[line for line in metadata_lines if line is not None],
                    f"Classifier selected: {classifier_name}",
                    f"Best classifier parameters: {params}",
                    f"F1-macro (CV mean): {round(best_f1, 4)} +/- {round(f1_std, 4)}",
                    f"Accuracy (CV mean): {round(best_acc, 4)} +/- {round(acc_std, 4)}",
                    (
                        f"AUC-ROC (CV mean): {round(best_auc, 4)} +/- {round(auc_std, 4)}"
                        if best_auc is not None
                        else "AUC-ROC (CV mean): N/A"
                    ),
                    "------------------------------",
                ]
            )
        )
        self.csv_rows.append(
            {
                **dataset_metadata,
                "classifier_name": classifier_name,
                "parameters": str(params),
                "f1_macro": round(best_f1, 4),
                "f1_std": round(f1_std, 4),
                "accuracy": round(best_acc, 4),
                "acc_std": round(acc_std, 4),
                "auc_roc": round(best_auc, 4) if best_auc is not None else "N/A",
                "auc_std": round(auc_std, 4) if auc_std is not None else "N/A",
            }
        )

    def _run_regression_dataset(
        self, dataset_id: str, dataset: dict, dataset_metadata: dict
    ) -> None:
        if self.regression_model_trainer is None:
            raise RuntimeError("Regression trainer was not initialized.")
        selected_result, evaluations = (
            self.regression_model_trainer.evaluate_and_select_models(
                dataset["X"], dataset["y"]
            )
        )
        self.regression_evaluations_by_dataset[dataset_id] = evaluations

        for evaluation in evaluations:
            regressor_key = evaluation["regressor_key"]
            evaluation_metadata = {
                **dataset_metadata,
                "task": "regression",
                "regressor_key": regressor_key,
                "regressor_name": _REGRESSOR_NAMES[regressor_key],
            }
            self.regression_evaluation_rows.append(
                {
                    **evaluation_metadata,
                    "rmse": evaluation["rmse"],
                    "rmse_std": evaluation["rmse_std"],
                    "mae": evaluation["mae"],
                    "mae_std": evaluation["mae_std"],
                    "r2": evaluation["r2"],
                    "r2_std": evaluation["r2_std"],
                    "selected": evaluation["selected"],
                }
            )
            for outer_score in evaluation["outer_scores"]:
                self.regression_evaluation_fold_rows.append(
                    {
                        **evaluation_metadata,
                        "outer_split": outer_score["outer_split"],
                        "outer_repeat": outer_score["outer_repeat"],
                        "outer_fold": outer_score["outer_fold"],
                        "rmse": outer_score["rmse"],
                        "mae": outer_score["mae"],
                        "r2": outer_score["r2"],
                        "selected": evaluation["selected"],
                    }
                )
            for optimization_search in evaluation.get("optimization_searches", []):
                self.regression_optimization_rows.append(
                    {
                        **evaluation_metadata,
                        "search_scope": "outer",
                        **optimization_search,
                        "selected": evaluation["selected"],
                    }
                )

        selected_evaluation = self._selected_regression_evaluation(evaluations)
        selected_key = selected_evaluation["regressor_key"]
        selected_metadata = {
            **dataset_metadata,
            "task": "regression",
            "regressor_key": selected_key,
            "regressor_name": _REGRESSOR_NAMES[selected_key],
        }
        final_optimization_search = selected_evaluation.get("final_optimization_search")
        if final_optimization_search is not None:
            self.regression_optimization_rows.append(
                {
                    **selected_metadata,
                    "search_scope": "final",
                    "outer_split": None,
                    "outer_repeat": None,
                    "outer_fold": None,
                    **asdict(final_optimization_search),
                    "selected": True,
                }
            )

        (
            best_model,
            best_params,
            best_rmse,
            rmse_std,
            best_mae,
            mae_std,
            best_r2,
            r2_std,
        ) = selected_result
        params = self._clean_params(best_params)
        self.regression_csv_rows.append(
            {
                **selected_metadata,
                "parameters": params,
                "rmse": best_rmse,
                "rmse_std": rmse_std,
                "mae": best_mae,
                "mae_std": mae_std,
                "r2": best_r2,
                "r2_std": r2_std,
            }
        )

        metadata_lines = [
            f"Family: {dataset_metadata['family']}"
            if dataset_metadata["family"] is not None
            else None,
            f"Method: {dataset_metadata['method']}"
            if dataset_metadata["method"] is not None
            else None,
            f"Variant: {dataset_metadata['variant']}"
            if dataset_metadata["variant"] is not None
            else None,
            f"Level: {dataset_metadata['level']}"
            if dataset_metadata["level"] is not None
            else None,
            f"Description: {dataset_metadata['description']}"
            if dataset_metadata["description"] is not None
            else None,
        ]
        cv_config = self.config.CV_CONFIG
        self.regression_final_results.append(
            "\n".join(
                [
                    f"Regression results for dataset {dataset_id}:",
                    "Task: regression",
                    *[line for line in metadata_lines if line is not None],
                    f"Regressor selected: {_REGRESSOR_NAMES[selected_key]}",
                    f"Final model parameters: {params}",
                    f"RMSE (outer-CV mean): {best_rmse} +/- {rmse_std}",
                    f"MAE (outer-CV mean): {best_mae} +/- {mae_std}",
                    f"R-squared (outer-CV mean): {best_r2} +/- {r2_std}",
                    (
                        "Outer CV: RepeatedKFold "
                        f"(n_splits={cv_config['n_splits']}, "
                        f"n_repeats={cv_config['n_repeats']})"
                    ),
                    (
                        "Inner CV: shuffled KFold "
                        f"(n_splits={cv_config['inner_n_splits']})"
                    ),
                    f"Random state: {self.config.RANDOM_STATE}",
                    f"Optimization trial budget: {self.config.N_TRIALS}",
                    f"Smoke execution: {self.config.SMOKE}",
                    "------------------------------",
                ]
            )
        )

    def _write_classification_artifacts(self) -> None:
        final_report = "\n".join(self.final_results)
        self.result_manager.write_results(final_report)
        logger.info("Final report written to %s", self.config.RESULTS_FILE)

        output_path = Path(self.config.PATHS["OUTPUT"])
        csv_path = output_path / "results.csv"
        self.result_manager.write_csv(self.csv_rows, csv_path, smoke=self.config.SMOKE)
        logger.info("CSV file written to %s", csv_path)
        self.result_manager.write_evaluations_csv(
            self.evaluation_rows,
            output_path / "evaluations.csv",
            smoke=self.config.SMOKE,
        )
        self.result_manager.write_evaluation_folds_csv(
            self.evaluation_fold_rows,
            output_path / "evaluation_folds.csv",
            smoke=self.config.SMOKE,
        )
        self.result_manager.write_evaluation_class_folds_csv(
            self.evaluation_class_fold_rows,
            output_path / "evaluation_class_folds.csv",
            smoke=self.config.SMOKE,
        )
        self.result_manager.write_evaluation_class_summary_csv(
            self.evaluation_class_summary_rows,
            output_path / "evaluation_class_summary.csv",
            smoke=self.config.SMOKE,
        )
        self.result_manager.write_optimization_searches_csv(
            self.optimization_rows,
            output_path / "optimization_searches.csv",
            smoke=self.config.SMOKE,
        )
        self.result_manager.write_optimization_provenance_json(
            self._optimization_provenance(),
            output_path / "optimization_provenance.json",
        )
        self.result_manager.write_classification_metric_provenance_json(
            self._classification_metric_provenance(),
            output_path / "classification_metric_provenance.json",
        )
        self.result_manager.write_evaluation_figures(
            self.evaluation_fold_rows,
            smoke=self.config.SMOKE,
        )

    def _write_regression_artifacts(self) -> None:
        output_path = Path(self.config.PATHS["OUTPUT"])
        self.result_manager.write_regression_results(
            "\n".join(self.regression_final_results),
            output_path / "regression_results.txt",
        )
        self.result_manager.write_regression_results_csv(
            self.regression_csv_rows,
            output_path / "regression_results.csv",
            smoke=self.config.SMOKE,
        )
        self.result_manager.write_regression_evaluations_csv(
            self.regression_evaluation_rows,
            output_path / "regression_evaluations.csv",
            smoke=self.config.SMOKE,
        )
        self.result_manager.write_regression_evaluation_folds_csv(
            self.regression_evaluation_fold_rows,
            output_path / "regression_evaluation_folds.csv",
            smoke=self.config.SMOKE,
        )
        self.result_manager.write_regression_optimization_searches_csv(
            self.regression_optimization_rows,
            output_path / "regression_optimization_searches.csv",
            smoke=self.config.SMOKE,
        )
        self.result_manager.write_regression_optimization_provenance_json(
            self._regression_optimization_provenance(),
            output_path / "regression_optimization_provenance.json",
        )

    def _run_evaluation(self) -> None:
        if self.config.SMOKE:
            configured_tasks = {
                spec.get("task", "classification")
                for spec in self.config.DATASETS.values()
            }
            logger.info(
                "SMOKE TEST - reduced search and cross-validation settings. "
                "Results are not suitable for final model selection."
            )
            print(
                _SMOKE_WARNING
                if configured_tasks == {"classification"}
                else _TASK_SMOKE_WARNING
            )

        self._warn_for_low_optimization_budget()
        datasets = load_datasets(self.config)
        classification_executed = False
        regression_executed = False

        for dataset_id, dataset in datasets.items():
            metadata = dataset.get("metadata", {})
            dataset_metadata = self._dataset_metadata(dataset_id, metadata)
            task = dataset.get(
                "task",
                self.config.DATASETS[dataset_id].get("task", "classification"),
            )
            logger.info("Training with %s dataset %s.", task, dataset_id)
            if task == "classification":
                self._run_classification_dataset(dataset_id, dataset, dataset_metadata)
                classification_executed = True
            elif task == "regression":
                self._run_regression_dataset(dataset_id, dataset, dataset_metadata)
                regression_executed = True
            else:
                raise ValueError(
                    f"Dataset {dataset_id!r} has unsupported task {task!r}."
                )

        if classification_executed:
            self._write_classification_artifacts()
        if regression_executed:
            self._write_regression_artifacts()
