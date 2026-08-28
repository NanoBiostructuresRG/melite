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
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.svm import SVC

from melite.config import Config
from melite.load_dataset import load_datasets
from melite.model_training import MultiModelTrainer
from melite.optimization import get_optimization_backend_info, optuna_logging_scope
from melite.optimization_policy import OPTIMIZATION_POLICY
from melite.result_manager import ResultManager
from melite.search_spaces import get_search_space
from melite.version import __version__

logger = logging.getLogger(__name__)

_SMOKE_WARNING = (
    "\n[SMOKE TEST] Using reduced search and cross-validation settings. "
    "Results are not suitable for final classifier selection.\n"
)

_CLASSIFIER_NAMES = {
    "svc": "SVC",
    "rf": "RandomForestClassifier",
    "xgb": "XGBClassifier",
    "stack": "StackingClassifier",
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
        self.pipeline = Pipeline(self.config)
        self.result_manager = ResultManager(self.config.RESULTS_FILE)
        self.final_results: list[str] = []
        self.csv_rows: list[dict[str, Any]] = []
        self.evaluations_by_dataset: dict[str, list[dict[str, Any]]] = {}
        self.evaluation_rows: list[dict[str, Any]] = []
        self.evaluation_fold_rows: list[dict[str, Any]] = []
        self.optimization_rows: list[dict[str, Any]] = []

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
    def _classifier_name(model):
        if isinstance(model, SklearnPipeline) and isinstance(model.steps[-1][1], SVC):
            return "SVC"
        return model.__class__.__name__

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

    def run(self) -> None:
        """Execute the evaluation pipeline for all configured datasets.

        Iterates over the normalized ``config.DATASETS`` registry, evaluates all
        active classifiers for each dataset, selects the best classifier, and writes
        the result and evaluation evidence outputs.

        Notes
        -----
        When smoke mode is active, a visible banner is printed to stdout
        regardless of the logging level, to ensure the user is aware that
        results are not suitable for final classifier selection.
        """
        # The CLI configures MELITE logging before entering Main.run();
        # programmatic callers may configure the logger directly.
        verbose = logger.isEnabledFor(logging.INFO)
        with optuna_logging_scope(verbose=verbose):
            self._run_evaluation()

    def _warn_for_low_optimization_budget(self) -> None:
        if (
            not self.config.SMOKE
            and self.config.N_TRIALS <= OPTIMIZATION_POLICY.n_startup_trials
            and any(
                get_search_space(classifier_key) is not None
                for classifier_key in self.config.ACTIVE_CLASSIFIERS
            )
        ):
            logger.warning(
                "The requested optimization budget remains within TPE startup "
                "sampling and does not reach model-based TPE."
            )

    def _run_evaluation(self) -> None:
        if self.config.SMOKE:
            logger.info(
                "SMOKE TEST - reduced search and cross-validation settings. "
                "Results are not suitable for final classifier selection."
            )
            print(_SMOKE_WARNING)

        self._warn_for_low_optimization_budget()

        datasets = load_datasets(self.config)

        for dataset_id, dataset in datasets.items():
            X_train = dataset["X"]
            y_train = dataset["y"]
            metadata = dataset.get("metadata", {})
            family = metadata.get("family")
            method = metadata.get("method")
            variant = metadata.get("variant")
            level = metadata.get("level")
            description = metadata.get("description")
            reduction_type = self._legacy_reduction_type(metadata)
            dataset_metadata = {
                "dataset": dataset_id,
                "family": family,
                "method": method,
                "variant": variant,
                "level": level,
                "description": description,
                "reduction_type": reduction_type,
            }

            logger.info("Training with dataset %s.", dataset_id)

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
            final_optimization_search = selected_evaluation.get(
                "final_optimization_search"
            )
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
            classifier_name = self._classifier_name(best_model)

            metadata_lines = [
                f"Family: {family}" if family is not None else None,
                f"Method: {method}" if method is not None else None,
                f"Variant: {variant}" if variant is not None else None,
                f"Level: {level}" if level is not None else None,
                f"Description: {description}" if description is not None else None,
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
                    "dataset": dataset_id,
                    "family": family,
                    "method": method,
                    "variant": variant,
                    "level": level,
                    "description": description,
                    "reduction_type": reduction_type,
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

        final_report = "\n".join(self.final_results)
        self.result_manager.write_results(final_report)
        logger.info("Final report written to %s", self.config.RESULTS_FILE)

        csv_path = Path(self.config.PATHS["OUTPUT"]) / "results.csv"
        self.result_manager.write_csv(self.csv_rows, csv_path, smoke=self.config.SMOKE)
        logger.info("CSV file written to %s", csv_path)

        evaluations_path = Path(self.config.PATHS["OUTPUT"]) / "evaluations.csv"
        self.result_manager.write_evaluations_csv(
            self.evaluation_rows, evaluations_path, smoke=self.config.SMOKE
        )

        folds_path = Path(self.config.PATHS["OUTPUT"]) / "evaluation_folds.csv"
        self.result_manager.write_evaluation_folds_csv(
            self.evaluation_fold_rows, folds_path, smoke=self.config.SMOKE
        )

        optimization_path = (
            Path(self.config.PATHS["OUTPUT"]) / "optimization_searches.csv"
        )
        self.result_manager.write_optimization_searches_csv(
            self.optimization_rows, optimization_path, smoke=self.config.SMOKE
        )

        provenance_path = (
            Path(self.config.PATHS["OUTPUT"]) / "optimization_provenance.json"
        )
        self.result_manager.write_optimization_provenance_json(
            self._optimization_provenance(), provenance_path
        )

        self.result_manager.write_evaluation_figures(
            self.evaluation_fold_rows,
            smoke=self.config.SMOKE,
        )
