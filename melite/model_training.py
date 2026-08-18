# SPDX-License-Identifier: LGPL-3.0-or-later
"""Model training, grid search and cross-validation for MELITE.

This module implements the multi-model benchmarking core. It defines an
abstract base class :class:`ModelTrainer` and the concrete implementation
:class:`MultiModelTrainer`, which evaluates SVC, Random Forest, and XGBoost
classifiers with nested cross-validation and supports opt-in stacking.
"""

from abc import ABC, abstractmethod

import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

__all__ = ["MultiModelTrainer"]


class ModelTrainer(ABC):
    """Abstract base class for MELITE model trainers.

    Parameters
    ----------
    config : melite.config.Config
        MELITE configuration object providing CV settings and hyperparameter
        grids.
    """

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def train_and_select_best_model(self, X_train, y_train, reduction_type, level):
        """Train all models and return the best configuration.

        Parameters
        ----------
        X_train : numpy.ndarray
            Feature matrix of shape ``(n_samples, n_features)``.
        y_train : numpy.ndarray
            Label vector of shape ``(n_samples,)``.
        reduction_type : str
            Dimensionality reduction method (e.g. ``"PCA"``).
        level : int
            Variance retention level (e.g. ``85``).

        Returns
        -------
        tuple
            Eight-element tuple: ``(model, params, f1, f1_std, acc, acc_std,
            auc, auc_std)``.
        """


class MultiModelTrainer(ModelTrainer):
    """Train and select the best model across SVC, Random Forest and XGBoost.

    Uses nested cross-validation for tunable model families: an inner
    stratified grid search selects hyperparameters independently within each
    outer repeated-stratified fold, and the outer folds provide F1-macro,
    Accuracy, and AUC-ROC estimates. Stacking is evaluated directly.

    Parameters
    ----------
    config : melite.config.Config
        MELITE configuration object. Must expose :meth:`~melite.config.Config.get_cv_config`
        and :attr:`~melite.config.Config.PARAM_GRID`.

    Notes
    -----
    Model instances are created lazily via ``model_builders`` — a dict of
    zero-argument callables — so each grid search starts from a fresh,
    unfitted estimator.
    """

    def __init__(self, config):
        super().__init__(config)

        rs = getattr(self.config, "RANDOM_STATE", 42)

        self.model_builders = {
            "svc": lambda: SklearnPipeline([
                ("scaler", StandardScaler()),
                ("svc", SVC(probability=True, random_state=rs)),
            ]),
            "rf": lambda: RandomForestClassifier(random_state=rs, n_jobs=-1),
            "xgb": lambda: XGBClassifier(eval_metric="logloss", random_state=rs, n_jobs=-1),
            "stack": lambda: self._build_stacking_classifier(rs),
        }
        self.active_models = self._validate_active_models()

    def _build_stacking_classifier(self, random_state):
        cv_cfg = self.config.get_cv_config()
        # StackingClassifier uses cross_val_predict internally, which requires
        # one partition of the data rather than repeated test assignments.
        stacking_cv = StratifiedKFold(
            n_splits=cv_cfg["inner_n_splits"],
            shuffle=True,
            random_state=cv_cfg["random_state"],
        )
        return StackingClassifier(
            estimators=[
                ("svc", SklearnPipeline([
                    ("scaler", StandardScaler()),
                    ("svc", SVC(probability=True, random_state=random_state)),
                ])),
                ("rf", RandomForestClassifier(random_state=random_state, n_jobs=-1)),
                (
                    "xgb",
                    XGBClassifier(
                        eval_metric="logloss",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ],
            final_estimator=LogisticRegression(
                random_state=random_state,
                max_iter=1000,
            ),
            cv=stacking_cv,
            stack_method="predict_proba",
            passthrough=False,
            n_jobs=-1,
        )

    def _validate_active_models(self):
        active_models = getattr(self.config, "ACTIVE_MODELS", list(self.model_builders))
        if not active_models:
            raise ValueError("ACTIVE_MODELS must contain at least one model key.")

        unknown = [model for model in active_models if model not in self.model_builders]
        if unknown:
            unknown_models = ", ".join(unknown)
            valid_models = ", ".join(self.model_builders)
            raise ValueError(
                f"Unknown active model(s): {unknown_models}. "
                f"Valid model keys are: {valid_models}."
            )

        return list(active_models)

    def _build_outer_cv(self):
        cv_cfg = self.config.get_cv_config()
        return RepeatedStratifiedKFold(
            n_splits=cv_cfg["n_splits"],
            n_repeats=cv_cfg["n_repeats"],
            random_state=cv_cfg["random_state"],
        )

    def _build_inner_cv(self):
        cv_cfg = self.config.get_cv_config()
        return StratifiedKFold(
            n_splits=cv_cfg["inner_n_splits"],
            shuffle=True,
            random_state=cv_cfg["random_state"],
        )

    def _build_cv_strategy(self):
        """Return the outer CV strategy (backward-compatible internal alias)."""
        return self._build_outer_cv()

    def _filter_param_grid(self, model_name):
        return [
            {k: v for k, v in g.items() if k != "model"}
            for g in self.config.PARAM_GRID
            if g["model"][0] == model_name
        ]

    def _build_grid_search(self, model, param_grid):
        return GridSearchCV(
            model,
            param_grid,
            scoring="f1_macro",
            cv=self._build_inner_cv(),
            n_jobs=-1,
        )

    def perform_grid_search(self, model, X_train, y_train, param_grid):
        """Run grid search and return the best estimator and parameters.

        Parameters
        ----------
        model : estimator
            Unfitted scikit-learn compatible estimator.
        X_train : numpy.ndarray
            Feature matrix of shape ``(n_samples, n_features)``.
        y_train : numpy.ndarray
            Label vector of shape ``(n_samples,)``.
        param_grid : list of dict
            Hyperparameter grid as expected by
            :class:`~sklearn.model_selection.GridSearchCV`.

        Returns
        -------
        best_estimator : estimator
            Refitted estimator with the best hyperparameters.
        best_params : dict
            Best hyperparameter combination found.
        """
        grid = self._build_grid_search(model, param_grid)
        grid.fit(X_train, y_train)
        return grid.best_estimator_, grid.best_params_

    def cross_validate_model(self, model, X_train, y_train):
        """Evaluate an estimator with the outer repeated-stratified CV.

        Parameters
        ----------
        model : estimator
            Scikit-learn compatible estimator. For tunable families this is
            an unfitted :class:`~sklearn.model_selection.GridSearchCV` object.
        X_train : numpy.ndarray
            Feature matrix of shape ``(n_samples, n_features)``.
        y_train : numpy.ndarray
            Label vector of shape ``(n_samples,)``.

        Returns
        -------
        f1_mean : float
        f1_std : float
        acc_mean : float
        acc_std : float
        auc_mean : float or None
        auc_std : float or None
        """
        cv = self._build_outer_cv()
        scoring = {"f1": "f1_macro", "acc": "accuracy", "auc": "roc_auc"}
        scores = cross_validate(
            model, X_train, y_train,
            scoring=scoring, cv=cv, n_jobs=1, return_train_score=False,
        )

        f1_mean, f1_std = scores["test_f1"].mean(), scores["test_f1"].std()
        acc_mean, acc_std = scores["test_acc"].mean(), scores["test_acc"].std()
        auc_vals = scores.get("test_auc")
        auc_mean, auc_std = (
            (auc_vals.mean(), auc_vals.std()) if auc_vals is not None else (None, None)
        )

        return f1_mean, f1_std, acc_mean, acc_std, auc_mean, auc_std

    def _evaluate_model_family(self, model_name, X_train, y_train):
        model = self.model_builders[model_name]()
        if model_name == "stack":
            evaluation_estimator = model
        else:
            param_grid = self._filter_param_grid(model_name)
            evaluation_estimator = self._build_grid_search(model, param_grid)
        return self.cross_validate_model(evaluation_estimator, X_train, y_train)

    def train_and_select_best_model(self, X_train, y_train, reduction_type, level):
        """Train all active models and return the best configuration.

        Each tunable family is evaluated with nested cross-validation; stacking
        is evaluated directly. The family with the highest outer mean F1-macro
        is selected, then a fresh winning estimator is fitted on all data.

        Parameters
        ----------
        X_train : numpy.ndarray
            Feature matrix of shape ``(n_samples, n_features)``.
        y_train : numpy.ndarray
            Label vector of shape ``(n_samples,)``.
        reduction_type : str
            Dimensionality reduction method (e.g. ``"PCA"``). Used for logging.
        level : int
            Variance retention level (e.g. ``85``). Used for logging.

        Returns
        -------
        model : estimator
            Best fitted estimator.
        params : dict
            Best hyperparameter combination.
        f1 : float
            Mean F1-macro across CV folds.
        f1_std : float
            Standard deviation of F1-macro across CV folds.
        acc : float
            Mean accuracy across CV folds.
        acc_std : float
            Standard deviation of accuracy across CV folds.
        auc : float or None
            Mean AUC-ROC across CV folds, or ``None`` if not available.
        auc_std : float or None
            Standard deviation of AUC-ROC across CV folds, or ``None``.
        """
        best = {
            "model_name": None,
            "f1": -1, "f1_std": 0,
            "acc": 0, "acc_std": 0,
            "auc": None, "auc_std": None,
        }

        for model_name in self.active_models:
            f1_mean, f1_std, acc_mean, acc_std, auc_mean, auc_std = (
                self._evaluate_model_family(model_name, X_train, y_train)
            )

            if f1_mean > best["f1"]:
                best.update({
                    "model_name": model_name,
                    "f1": f1_mean, "f1_std": f1_std,
                    "acc": acc_mean, "acc_std": acc_std,
                    "auc": auc_mean, "auc_std": auc_std,
                })

        winning_model = self.model_builders[best["model_name"]]()
        if best["model_name"] == "stack":
            winning_model.fit(X_train, y_train)
            params = {}
        else:
            param_grid = self._filter_param_grid(best["model_name"])
            winning_model, params = self.perform_grid_search(
                winning_model, X_train, y_train, param_grid
            )

        return (
            winning_model, params,
            best["f1"], best["f1_std"],
            best["acc"], best["acc_std"],
            best["auc"], best["auc_std"],
        )
