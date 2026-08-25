# SPDX-License-Identifier: LGPL-3.0-or-later
"""Model training, grid search and cross-validation for MELITE.

This module implements the multi-classifier evaluation core. It defines an
abstract base class :class:`ModelTrainer` and the concrete implementation
:class:`MultiModelTrainer`, which evaluates SVC, Random Forest, and XGBoost
classifiers with nested cross-validation and supports opt-in stacking.
"""

from abc import ABC, abstractmethod

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
    """Abstract base class for MELITE classifier trainers.

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
        """Train all classifiers and return the best configuration.

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
    """Train and select the best classifier across SVC, Random Forest and XGBoost.

    Uses nested cross-validation for tunable classifiers: an inner
    stratified grid search selects hyperparameters independently within each
    outer repeated-stratified fold, and the outer folds provide F1-macro,
    Accuracy, and AUC-ROC estimates. Stacking is evaluated directly.

    Parameters
    ----------
    config : melite.config.Config
        MELITE configuration object providing classifier, CV, and random-seed
        settings together with the internal hyperparameter search grid.

    Notes
    -----
    Model instances are created lazily via ``model_builders`` — a dict of
    zero-argument callables — so each grid search starts from a fresh,
    unfitted estimator.
    """

    def __init__(self, config):
        super().__init__(config)

        rs = self.config.RANDOM_STATE

        self.model_builders = {
            "svc": lambda: SklearnPipeline(
                [
                    ("scaler", StandardScaler()),
                    ("svc", SVC(probability=False, random_state=rs)),
                ]
            ),
            "rf": lambda: RandomForestClassifier(random_state=rs, n_jobs=1),
            "xgb": lambda: XGBClassifier(
                eval_metric="logloss", random_state=rs, n_jobs=1
            ),
            "stack": lambda: self._build_stacking_classifier(rs),
        }
        self.active_classifiers = self._validate_active_classifiers()

    def _build_stacking_classifier(self, random_state):
        cv_cfg = self.config.CV_CONFIG
        # StackingClassifier uses cross_val_predict internally, which requires
        # one partition of the data rather than repeated test assignments.
        stacking_cv = StratifiedKFold(
            n_splits=cv_cfg["inner_n_splits"],
            shuffle=True,
            random_state=random_state,
        )
        return StackingClassifier(
            estimators=[
                (
                    "svc",
                    SklearnPipeline(
                        [
                            ("scaler", StandardScaler()),
                            ("svc", SVC(probability=True, random_state=random_state)),
                        ]
                    ),
                ),
                ("rf", RandomForestClassifier(random_state=random_state, n_jobs=1)),
                (
                    "xgb",
                    XGBClassifier(
                        eval_metric="logloss",
                        random_state=random_state,
                        n_jobs=1,
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

    def _validate_active_classifiers(self):
        active_classifiers = self.config.ACTIVE_CLASSIFIERS
        if not active_classifiers:
            raise ValueError(
                "ACTIVE_CLASSIFIERS must contain at least one classifier key."
            )

        unknown = [
            classifier
            for classifier in active_classifiers
            if classifier not in self.model_builders
        ]
        if unknown:
            unknown_classifiers = ", ".join(unknown)
            valid_classifiers = ", ".join(self.model_builders)
            raise ValueError(
                f"Unknown active classifier(s): {unknown_classifiers}. "
                f"Valid classifier keys are: {valid_classifiers}."
            )

        return list(active_classifiers)

    def _build_outer_cv(self):
        cv_cfg = self.config.CV_CONFIG
        return RepeatedStratifiedKFold(
            n_splits=cv_cfg["n_splits"],
            n_repeats=cv_cfg["n_repeats"],
            random_state=self.config.RANDOM_STATE,
        )

    def _build_inner_cv(self):
        cv_cfg = self.config.CV_CONFIG
        return StratifiedKFold(
            n_splits=cv_cfg["inner_n_splits"],
            shuffle=True,
            random_state=self.config.RANDOM_STATE,
        )

    def _build_cv_strategy(self):
        """Return the outer CV strategy (backward-compatible internal alias)."""
        return self._build_outer_cv()

    def _filter_param_grid(self, classifier_name):
        return [
            {k: v for k, v in g.items() if k != "model"}
            for g in self.config._param_grid
            if g["model"][0] == classifier_name
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
        evaluation = self._cross_validate_model_with_scores(model, X_train, y_train)
        return (
            evaluation["f1_macro"],
            evaluation["f1_std"],
            evaluation["accuracy"],
            evaluation["acc_std"],
            evaluation["auc_roc"],
            evaluation["auc_std"],
        )

    def _cross_validate_model_with_scores(self, model, X_train, y_train):
        cv_cfg = self.config.CV_CONFIG
        cv = self._build_outer_cv()
        scoring = {"f1": "f1_macro", "acc": "accuracy", "auc": "roc_auc"}
        scores = cross_validate(
            model,
            X_train,
            y_train,
            scoring=scoring,
            cv=cv,
            n_jobs=1,
            return_train_score=False,
        )

        f1_vals = scores["test_f1"]
        acc_vals = scores["test_acc"]
        auc_vals = scores.get("test_auc")
        auc_mean, auc_std = (
            (auc_vals.mean(), auc_vals.std()) if auc_vals is not None else (None, None)
        )
        n_splits = cv_cfg["n_splits"]
        outer_scores = [
            {
                "outer_split": outer_split,
                "outer_repeat": outer_split // n_splits,
                "outer_fold": outer_split % n_splits,
                "f1_macro": f1_value,
                "accuracy": acc_vals[outer_split],
                "auc_roc": (auc_vals[outer_split] if auc_vals is not None else None),
            }
            for outer_split, f1_value in enumerate(f1_vals)
        ]

        return {
            "f1_macro": f1_vals.mean(),
            "f1_std": f1_vals.std(),
            "accuracy": acc_vals.mean(),
            "acc_std": acc_vals.std(),
            "auc_roc": auc_mean,
            "auc_std": auc_std,
            "outer_scores": outer_scores,
        }

    def _evaluate_classifier(self, classifier_name, X_train, y_train):
        model = self.model_builders[classifier_name]()
        if classifier_name == "stack":
            evaluation_estimator = model
        else:
            param_grid = self._filter_param_grid(classifier_name)
            evaluation_estimator = self._build_grid_search(model, param_grid)
        evaluation = self._cross_validate_model_with_scores(
            evaluation_estimator, X_train, y_train
        )
        return {
            "classifier_key": classifier_name,
            **evaluation,
            "selected": False,
        }

    def evaluate_and_select_models(self, X_train, y_train, reduction_type, level):
        """Evaluate every active classifier and return the selected model and evidence.

        Each tunable classifier is evaluated with nested cross-validation; stacking
        is evaluated directly. The classifier with the highest outer mean F1-macro
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
        selected_result : tuple
            Existing eight-element selected-model result.
        evaluations : list of dict
            Outer-CV aggregate and per-split evidence for every active classifier.
        """
        evaluations = [
            self._evaluate_classifier(classifier_name, X_train, y_train)
            for classifier_name in self.active_classifiers
        ]
        best = evaluations[0]
        for evaluation in evaluations[1:]:
            if evaluation["f1_macro"] > best["f1_macro"]:
                best = evaluation
        best["selected"] = True

        winning_model = self.model_builders[best["classifier_key"]]()
        if best["classifier_key"] == "stack":
            winning_model.fit(X_train, y_train)
            params = {}
        else:
            param_grid = self._filter_param_grid(best["classifier_key"])
            winning_model, params = self.perform_grid_search(
                winning_model, X_train, y_train, param_grid
            )

        selected_result = (
            winning_model,
            params,
            best["f1_macro"],
            best["f1_std"],
            best["accuracy"],
            best["acc_std"],
            best["auc_roc"],
            best["auc_std"],
        )
        return selected_result, evaluations

    def train_and_select_best_model(self, X_train, y_train, reduction_type, level):
        """Train all active classifiers and return the best configuration."""
        selected_result, _ = self.evaluate_and_select_models(
            X_train, y_train, reduction_type, level
        )
        return selected_result
