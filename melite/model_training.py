# SPDX-License-Identifier: LGPL-3.0-or-later
"""Model optimization, training, and cross-validation for MELITE.

This module implements the multi-classifier evaluation core. It defines an
abstract base class :class:`ModelTrainer` and the concrete implementation
:class:`MultiModelTrainer`, which evaluates SVC, Random Forest, and XGBoost
classifiers with nested cross-validation and supports opt-in stacking.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from melite.optimization import OptunaSearchClassifier, optimize_and_refit
from melite.search_spaces import get_search_space

__all__ = ["MultiModelTrainer"]


class ModelTrainer(ABC):
    """Abstract base class for MELITE classifier trainers.

    Parameters
    ----------
    config : melite.config.Config
        MELITE configuration object providing evaluation and optimization settings.
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

    Uses nested cross-validation for tunable classifiers: an inner Optuna
    search selects hyperparameters independently within each outer
    repeated-stratified fold, and the outer folds provide F1-macro, Accuracy,
    and AUC-ROC estimates. Non-tunable classifiers are evaluated directly.

    Parameters
    ----------
    config : melite.config.Config
        MELITE configuration object providing classifier, CV, and random-seed
        settings together with the optimization trial budget.

    Notes
    -----
    Model instances are created lazily via ``model_builders`` — a dict of
    zero-argument callables — so each fit starts from a fresh,
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

    def _build_cv_strategy(self):
        """Return the outer CV strategy (backward-compatible internal alias)."""
        return self._build_outer_cv()

    def cross_validate_model(self, model, X_train, y_train):
        """Evaluate an estimator with the outer repeated-stratified CV.

        Parameters
        ----------
        model : estimator
            Scikit-learn compatible estimator.
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
        tunable = isinstance(model, OptunaSearchClassifier)
        cross_validate_kwargs = {
            "scoring": scoring,
            "cv": cv,
            "n_jobs": 1,
            "return_train_score": False,
        }
        if tunable:
            cross_validate_kwargs.update(
                return_estimator=True,
                error_score="raise",
            )
        scores = cross_validate(model, X_train, y_train, **cross_validate_kwargs)

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
        optimization_searches = []
        if tunable:
            fitted_wrappers = scores.pop("estimator")
            optimization_searches = [
                {
                    "outer_split": outer_split,
                    "outer_repeat": outer_split // n_splits,
                    "outer_fold": outer_split % n_splits,
                    **asdict(wrapper.optimization_result_),
                }
                for outer_split, wrapper in enumerate(fitted_wrappers)
            ]

        return {
            "f1_macro": f1_vals.mean(),
            "f1_std": f1_vals.std(),
            "accuracy": acc_vals.mean(),
            "acc_std": acc_vals.std(),
            "auc_roc": auc_mean,
            "auc_std": auc_std,
            "outer_scores": outer_scores,
            "optimization_searches": optimization_searches,
        }

    def _evaluate_classifier(self, classifier_name, X_train, y_train):
        model = self.model_builders[classifier_name]()
        search_space = get_search_space(classifier_name)
        evaluation_estimator = (
            model
            if search_space is None
            else OptunaSearchClassifier(
                estimator=model,
                search_space=search_space,
                inner_n_splits=self.config.CV_CONFIG["inner_n_splits"],
                n_trials=self.config.N_TRIALS,
                random_state=self.config.RANDOM_STATE,
            )
        )
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

        Each tunable classifier is evaluated with nested cross-validation;
        non-tunable classifiers are evaluated directly. The classifier with the
        highest outer mean F1-macro is selected, then a fresh winning estimator
        is fitted on all data.

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
        winning_space = get_search_space(best["classifier_key"])
        if winning_space is None:
            winning_model.fit(X_train, y_train)
            params = {}
            best["final_optimization_search"] = None
        else:
            winning_model, optimization_result = optimize_and_refit(
                winning_model,
                winning_space,
                X_train,
                y_train,
                inner_n_splits=self.config.CV_CONFIG["inner_n_splits"],
                n_trials=self.config.N_TRIALS,
                random_state=self.config.RANDOM_STATE,
            )
            params = optimization_result.best_params
            best["final_optimization_search"] = optimization_result

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
