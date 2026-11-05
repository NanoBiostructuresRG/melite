# SPDX-License-Identifier: LGPL-3.0-or-later
"""Model optimization, training, and cross-validation for MELITE.

This module implements the multi-classifier evaluation core. It defines an
abstract base class :class:`ModelTrainer` and the concrete implementation
:class:`MultiModelTrainer`, which evaluates the configured subset of MELITE's
supported classifiers with nested cross-validation.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from melite.classification_metrics import (
    analyze_classification_target,
    classification_scoring,
)
from melite.optimization import OptunaSearchClassifier, optimize_and_refit
from melite.probability_calibration import ProbabilityCalibratedClassifier
from melite.search_spaces import get_search_space
from melite.xgboost_adapter import LabelIdentityXGBClassifier
from xgboost import XGBClassifier

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
    """Train and select the best classifier from the configured supported subset.

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
        catboost_config = dict(getattr(self.config, "CATBOOST_CONFIG", {}))

        def build_catboost(categorical_features=()):
            categorical_config = (
                {"cat_features": tuple(categorical_features)}
                if categorical_features
                else {}
            )
            return CatBoostClassifier(
                random_seed=rs,
                thread_count=1,
                verbose=False,
                allow_writing_files=False,
                task_type="CPU",
                **catboost_config,
                **categorical_config,
            )

        self.model_builders = {
            "svc": lambda: SklearnPipeline(
                [
                    ("scaler", StandardScaler()),
                    ("svc", SVC(random_state=rs)),
                ]
            ),
            "lr": lambda: SklearnPipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "lr",
                        LogisticRegression(
                            solver="lbfgs",
                            max_iter=1000,
                            random_state=rs,
                        ),
                    ),
                ]
            ),
            "gnb": lambda: GaussianNB(),
            "knn": lambda: SklearnPipeline(
                [
                    ("scaler", StandardScaler()),
                    ("knn", KNeighborsClassifier(metric="minkowski")),
                ]
            ),
            "dt": lambda: DecisionTreeClassifier(random_state=rs),
            "rf": lambda: RandomForestClassifier(random_state=rs, n_jobs=1),
            "ada": lambda: AdaBoostClassifier(random_state=rs),
            "xgb": lambda: LabelIdentityXGBClassifier(
                eval_metric="logloss", random_state=rs, n_jobs=1
            ),
            "lgbm": lambda: LGBMClassifier(
                random_state=rs,
                n_jobs=1,
                verbosity=-1,
            ),
            "catboost": build_catboost,
            "stack": lambda: self._build_stacking_classifier(rs),
        }
        self.active_classifiers = self._validate_active_classifiers()

    @staticmethod
    def _categorical_feature_names(X) -> list[str]:
        if not isinstance(X, pd.DataFrame):
            return []
        return list(X.attrs.get("categorical_features", []))

    @staticmethod
    def _select_rows(X, indices):
        if isinstance(X, pd.DataFrame):
            return X.iloc[indices]
        return np.asarray(X)[indices]

    def _build_model_for_target(self, classifier_name, y, X=None):
        """Build one classifier with target-specific probability support."""
        if classifier_name == "catboost":
            categorical_features = self._categorical_feature_names(X)
            model = self.model_builders[classifier_name](categorical_features)
        else:
            model = self.model_builders[classifier_name]()
        target = analyze_classification_target(y)
        if classifier_name != "svc" or not target.is_multiclass:
            return model

        return SklearnPipeline(
            [
                ("scaler", model.named_steps["scaler"]),
                (
                    "svc",
                    ProbabilityCalibratedClassifier(
                        estimator=model.named_steps["svc"],
                        max_calibration_cv=5,
                    ),
                ),
            ]
        )

    def _validate_class_distribution(self, y, classifier_name):
        """Reject class counts incompatible with configured nested folds."""
        target = analyze_classification_target(y)
        outer_splits = self.config.CV_CONFIG["n_splits"]
        inner_splits = self.config.CV_CONFIG["inner_n_splits"]

        for class_label, count in zip(target.classes, target.counts):
            if count < outer_splits:
                raise ValueError(
                    "Classification class distribution is incompatible with outer "
                    f"cross-validation: class {class_label!r} has {count} samples, "
                    f"but n_splits={outer_splits} requires at least "
                    f"{outer_splits}."
                )

            largest_outer_validation = (count + outer_splits - 1) // outer_splits
            smallest_outer_training = count - largest_outer_validation
            if smallest_outer_training < inner_splits:
                raise ValueError(
                    "Classification class distribution is incompatible with inner "
                    f"cross-validation: class {class_label!r} can have only "
                    f"{smallest_outer_training} samples in an outer training fold, "
                    f"but inner_n_splits={inner_splits} requires at least "
                    f"{inner_splits}."
                )

            if target.is_multiclass and classifier_name in {"svc", "stack"}:
                largest_inner_validation = (
                    smallest_outer_training + inner_splits - 1
                ) // inner_splits
                smallest_calibration_training = (
                    smallest_outer_training - largest_inner_validation
                )
                if smallest_calibration_training < 2:
                    raise ValueError(
                        "Classification class distribution is incompatible with "
                        "fold-local probability calibration: class "
                        f"{class_label!r} can have only "
                        f"{smallest_calibration_training} samples in an inner "
                        "training partition; at least 2 are required."
                    )
        return target

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
                            (
                                "svc",
                                ProbabilityCalibratedClassifier(
                                    estimator=SVC(random_state=random_state),
                                    max_calibration_cv=5,
                                ),
                            ),
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
        target = analyze_classification_target(y_train)
        scoring = classification_scoring(target)
        tunable = isinstance(model, OptunaSearchClassifier)
        outer_splits = list(cv.split(X_train, y_train))
        cross_validate_kwargs = {
            "scoring": scoring,
            "cv": outer_splits,
            "n_jobs": 1,
            "return_train_score": False,
            "return_estimator": True,
            "error_score": "raise",
        }
        scores = cross_validate(model, X_train, y_train, **cross_validate_kwargs)

        f1_vals = scores["test_f1"]
        acc_vals = scores["test_acc"]
        auc_vals = scores["test_auc"]
        fitted_estimators = scores.pop("estimator")
        for metric_name, values in (
            ("F1-macro", f1_vals),
            ("Accuracy", acc_vals),
            ("ROC-AUC", auc_vals),
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"Outer cross-validation produced non-finite {metric_name} values."
                )
        auc_mean, auc_std = auc_vals.mean(), auc_vals.std()
        n_splits = cv_cfg["n_splits"]
        outer_scores = []
        y_array = np.asarray(y_train)
        for outer_split, ((_, test_indices), fitted_estimator) in enumerate(
            zip(outer_splits, fitted_estimators)
        ):
            y_test = y_array[test_indices]
            X_test = self._select_rows(X_train, test_indices)
            y_pred = fitted_estimator.predict(X_test)
            precision, recall, per_class_f1, support = precision_recall_fscore_support(
                y_test,
                y_pred,
                labels=list(target.classes),
                zero_division=0,
            )
            if int(np.sum(support)) != len(test_indices):
                raise RuntimeError(
                    "Per-class outer-fold supports do not match the held-out "
                    "observation count."
                )
            per_class_metrics = {
                class_label: {
                    "precision": float(precision[class_index]),
                    "recall": float(recall[class_index]),
                    "f1": float(per_class_f1[class_index]),
                    "support": int(support[class_index]),
                }
                for class_index, class_label in enumerate(target.classes)
            }
            outer_scores.append(
                {
                    "outer_split": outer_split,
                    "outer_repeat": outer_split // n_splits,
                    "outer_fold": outer_split % n_splits,
                    "f1_macro": f1_vals[outer_split],
                    "accuracy": acc_vals[outer_split],
                    "auc_roc": auc_vals[outer_split],
                    "per_class_metrics": per_class_metrics,
                }
            )
        optimization_searches = []
        if tunable:
            optimization_searches = [
                {
                    "outer_split": outer_split,
                    "outer_repeat": outer_split // n_splits,
                    "outer_fold": outer_split % n_splits,
                    **asdict(wrapper.optimization_result_),
                }
                for outer_split, wrapper in enumerate(fitted_estimators)
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
        self._validate_class_distribution(y_train, classifier_name)
        model = self._build_model_for_target(classifier_name, y_train, X_train)
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

        winning_model = self._build_model_for_target(
            best["classifier_key"], y_train, X_train
        )
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
