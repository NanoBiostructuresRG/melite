# SPDX-License-Identifier: LGPL-3.0-or-later
__all__ = ["MultiModelTrainer"]
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_validate
from abc import ABC, abstractmethod
import numpy as np

class ModelTrainer(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def train_and_select_best_model(self, X_train, y_train, reduction_type, level):
        pass

class MultiModelTrainer(ModelTrainer):
    def __init__(self, config):
        super().__init__(config)

        rs = getattr(self.config, "RANDOM_STATE", 42) 

        # Dictionary of ML-models
        self.model_builders = {
            "svc": lambda: SVC(probability=True, random_state=rs),
            "rf": lambda: RandomForestClassifier(random_state=rs, n_jobs=-1),
            "xgb": lambda: XGBClassifier(eval_metric="logloss", random_state=rs, n_jobs=-1),
        }

    def _build_cv_strategy(self):
        cv_cfg = self.config.get_cv_config()
        return RepeatedStratifiedKFold(
            n_splits=cv_cfg["n_splits"],
            n_repeats=cv_cfg["n_repeats"],
            random_state=cv_cfg["random_state"],
        )

    def _filter_param_grid(self, model_name):
        return [
            {k: v for k, v in g.items() if k != "model"}
            for g in self.config.PARAM_GRID
            if g["model"][0] == model_name
        ]

    def perform_grid_search(self, model, X_train, y_train, param_grid):
        cv = self._build_cv_strategy()
        grid = GridSearchCV(model, param_grid, scoring="f1_macro", cv=cv, n_jobs=-1)
        grid.fit(X_train, y_train)
        return grid.best_estimator_, grid.best_params_

    def cross_validate_model(self, model, X_train, y_train):
        cv = self._build_cv_strategy()
        scoring = {"f1": "f1_macro", "acc": "accuracy", "auc": "roc_auc"}
        scores = cross_validate(
            model,
            X_train,
            y_train,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            return_train_score=False,
        )

        f1_mean, f1_std = scores["test_f1"].mean(), scores["test_f1"].std()
        acc_mean, acc_std = scores["test_acc"].mean(), scores["test_acc"].std()
        auc_vals = scores.get("test_auc")
        auc_mean, auc_std = (
            (auc_vals.mean(), auc_vals.std()) if auc_vals is not None else (None, None)
        )

        return f1_mean, f1_std, acc_mean, acc_std, auc_mean, auc_std


    def train_and_select_best_model(self, X_train, y_train, reduction_type, level):
        best = {
            "model": None,
            "params": None,
            "f1": -1,
            "f1_std": 0,
            "acc": 0,
            "acc_std": 0,
            "auc": None,
            "auc_std": None,
        }

        for model_name in self.model_builders:
            model = self.model_builders[model_name]()
            param_grid = self._filter_param_grid(model_name)

            tuned_model, params = self.perform_grid_search(model, X_train, y_train, param_grid)

            (
                f1_mean,
                f1_std,
                acc_mean,
                acc_std,
                auc_mean,
                auc_std,
            ) = self.cross_validate_model(tuned_model, X_train, y_train)

            if f1_mean > best["f1"]:
                best.update(
                    {
                        "model": tuned_model,
                        "params": params,
                        "f1": f1_mean,
                        "f1_std": f1_std,
                        "acc": acc_mean,
                        "acc_std": acc_std,
                        "auc": auc_mean,
                        "auc_std": auc_std,
                    }
                )

        return (
            best["model"],
            best["params"],
            best["f1"],
            best["f1_std"],
            best["acc"],
            best["acc_std"],
            best["auc"],
            best["auc_std"],
        )