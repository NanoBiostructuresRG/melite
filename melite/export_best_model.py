# SPDX-License-Identifier: LGPL-3.0-or-later
"""Final model export workflow for MELITE.

This module provides :class:`Finalizer`, which reads an evaluation results file,
lets the user select a configuration row, reconstructs the selected model,
retrains it on all available data, and serialises the trained model as a
``.pkl`` artifact.

A smoke-mode guard prevents accidental export of smoke-mode results.
"""

import ast
import json
import logging
import math
import re
import sys
from numbers import Real
from pathlib import Path
from typing import Any, Tuple

from catboost import CatBoostClassifier
import joblib
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from melite.config import Config
from melite.load_dataset import _load_one_dataset, load_datasets
from melite.model_catalog import MODEL_CATALOG
from melite.probability_calibration import ProbabilityCalibratedClassifier
from melite.regression_models import (
    build_regression_model,
    validate_polynomial_expansion,
    validate_regression_inputs,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

__all__ = ["Finalizer"]

logger = logging.getLogger(__name__)

MODEL_MAP = {
    "GaussianNB": GaussianNB,
    "RandomForestClassifier": RandomForestClassifier,
    "XGBClassifier": XGBClassifier,
}

METRIC_COLUMNS = [
    "dataset",
    "family",
    "method",
    "variant",
    "level",
    "description",
    "reduction_type",
    "classifier_name",
    "f1_macro",
    "accuracy",
    "auc_roc",
]
REGRESSION_METRIC_COLUMNS = [
    "dataset",
    "task",
    "family",
    "method",
    "variant",
    "level",
    "description",
    "reduction_type",
    "regressor_key",
    "regressor_name",
    "rmse",
    "mae",
    "r2",
]
REGRESSION_RESULT_COLUMNS = [
    "dataset",
    "task",
    "family",
    "method",
    "variant",
    "level",
    "description",
    "reduction_type",
    "regressor_key",
    "regressor_name",
    "parameters",
    "rmse",
    "rmse_std",
    "mae",
    "mae_std",
    "r2",
    "r2_std",
    "smoke",
]
_REGRESSOR_NAMES = {
    model.key: model.public_name
    for model in MODEL_CATALOG
    if model.prediction_task == "regression"
}
_REGRESSION_PARAMETER_NAMES = {
    "linear": frozenset(),
    "poly": frozenset(),
    "ridge": frozenset({"ridge__alpha"}),
    "lasso": frozenset({"lasso__alpha"}),
}


def _has_value(value: Any) -> bool:
    return value is not None and not pd.isna(value) and str(value).strip() != ""


def _safe_filename_part(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("_")


def _build_cv_strategy(
    cv_config: dict | None,
    random_state: int,
):
    if cv_config is None:
        cv_config = {
            "n_splits": 5,
            "n_repeats": 3,
            "inner_n_splits": 3,
        }
    # StackingClassifier uses cross_val_predict internally, which requires
    # one partition of the data rather than repeated test assignments.
    return StratifiedKFold(
        n_splits=cv_config["inner_n_splits"],
        shuffle=True,
        random_state=random_state,
    )


def _build_stacking_classifier(
    random_state: int,
    cv_config: dict | None = None,
) -> StackingClassifier:
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
        cv=_build_cv_strategy(cv_config, random_state),
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1,
    )


class DatasetLoader:
    """Load feature matrices and labels for model retraining.

    Parameters
    ----------
    cfg : melite.config.Config
        MELITE configuration object providing dataset and input paths.
    """

    _CANDIDATE_KEYS = ("X{lvl}", "X_{lvl}", "{rtype}{lvl}")

    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._data_root = Path(cfg.PATHS["DATASET"])
        self._labels: np.ndarray | None = None

    def load_row(self, row: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Load the dataset referenced by a result row.

        Dataset-registry result rows are resolved by their ``dataset`` id in
        ``cfg.DATASETS``. Older result rows without ``dataset`` fall back to
        the legacy ``reduction_type`` + ``level`` lookup.
        """
        if "dataset" in row and _has_value(row.get("dataset")):
            dataset_id = str(row.get("dataset"))
            try:
                dataset = load_datasets(self._cfg)[dataset_id]
            except KeyError as exc:
                raise KeyError(
                    f"Dataset '{dataset_id}' from results.csv is not registered "
                    "in cfg.DATASETS."
                ) from exc
            return dataset["X"], dataset["y"]

        return self.load(row.reduction_type, int(row.level))

    def load(self, reduction: str, level: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load feature matrix and label vector for a given configuration.

        Tries an individual file (e.g. ``PCA85.npz``) first, then falls back
        to an aggregated archive (e.g. ``PCAs.npz``).

        Parameters
        ----------
        reduction : str
            Reduction method prefix, e.g. ``"PCA"``.
        level : int
            Variance retention level, e.g. ``85``.

        Returns
        -------
        X : numpy.ndarray
            Feature matrix of shape ``(n_samples, n_features)``.
        y : numpy.ndarray
            Label vector of shape ``(n_samples,)``.

        Raises
        ------
        FileNotFoundError
            If neither an individual nor an aggregated file is found.
        KeyError
            If the requested level is absent from the aggregated archive.
        """
        X = self._try_individual_file(reduction, level)
        if X is None:
            X = self._try_aggregated_file(reduction, level)
        if X is None:
            raise FileNotFoundError(
                f"No data found for {reduction}{level}: neither an individual "
                f"file nor an entry inside an aggregated archive is present."
            )
        labels = self._ensure_labels()
        return X, labels

    def _try_individual_file(self, reduction: str, level: int) -> np.ndarray | None:
        fp = self._data_root / f"{reduction}{level}.npz"
        if not fp.exists():
            return None
        dataset_id = f"{reduction}{level}"
        spec = {
            "path": fp,
            "label_path": Path(self._cfg.PATHS["INPUT"]) / "labels.npy",
            "metadata": {
                "family": "dimensionality",
                "method": reduction,
                "level": level,
            },
        }
        dataset = _load_one_dataset(dataset_id, spec)
        self._labels = dataset["y"]
        return dataset["X"]

    def _try_aggregated_file(self, reduction: str, level: int) -> np.ndarray | None:
        fp = self._data_root / f"{reduction}s.npz"
        if not fp.exists():
            return None
        arr = np.load(fp)
        for pattern in self._CANDIDATE_KEYS:
            key = pattern.format(rtype=reduction, lvl=level)
            if key in arr:
                labels = self._ensure_labels()
                X = arr[key]
                if X.ndim != 2:
                    raise ValueError(
                        f"Legacy dataset '{reduction}{level}' X must be 2D; "
                        f"got shape {X.shape}."
                    )
                if not np.issubdtype(X.dtype, np.number):
                    raise ValueError(
                        f"Legacy dataset '{reduction}{level}' X must be numeric; "
                        f"got dtype {X.dtype}."
                    )
                if len(labels) != X.shape[0]:
                    raise ValueError(
                        f"Legacy dataset '{reduction}{level}' X/y length mismatch: "
                        f"X has {X.shape[0]} rows, y has {len(labels)} labels."
                    )
                return X
        raise KeyError(f"Level {level} not found inside {fp.name}.")

    def _ensure_labels(self) -> np.ndarray:
        labels = self._labels
        if labels is None:
            label_path = Path(self._cfg.PATHS["INPUT"]) / "labels.npy"
            labels = np.load(label_path)
            self._labels = labels
        return labels


class Finalizer:
    """Retrain a selected model on all data and export a ``.pkl`` artifact.

    Parameters
    ----------
    csv_path : pathlib.Path
        Path to the ``results.csv`` file produced by the evaluation workflow.
    output_dir : pathlib.Path
        Directory where the ``.pkl`` artifact will be saved.
    cfg : melite.config.Config
        MELITE configuration object.
    row_index : int or None, optional
        Row index from *csv_path* to export non-interactively. If ``None``,
        the user is prompted to select a row interactively. Default is ``None``.
    force : bool, optional
        If ``True``, override the smoke-mode export guard and proceed with a
        visible warning. Default is ``False``.

    Raises
    ------
    FileNotFoundError
        If *csv_path* does not exist.
    ValueError
        If *csv_path* uses the legacy ``model_name`` column instead of
        ``classifier_name``.
    """

    def __init__(
        self,
        csv_path: Path,
        output_dir: Path,
        cfg: Config,
        row_index: int | None = None,
        force: bool = False,
    ):
        self._csv_path = csv_path
        self._output_dir = output_dir
        self._cfg = cfg
        self._row_index = row_index
        self._force = force

        if not Path(csv_path).exists():
            raise FileNotFoundError(
                f"Results file not found: {csv_path}. "
                "Run 'melite run' first to generate evaluation results."
            )

        self._metrics = pd.read_csv(csv_path)
        if (
            "classifier_name" not in self._metrics.columns
            and "model_name" in self._metrics.columns
        ):
            raise ValueError(
                "The results CSV column 'model_name' was renamed to "
                "'classifier_name' in MELITE v0.2.4. Regenerate results.csv "
                "with the current MELITE version before exporting."
            )
        self._result_task = self._identify_result_schema()
        self._loader = DatasetLoader(cfg)

    def _identify_result_schema(self) -> str:
        columns = list(self._metrics.columns)
        if "classifier_name" in columns:
            if any(
                column in columns
                for column in ("regressor_key", "regressor_name")
            ) or (
                "task" in columns
                and self._metrics["task"]
                .astype(str)
                .str.strip()
                .eq("regression")
                .any()
            ):
                raise ValueError(
                    "Results CSV mixes classification and regression task schemas."
                )
            return "classification"

        regression_markers = {"task", "regressor_key", "regressor_name"}
        if regression_markers.intersection(columns):
            if columns != REGRESSION_RESULT_COLUMNS:
                raise ValueError(
                    "Regression results CSV must use the exact regression_results.csv "
                    f"schema: {REGRESSION_RESULT_COLUMNS}."
                )
            return "regression"

        raise ValueError(
            "Unsupported results CSV schema: expected 'classifier_name' for "
            "classification or the explicit regression_results.csv schema."
        )

    def _check_smoke_guard(self, row: pd.Series) -> None:
        """Block export of smoke-mode results unless ``--force`` is active.

        Parameters
        ----------
        row : pandas.Series
            Selected result row from ``results.csv``.

        Notes
        -----
        The ``smoke`` column value is read as a string from CSV and compared
        case-insensitively. If ``smoke == "True"`` and ``force`` is ``False``,
        the process exits with code 1.
        """
        smoke_val = row.get("smoke", False)
        is_smoke = str(smoke_val).strip().lower() == "true"
        if is_smoke and not self._force:
            print(
                "\n[ERROR] This result was generated in smoke mode and is not "
                "suitable for final model export.\n"
                "        Run 'melite run' (without --smoke) to generate full "
                "evaluation results,\n"
                "        or use 'melite export --force' to override this guard.\n"
            )
            logger.error("Export blocked: smoke-mode result. Use --force to override.")
            sys.exit(1)
        if is_smoke and self._force:
            print(
                "\n[WARNING] Exporting a smoke-mode result. "
                "This model is not intended for final use.\n"
            )
            logger.warning("Exporting smoke-mode result (--force override active).")

    def run(self) -> None:
        """Execute the full export workflow.

        Displays the metrics table, prompts or uses ``--row`` to select a
        configuration, checks the smoke guard, loads the dataset, reconstructs the
        selected model, retrains it on all available data, and saves the ``.pkl``
        artifact.

        Notes
        -----
        The smoke guard calls :func:`sys.exit` with code 1 if the selected row
        was generated in smoke mode and ``force`` is ``False``.
        """
        self._show_metrics()
        row = self._get_selected_row()
        self._check_smoke_guard(row)
        if self._result_task == "regression":
            self._validate_regression_row(row)
            X, y = self._loader.load_row(row)
            X, y = validate_regression_inputs(X, y)
            regressor_key = str(row.regressor_key)
            if regressor_key == "poly":
                validate_polynomial_expansion(X.shape[1])
            model = self._build_regression_model(
                regressor_key,
                str(row.regressor_name),
                row.parameters,
            )
            model_name = str(row.regressor_name)
        else:
            X, y = self._loader.load_row(row)
            model = self._build_model(
                row.classifier_name,
                row.parameters,
                cv_config=self._cfg.CV_CONFIG,
                random_state=self._cfg.RANDOM_STATE,
            )
            model_name = row.classifier_name

        logger.info(
            "Training %s on %s using all available data...",
            model_name,
            self._row_dataset_label(row),
        )
        print(
            f"\nTraining {model_name} on {self._row_dataset_label(row)} "
            "using all available data..."
        )
        model.fit(X, y)
        artefact_path = self._save_model(model, row)
        logger.info("Model saved to: %s", artefact_path.resolve())
        print(f"\nModel saved to: {artefact_path.resolve()}")

    def _show_metrics(self) -> None:
        metric_columns = (
            REGRESSION_METRIC_COLUMNS
            if self._result_task == "regression"
            else METRIC_COLUMNS
        )
        cols = [c for c in metric_columns if c in self._metrics.columns]
        print(self._metrics[cols].to_string(index=True, float_format="%.4f"))

    def _get_selected_row(self) -> pd.Series:
        if self._row_index is None:
            return self._prompt_row()
        if 0 <= self._row_index < len(self._metrics):
            return self._metrics.iloc[self._row_index]
        raise ValueError(
            f"Invalid row index {self._row_index}; "
            f"expected a value between 0 and {len(self._metrics) - 1}."
        )

    def _prompt_row(self) -> pd.Series:
        while True:
            reply = input("\nEnter the row number to keep: ").strip()
            if reply.isdigit() and 0 <= int(reply) < len(self._metrics):
                return self._metrics.iloc[int(reply)]
            print("[ERR] Invalid row number; please try again.")

    def _validate_regression_row(self, row: pd.Series) -> None:
        if str(row.get("task", "")).strip() != "regression":
            raise ValueError(
                "Regression result row must declare task = 'regression'."
            )

        regressor_key = str(row.get("regressor_key", "")).strip()
        try:
            expected_name = _REGRESSOR_NAMES[regressor_key]
        except KeyError as exc:
            valid = ", ".join(_REGRESSOR_NAMES)
            raise ValueError(
                f"Unsupported regression model key: {regressor_key!r}. "
                f"Valid keys are: {valid}."
            ) from exc
        regressor_name = str(row.get("regressor_name", "")).strip()
        if regressor_name != expected_name:
            raise ValueError(
                f"Regression model identity mismatch for {regressor_key!r}: "
                f"expected {expected_name!r}, got {regressor_name!r}."
            )

        dataset_id = str(row.get("dataset", "")).strip()
        if not dataset_id or dataset_id not in self._cfg.DATASETS:
            raise KeyError(
                f"Dataset {dataset_id!r} from regression results is not "
                "registered in cfg.DATASETS."
            )
        configured_task = self._cfg.DATASETS[dataset_id].get(
            "task", "classification"
        )
        if configured_task != "regression":
            raise ValueError(
                f"Dataset {dataset_id!r} is configured for task "
                f"{configured_task!r}, not 'regression'."
            )

    @staticmethod
    def _parse_regression_parameters(serialised_params: str) -> dict[str, Any]:
        if not isinstance(serialised_params, str):
            raise ValueError("Regression parameters must be a JSON object string.")

        def reject_constant(value):
            raise ValueError(
                f"Regression parameters contain non-finite JSON value {value!r}."
            )

        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(
                        f"Regression parameters contain duplicate key {key!r}."
                    )
                result[key] = value
            return result

        try:
            params = json.loads(
                serialised_params,
                parse_constant=reject_constant,
                object_pairs_hook=unique_object,
            )
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(
                "Regression parameters must be valid strict JSON."
            ) from exc
        if not isinstance(params, dict):
            raise ValueError("Regression parameters JSON must contain an object.")
        return params

    @staticmethod
    def _build_regression_model(
        regressor_key: str,
        regressor_name: str,
        serialised_params: str,
    ) -> Any:
        try:
            expected_name = _REGRESSOR_NAMES[regressor_key]
            expected_parameters = _REGRESSION_PARAMETER_NAMES[regressor_key]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported regression model key: {regressor_key!r}."
            ) from exc
        if regressor_name != expected_name:
            raise ValueError(
                f"Regression model identity mismatch for {regressor_key!r}: "
                f"expected {expected_name!r}, got {regressor_name!r}."
            )

        params = Finalizer._parse_regression_parameters(serialised_params)
        actual_parameters = set(params)
        if actual_parameters != expected_parameters:
            raise ValueError(
                f"Invalid parameters for regressor {regressor_key!r}: expected "
                f"{sorted(expected_parameters)}, got {sorted(actual_parameters)}."
            )
        if expected_parameters:
            alpha_name = next(iter(expected_parameters))
            alpha = params[alpha_name]
            if (
                isinstance(alpha, bool)
                or not isinstance(alpha, Real)
                or not math.isfinite(float(alpha))
                or not 1e-4 <= alpha <= 1e2
            ):
                raise ValueError(
                    f"Regression parameter {alpha_name!r} must be a finite "
                    "number in the approved domain [0.0001, 100.0]."
                )

        model = build_regression_model(regressor_key)
        return model.set_params(**params)

    @staticmethod
    def _build_model(
        name: str,
        serialised_params: str,
        random_state: int,
        cv_config: dict | None = None,
    ) -> Any:
        params = ast.literal_eval(serialised_params)
        if name == "SVC":
            model = SklearnPipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "svc",
                        ProbabilityCalibratedClassifier(
                            estimator=SVC(),
                            max_calibration_cv=5,
                        ),
                    ),
                ]
            )
            svc_params = {
                f"svc__estimator__{key.removeprefix('svc__')}": value
                for key, value in params.items()
            }
            return model.set_params(**svc_params)
        if name == "LogisticRegression":
            model = SklearnPipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "lr",
                        LogisticRegression(
                            solver="lbfgs",
                            max_iter=1000,
                            random_state=random_state,
                        ),
                    ),
                ]
            )
            return model.set_params(**params)
        if name == "KNeighborsClassifier":
            model = SklearnPipeline(
                [
                    ("scaler", StandardScaler()),
                    ("knn", KNeighborsClassifier(metric="minkowski")),
                ]
            )
            return model.set_params(**params)
        if name == "DecisionTreeClassifier":
            model = DecisionTreeClassifier(random_state=random_state)
            return model.set_params(**params)
        if name == "AdaBoostClassifier":
            model = AdaBoostClassifier(random_state=random_state)
            return model.set_params(**params)
        if name == "LGBMClassifier":
            model = LGBMClassifier(
                random_state=random_state,
                n_jobs=1,
                verbosity=-1,
            )
            return model.set_params(**params)
        if name == "CatBoostClassifier":
            model = CatBoostClassifier(
                random_seed=random_state,
                thread_count=1,
                verbose=False,
                allow_writing_files=False,
                task_type="CPU",
            )
            return model.set_params(**params)
        if name == "StackingClassifier":
            model = _build_stacking_classifier(
                random_state=random_state,
                cv_config=cv_config,
            )
            return model.set_params(**params)
        try:
            return MODEL_MAP[name](**params)
        except KeyError as exc:
            raise ValueError(f"Unsupported classifier type: {name}") from exc

    @staticmethod
    def _row_dataset_label(row: pd.Series) -> str:
        if "dataset" in row and _has_value(row.get("dataset")):
            return _safe_filename_part(row.get("dataset"))
        return _safe_filename_part(f"{row.reduction_type}{int(row.level)}")

    def _save_model(self, model: Any, row: pd.Series) -> Path:
        self._output_dir.mkdir(exist_ok=True)
        model_name = (
            row.regressor_name
            if self._result_task == "regression"
            else row.classifier_name
        )
        filename = f"Model_{model_name}_{self._row_dataset_label(row)}.pkl"
        path = self._output_dir / filename
        joblib.dump(model, path)
        return path
