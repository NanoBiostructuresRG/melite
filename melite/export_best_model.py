# SPDX-License-Identifier: LGPL-3.0-or-later
"""Final model export workflow for MELITE.

This module provides :class:`Finalizer`, which reads an evaluation results file,
lets the user select a configuration row, reconstructs the selected model,
retrains it on all available data, and serialises the trained model as a
``.pkl`` artifact.

A smoke-mode guard prevents accidental export of smoke-mode results.
"""

import ast
import logging
import re
import sys
from pathlib import Path
from typing import Any, Tuple

import joblib
import numpy as np
import pandas as pd
from .config import Config
from .load_dataset import load_datasets, _load_one_dataset
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

__all__ = ["Finalizer"]

logger = logging.getLogger(__name__)

MODEL_MAP = {
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
                "with MELITE v0.2.4 before exporting."
            )
        self._loader = DatasetLoader(cfg)

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
        X, y = self._loader.load_row(row)
        model = self._build_model(
            row.classifier_name,
            row.parameters,
            cv_config=self._cfg.CV_CONFIG,
            random_state=self._cfg.RANDOM_STATE,
        )

        logger.info(
            "Training %s on %s using all available data...",
            row.classifier_name,
            self._row_dataset_label(row),
        )
        print(
            f"\nTraining {row.classifier_name} on {self._row_dataset_label(row)} "
            "using all available data..."
        )
        model.fit(X, y)
        artefact_path = self._save_model(model, row)
        logger.info("Model saved to: %s", artefact_path.resolve())
        print(f"\nModel saved to: {artefact_path.resolve()}")

    def _show_metrics(self) -> None:
        cols = [c for c in METRIC_COLUMNS if c in self._metrics.columns]
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
                    ("svc", SVC()),
                ]
            )
            svc_params = {
                key if "__" in key else f"svc__{key}": value
                for key, value in params.items()
            }
            svc_params.setdefault("svc__probability", True)
            return model.set_params(**svc_params)
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
        filename = f"Model_{row.classifier_name}_{self._row_dataset_label(row)}.pkl"
        path = self._output_dir / filename
        joblib.dump(model, path)
        return path
