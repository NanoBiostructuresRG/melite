# SPDX-License-Identifier: LGPL-3.0-or-later
import argparse
import ast
import sys
from pathlib import Path
from typing import Any, Tuple

import joblib
import numpy as np
import pandas as pd
from mosaic.config import Config
from mosaic.plot_metrics import plot_cv_distributions
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold


MODEL_MAP = {
    "SVC": SVC,
    "RandomForestClassifier": RandomForestClassifier,
    "XGBClassifier": XGBClassifier,
}

METRIC_COLUMNS = [
    "reduction_type",
    "level",
    "model_name",
    "f1_macro",
    "accuracy",
    "auc_roc",
]


class DatasetLoader:
    _CANDIDATE_KEYS = (
        "X{lvl}",
        "X_{lvl}",
        "{rtype}{lvl}",
    )

    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._data_root = Path(cfg.PATHS["DATASET"])
        self._labels: np.ndarray | None = None

    def load(self, reduction: str, level: int) -> Tuple[np.ndarray, np.ndarray]:
        X = self._try_individual_file(reduction, level)
        if X is None:
            X = self._try_aggregated_file(reduction, level)
        if X is None:
            raise FileNotFoundError(
                f"No data found for {reduction}{level}: neither an individual "
                f"file nor an entry inside an aggregated archive is present."
            )
        return X, self._labels

    def _try_individual_file(self, reduction: str, level: int) -> np.ndarray | None:
        fp = self._data_root / f"{reduction}{level}.npz"
        if not fp.exists():
            return None
        arr = np.load(fp)
        self._ensure_labels()
        return arr[arr.files[0]]  # first array by convention

    def _try_aggregated_file(self, reduction: str, level: int) -> np.ndarray | None:
        fp = self._data_root / f"{reduction}s.npz"
        if not fp.exists():
            return None
        arr = np.load(fp)
        for pattern in self._CANDIDATE_KEYS:
            key = pattern.format(rtype=reduction, lvl=level)
            if key in arr:
                self._ensure_labels()
                return arr[key]
        raise KeyError(f"Level {level} not found inside {fp.name}.")

    def _ensure_labels(self) -> None:
        if self._labels is None:
            label_path = Path(self._cfg.PATHS["INPUT"]) / "labels.npy"
            self._labels = np.load(label_path)


class Finalizer:
    def __init__(
        self,
        csv_path: Path,
        output_dir: Path,
        cfg: Config,
        row_index: int | None = None,
    ):
        self._csv_path = csv_path
        self._output_dir = output_dir
        self._cfg = cfg
        self._row_index = row_index
        self._metrics = pd.read_csv(csv_path)
        self._loader = DatasetLoader(cfg)

    def _cv_and_plot(self, model, X, y, row, save_dir: Path) -> None:
        cv_cfg = self._cfg.get_cv_config()
        cv = RepeatedStratifiedKFold(
            n_splits=cv_cfg["n_splits"],
            n_repeats=cv_cfg["n_repeats"],
            random_state=cv_cfg["random_state"],
        )
        scoring = {"f1": "f1_macro", "acc": "accuracy", "auc": "roc_auc"}
        scores = cross_validate(
            model, X, y, scoring=scoring, cv=cv, n_jobs=-1, return_train_score=False
        )

        plot_cv_distributions(
            scores["test_f1"],
            scores["test_acc"],
            scores.get("test_auc"),
            model_name=row.model_name,
            params=row.parameters,
            save_to=save_dir
            / f"{row.model_name}_{row.reduction_type}{row.level}.png",
        )

    def run(self) -> None:
        self._show_metrics()
        row = self._get_selected_row()
        X, y = self._loader.load(row.reduction_type, int(row.level))
        model = self._build_model(row.model_name, row.parameters)

        figures_dir = Path(self._cfg.PATHS["OUTPUT"]) / "figures"
        self._cv_and_plot(model, X, y, row, figures_dir)

        print(
            f"\nTraining {row.model_name} on {row.reduction_type}{row.level} "
            "using all available data..."
        )
        model.fit(X, y)
        artefact_path = self._save_model(model, row)
        print(f"\nModel saved to: {artefact_path.resolve()}")

    def _show_metrics(self) -> None:
        print(self._metrics[METRIC_COLUMNS].to_string(index=True, float_format="%.4f"))

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
    def _build_model(name: str, serialised_params: str) -> Any:
        params = ast.literal_eval(serialised_params)
        if name == "SVC":
            params = {**params, "probability": True}
        try:
            return MODEL_MAP[name](**params)
        except KeyError as exc:
            raise ValueError(f"Unsupported model type: {name}") from exc

    def _save_model(self, model: Any, row: pd.Series) -> Path:
        self._output_dir.mkdir(exist_ok=True)
        filename = f"Model_{row.model_name}_{row.reduction_type}{row.level}.pkl"
        path = self._output_dir / filename
        joblib.dump(model, path)
        return path


def _build_arg_parser(cfg: Config) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select a benchmark configuration and persist the retrained model."
    )
    parser.add_argument(
        "-c",
        "--csv",
        type=Path,
        default=Path(cfg.PATHS["OUTPUT"]) / "results.csv",
        help="Path to the CSV file produced by the benchmarking phase.",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=Path,
        default=Path(cfg.PATHS["OUTPUT"]),
        help="Destination directory for the *.pkl* file.",
    )
    parser.add_argument(
        "--row",
        type=int,
        default=None,
        help="Row index from the results CSV to export without interactive prompt.",
    )
    return parser


def main() -> None:
    cfg = Config()
    args = _build_arg_parser(cfg).parse_args()
    Finalizer(args.csv, args.outdir, cfg, row_index=args.row).run()


if __name__ == "__main__":
    main()
