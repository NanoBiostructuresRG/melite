# SPDX-License-Identifier: LGPL-3.0-or-later
"""Result writing utilities for MELITE.

This module provides :class:`ResultManager`, which writes evaluation outputs
to disk, including the human-readable TXT report and structured CSV evidence
produced by the main evaluation pipeline.
"""

import csv
import os
from datetime import datetime
from pathlib import Path

from .version import PROJECT_LICENSE, PROJECT_NAME, __version__

__all__ = ["ResultManager"]


class ResultManager:
    """Write evaluation results and evidence to TXT and CSV files.

    Parameters
    ----------
    output_file : str or pathlib.Path
        Full path to the TXT results file. The parent directory is created
        automatically if it does not exist.

    Examples
    --------
    >>> rm = ResultManager("output/results.txt")
    >>> rm.write_results("Model: SVC\\nF1: 0.85")
    >>> rm.write_csv([{"model_name": "SVC", "f1_macro": 0.85}],
    ...              "output/results.csv")
    """

    def __init__(self, output_file):
        self.output_file = output_file

        output_dir = os.path.dirname(self.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def _get_header(self):
        return f"""
=====================================================
                       {PROJECT_NAME}
            Multi-Model Classifier Evaluator
-----------------------------------------------------
Models: SVC, RandomForest, XGBoost, Stacking (opt-in)
CLI: melite run | melite export
Package: melite
Version: {__version__}
Licence: {PROJECT_LICENSE}
Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
-----------------------------------------------------
Repository: https://github.com/NanoBiostructuresRG/melite
=====================================================

"""

    def write_results(self, content: str) -> None:
        """Write a human-readable TXT report to disk.

        The report begins with a fixed header containing project metadata and
        execution timestamp, followed by *content*.

        Parameters
        ----------
        content : str
            Body of the report, typically the concatenated per-configuration
            result strings produced by the evaluation pipeline.

        Notes
        -----
        The header includes the current ``__version__`` string from
        :mod:`melite.version`, so the report always reflects the version that
        generated it.
        """
        try:
            with open(self.output_file, "w", encoding="utf-8") as f:
                f.write(self._get_header())
                f.write(content)
        except Exception as e:
            print(f"Error writing results: {e}")

    def write_csv(self, rows: list[dict], path: Path | str, smoke: bool = False) -> None:
        """Write selected evaluation results to a CSV file.

        Parameters
        ----------
        rows : list of dict
            List of result dictionaries, one per trained configuration. Each
            dict may include dataset identity and metadata fields in addition
            to model performance metrics.
        path : str or pathlib.Path
            Destination path for the CSV file. Parent directories are created
            automatically if they do not exist.
        smoke : bool, optional
            Whether the run was executed in smoke mode. When ``True``, a
            ``smoke`` column is set to ``True`` for every row, which causes
            :class:`~melite.export_best_model.Finalizer` to block export
            unless ``--force`` is passed. Default is ``False``.

        Notes
        -----
        If *rows* is empty, no file is written and the method returns silently.
        """
        if not rows:
            return

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "dataset", "family", "method", "variant", "level", "description",
            "reduction_type", "model_name", "parameters", "f1_macro", "f1_std",
            "accuracy", "acc_std", "auc_roc", "auc_std", "smoke",
        ]
        try:
            with open(path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow({**row, "smoke": smoke})
        except Exception as e:
            print(f"Error writing CSV: {e}")

    @staticmethod
    def _write_evaluation_csv(
        rows: list[dict],
        path: Path | str,
        fieldnames: list[str],
        smoke: bool,
        error_message: str,
    ) -> None:
        if not rows:
            return

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow({**row, "smoke": smoke})
        except Exception as e:
            print(f"{error_message}: {e}")

    def write_evaluations_csv(
        self, rows: list[dict], path: Path | str, smoke: bool = False
    ) -> None:
        """Write one aggregate evaluation row per dataset and model family."""
        fieldnames = [
            "dataset", "family", "method", "variant", "level", "description",
            "reduction_type", "model_name", "f1_macro", "f1_std", "accuracy",
            "acc_std", "auc_roc", "auc_std", "selected", "smoke",
        ]
        self._write_evaluation_csv(
            rows, path, fieldnames, smoke, "Error writing evaluations CSV"
        )

    def write_evaluation_folds_csv(
        self, rows: list[dict], path: Path | str, smoke: bool = False
    ) -> None:
        """Write one evaluation row per dataset, model family, and outer fold."""
        fieldnames = [
            "dataset", "family", "method", "variant", "level", "description",
            "reduction_type", "model_name", "outer_split", "outer_repeat",
            "outer_fold", "f1_macro", "accuracy", "auc_roc", "selected", "smoke",
        ]
        self._write_evaluation_csv(
            rows, path, fieldnames, smoke, "Error writing evaluation folds CSV"
        )
