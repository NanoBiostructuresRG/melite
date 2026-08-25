# SPDX-License-Identifier: LGPL-3.0-or-later
"""Result writing utilities for MELITE.

This module provides :class:`ResultManager`, which writes evaluation outputs
to disk, including the human-readable TXT report and structured CSV evidence
produced by the main evaluation pipeline.
"""

import csv
import os
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

from .plot_metrics import plot_f1_macro_evidence
from .version import __version__

__all__ = ["ResultManager"]

_REPORT_PROJECT_NAME = "MELITE"
_REPORT_LICENSE = "LGPL-3.0-or-later"


def _safe_filename_part(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("_")
    return safe or "dataset"


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
    >>> rm.write_csv([{"classifier_name": "SVC", "f1_macro": 0.85}],
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
                       {_REPORT_PROJECT_NAME}
            Multi-Model Classifier Evaluator
-----------------------------------------------------
Classifiers: SVC, RandomForest, XGBoost, Stacking (opt-in)
CLI: melite run | melite export
Package: melite
Version: {__version__}
Licence: {_REPORT_LICENSE}
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
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write(self._get_header())
            f.write(content)

    def write_csv(self, rows: list[dict], path: Path | str, smoke: bool = False) -> None:
        """Write selected evaluation results to a CSV file.

        Parameters
        ----------
        rows : list of dict
            List of result dictionaries, one per trained configuration. Each
            dict may include dataset identity and metadata fields in addition
            to classifier performance metrics.
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
            "reduction_type", "classifier_name", "parameters", "f1_macro", "f1_std",
            "accuracy", "acc_std", "auc_roc", "auc_std", "smoke",
        ]
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, "smoke": smoke})

    @staticmethod
    def _write_evaluation_csv(
        rows: list[dict],
        path: Path | str,
        fieldnames: list[str],
        smoke: bool,
    ) -> None:
        if not rows:
            return

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, "smoke": smoke})

    def write_evaluations_csv(
        self, rows: list[dict], path: Path | str, smoke: bool = False
    ) -> None:
        """Write one aggregate evaluation row per dataset and classifier."""
        fieldnames = [
            "dataset", "family", "method", "variant", "level", "description",
            "reduction_type", "classifier_name", "f1_macro", "f1_std", "accuracy",
            "acc_std", "auc_roc", "auc_std", "selected", "smoke",
        ]
        self._write_evaluation_csv(rows, path, fieldnames, smoke)

    def write_evaluation_folds_csv(
        self, rows: list[dict], path: Path | str, smoke: bool = False
    ) -> None:
        """Write one evaluation row per dataset, classifier, and outer fold."""
        fieldnames = [
            "dataset", "family", "method", "variant", "level", "description",
            "reduction_type", "classifier_name", "outer_split", "outer_repeat",
            "outer_fold", "f1_macro", "accuracy", "auc_roc", "selected", "smoke",
        ]
        self._write_evaluation_csv(rows, path, fieldnames, smoke)

    def write_evaluation_figures(
        self, rows: list[dict], smoke: bool = False
    ) -> None:
        """Write one outer-CV F1-macro evidence figure per dataset."""
        if not rows:
            return

        scores_by_dataset: dict[str, dict[str, list[float]]] = {}
        selected_by_dataset = {}
        for row in rows:
            dataset_id = row["dataset"]
            classifier_name = row["classifier_name"]
            classifier_scores = scores_by_dataset.setdefault(dataset_id, {})
            classifier_scores.setdefault(classifier_name, []).append(row["f1_macro"])
            if row["selected"] is True:
                selected_by_dataset[dataset_id] = classifier_name

        figures_dir = Path(self.output_file).parent / "figures"
        for dataset_id, classifier_scores in scores_by_dataset.items():
            save_to = (
                figures_dir
                / f"evaluation_f1_macro_{_safe_filename_part(dataset_id)}.png"
            )
            fig = plot_f1_macro_evidence(
                classifier_scores=classifier_scores,
                selected_classifier=selected_by_dataset[dataset_id],
                dataset_id=dataset_id,
                save_to=save_to,
                smoke=smoke,
            )
            plt.close(fig)
