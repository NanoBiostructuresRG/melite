# SPDX-License-Identifier: LGPL-3.0-or-later
__all__ = ["ResultManager"]
import csv
import os
from datetime import datetime
from pathlib import Path

from mosaic.version import __version__


class ResultManager:
    def __init__(self, output_file):
        self.output_file = output_file

        output_dir = os.path.dirname(self.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def _get_header(self):
        return f"""
=====================================================
                       MOSAIC
     A multi-model benchmarking toolkit for ML-CV
               with PCA/UMAP reduction
             and GridSearch optimization
-----------------------------------------------------
          Models: SVC, RandomForest, XGBoost
          Exporter CLI: mosaic export
-----------------------------------------------------
Developer: Flavio F. Contreras-Torres
Version: v{__version__} - May, 2025. Oviedo
Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
-----------------------------------------------------
GitHub: https://github.com/NanoBiostructuresRG
=====================================================

"""

    def write_results(self, content: str) -> None:
        try:
            with open(self.output_file, "w", encoding="utf-8") as f:
                f.write(self._get_header())
                f.write(content)
        except Exception as e:
            print(f"Error writing results: {e}")

    def write_csv(self, rows: list[dict], path: Path | str, smoke: bool = False) -> None:
        """Write benchmark results to a CSV file.

        Args:
            rows: List of result dicts, one per trained configuration.
            path: Destination path for the CSV file.
            smoke: Whether the run was executed in smoke mode.
        """
        if not rows:
            return

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "reduction_type", "level", "model_name", "parameters",
            "f1_macro", "f1_std", "accuracy", "acc_std", "auc_roc", "auc_std",
            "smoke",
        ]
        try:
            with open(path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow({**row, "smoke": smoke})
        except Exception as e:
            print(f"Error writing CSV: {e}")
