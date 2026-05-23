# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import csv
import numpy as np

from mosaic.config import Config
from mosaic.load_dataset import load_dataset
from mosaic.result_manager import ResultManager
from mosaic.model_training import MultiModelTrainer

_SMOKE_WARNING = (
    "\n[SMOKE TEST] Using reduced grid and CV. "
    "Results are not benchmark-quality.\n"
)


class Pipeline:
    def __init__(self, config: Config):
        self.config = config
        self.model_trainer = MultiModelTrainer(config)

    def run(self, X_train, y_train, reduction_type, level):
        return self.model_trainer.train_and_select_best_model(
            X_train, y_train, reduction_type, level
        )


class Main:
    def __init__(self, smoke: bool = False, user_config=None):
        self.config = Config(smoke=smoke, user_config=user_config)
        self.config.setup()
        self.pipeline = Pipeline(self.config)
        self.result_manager = ResultManager(self.config.RESULTS_FILE)
        self.final_results = []
        self.csv_rows = []

    # ------------------------------------------------------------------ #
    @staticmethod
    def _clean_params(params):
        return {
            k: round(float(v), 4) if isinstance(v, (float, np.floating)) else v
            for k, v in params.items()
        }

    # ------------------------------------------------------------------ #
    def run(self):
        if self.config.SMOKE:
            print(_SMOKE_WARNING)

        for reduction_type in self.config.REDUCTION_TYPES:
            print(f"Running with {reduction_type}...")

            dataset = load_dataset(
                self.config, reduction_type, self.config.REDUCTION_LEVELS
            )
            if not dataset:
                print(f"No data found for {reduction_type}. Skipping.")
                continue

            for key, (X_train, y_train) in dataset.items():
                level = int(key.replace(reduction_type, ""))
                print(f"Training with {key} (level={level}).")

                (
                    best_model,
                    best_params,
                    best_f1,
                    f1_std,
                    best_acc,
                    acc_std,
                    best_auc,
                    auc_std,
                ) = self.pipeline.run(X_train, y_train, reduction_type, level)

                params = self._clean_params(best_params)
                model_name = best_model.__class__.__name__

                self.final_results.append(
                    "\n".join(
                        [
                            f"Results for {key} (level {level}):",
                            f"Model Selected: {model_name}",
                            f"Best ML-model Parameters: {params}",
                            f"F1-macro (CV mean): {round(best_f1, 4)} ± {round(f1_std, 4)}",
                            f"Accuracy (CV mean): {round(best_acc, 4)} ± {round(acc_std, 4)}",
                            (
                                f"AUC-ROC (CV mean): {round(best_auc, 4)} ± {round(auc_std, 4)}"
                                if best_auc is not None
                                else "AUC-ROC (CV mean): N/A"
                            ),
                            "------------------------------",
                        ]
                    )
                )

                self.csv_rows.append(
                    {
                        "reduction_type": reduction_type,
                        "level": int(key.replace(reduction_type, "")),
                        "model_name": model_name,
                        "parameters": str(params),
                        "f1_macro": round(best_f1, 4),
                        "f1_std": round(f1_std, 4),
                        "accuracy": round(best_acc, 4),
                        "acc_std": round(acc_std, 4),
                        "auc_roc": round(best_auc, 4) if best_auc is not None else "N/A",
                        "auc_std": round(auc_std, 4) if auc_std is not None else "N/A",
                    }
                )

        final_report = "\n".join(self.final_results)
        self.result_manager.write_results(final_report)
        print("Final report written to", self.config.RESULTS_FILE)

        csv_path = os.path.join(self.config.PATHS["OUTPUT"], "results.csv")
        with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
            fieldnames = [
                "reduction_type", "level", "model_name", "parameters",
                "f1_macro", "f1_std", "accuracy", "acc_std", "auc_roc", "auc_std",
            ]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.csv_rows)

        print(f"CSV file written to {csv_path}")
