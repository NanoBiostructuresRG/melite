# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
import numpy as np
from pathlib import Path

from mosaic.config import Config
from mosaic.load_dataset import load_dataset
from mosaic.result_manager import ResultManager
from mosaic.model_training import MultiModelTrainer

logger = logging.getLogger(__name__)

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
            logger.info("SMOKE TEST — reduced grid and CV. Results are not benchmark-quality.")
            print(_SMOKE_WARNING)

        for reduction_type in self.config.REDUCTION_TYPES:
            logger.info("Running with %s...", reduction_type)

            dataset = load_dataset(
                self.config, reduction_type, self.config.REDUCTION_LEVELS
            )
            if not dataset:
                logger.warning("No data found for %s. Skipping.", reduction_type)
                continue

            for key, (X_train, y_train) in dataset.items():
                level = int(key.replace(reduction_type, ""))
                logger.info("Training with %s (level=%d).", key, level)

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
        logger.info("Final report written to %s", self.config.RESULTS_FILE)

        csv_path = Path(self.config.PATHS["OUTPUT"]) / "results.csv"
        self.result_manager.write_csv(self.csv_rows, csv_path, smoke=self.config.SMOKE)
        logger.info("CSV file written to %s", csv_path)
