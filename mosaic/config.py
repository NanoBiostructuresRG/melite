# SPDX-License-Identifier: LGPL-3.0-or-later
"""Configuration loader for MOSAIC.

Reads mosaic/config_default.toml as the base configuration.
An optional user-supplied TOML file can override any key.
Hyperparameter grids are defined here in Python — they are
developer-facing and not expected to change between runs.
"""

__all__ = ["Config"]
import os
import random
import tomllib
from pathlib import Path

import numpy as np
from sklearn.model_selection import ParameterGrid

# Path to the default configuration file bundled with the package
_DEFAULT_CONFIG = Path(__file__).parent / "config_default.toml"


def _load_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base recursively. Override wins on conflict."""
    result = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class Config:
    """
    Configuration container for MOSAIC.

    Loads defaults from config_default.toml. If user_config is provided,
    its values are merged over the defaults (user values win, missing keys
    fall back to defaults).

    Args:
        smoke: If True, use reduced CV and grids for lightweight runs.
        user_config: Optional path to a user-supplied TOML file.
    """

    def __init__(
        self,
        smoke: bool = False,
        user_config: Path | None = None,
    ):
        self.SMOKE = smoke

        # Load and merge configuration
        cfg = _load_toml(_DEFAULT_CONFIG)
        if user_config is not None:
            user_cfg = _load_toml(Path(user_config))
            cfg = _deep_merge(cfg, user_cfg)

        # Paths
        self.PATHS = {
            "INPUT":   cfg["paths"]["input"],
            "DATASET": cfg["paths"]["dataset"],
            "OUTPUT":  cfg["paths"]["output"],
        }
        self.RESULTS_FILE = os.path.join(self.PATHS["OUTPUT"], "results.txt")

        # Benchmark settings
        self.RANDOM_STATE     = cfg["benchmark"]["random_state"]
        self.REDUCTION_TYPES  = cfg["benchmark"]["reduction_types"]
        self.REDUCTION_LEVELS = cfg["benchmark"]["levels"]
        self.ACTIVE_MODELS    = cfg["models"]["active"]

        # Cross-validation
        cv_section = cfg["cv_smoke"] if smoke else cfg["cv"]
        self.CV_CONFIG = {
            "n_splits":     cv_section["n_splits"],
            "n_repeats":    cv_section["n_repeats"],
            "random_state": self.RANDOM_STATE,
        }

        # Hyperparameter grids — developer-facing, defined in Python
        self.PARAM_GRID = self._build_param_grid()
        self.PARAM_GRID_BY_MODEL = self._group_param_grid_by_model()

    # ------------------------------------------------------------------ #
    # Hyperparameter grids
    # ------------------------------------------------------------------ #

    def _build_param_grid(self) -> list:
        if self.SMOKE:
            return [
                {
                    "model": ["svc"],
                    "kernel": ["linear"],
                    "C": [1],
                },
                {
                    "model": ["rf"],
                    "n_estimators": [50],
                    "max_depth": [5],
                    "max_features": ["sqrt"],
                    "min_samples_split": [2],
                    "min_samples_leaf": [1],
                },
                {
                    "model": ["xgb"],
                    "n_estimators": [20],
                    "learning_rate": [0.1],
                    "max_depth": [3],
                    "subsample": [0.8],
                    "colsample_bytree": [1.0],
                    "gamma": [0],
                    "reg_alpha": [0],
                    "reg_lambda": [1],
                },
            ]
        return [
            {
                "model": ["svc"],
                "kernel": ["poly"],
                "C": [0.01, 0.1, 1, 10],
                "coef0": [0.0, 0.1, 0.02, 0.6, 0.8, 1],
                "gamma": [0.001, 0.002, 0.004, 0.008, 0.01, 0.02, 0.04, 0.08, 0.1, 0.2],
                "degree": [3, 4, 5],
            },
            {
                "model": ["svc"],
                "kernel": ["rbf"],
                "C": [0.01, 0.02, 0.1, 0.02, 1, 2, 10, 20],
                "gamma": [0.001, 0.002, 0.004, 0.008, 0.01, 0.02, 0.04, 0.08, 0.1, 0.2],
            },
            {
                "model": ["rf"],
                "n_estimators": [200, 400, 800],
                "max_depth": [None, 10, 20, 30, 40],
                "max_features": ["sqrt", "log2"],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
            },
            {
                "model": ["xgb"],
                "n_estimators": [300, 400, 600],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [4, 6, 8],
                "subsample": [0.7, 0.85],
                "colsample_bytree": [0.7, 1.0],
                "gamma": [0, 0.01, 1, 5],
                "reg_alpha": [0, 0.5],
                "reg_lambda": [1, 2],
            },
        ]

    def _group_param_grid_by_model(self) -> dict:
        grids: dict = {}
        for entry in self.PARAM_GRID:
            model = entry["model"][0]
            grids.setdefault(model, []).append(
                {k: v for k, v in entry.items() if k != "model"}
            )
        return {m: ParameterGrid(g) for m, g in grids.items()}

    # ------------------------------------------------------------------ #
    # Public accessors
    # ------------------------------------------------------------------ #

    def get_cv_config(self) -> dict:
        return self.CV_CONFIG

    def get_param_grid(self, model: str) -> ParameterGrid:
        return self.PARAM_GRID_BY_MODEL[model]

    def setup(self) -> None:
        """Create output directories and set random seeds.

        Call this once from the pipeline entry point, not in __init__,
        so that Config can be instantiated in tests without filesystem
        side effects.
        """
        for path in self.PATHS.values():
            os.makedirs(path, exist_ok=True)
        random.seed(self.RANDOM_STATE)
        np.random.seed(self.RANDOM_STATE)
