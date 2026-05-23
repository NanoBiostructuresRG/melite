# SPDX-License-Identifier: LGPL-3.0-or-later
"""Configuration loader for MOSAIC.

Reads ``mosaic/config_default.toml`` as the base configuration.
An optional user-supplied TOML file can override any key via deep merge.
Hyperparameter grids are defined here in Python — they are developer-facing
and not expected to change between runs.

The :class:`Config` object is the single entry point for all runtime
settings. It is designed to be instantiated without filesystem side effects;
call :meth:`Config.setup` explicitly from pipeline entry points.
"""

import os
import random
import tomllib
from pathlib import Path

import numpy as np
from sklearn.model_selection import ParameterGrid

__all__ = ["Config"]

# Path to the default configuration file bundled with the package
_DEFAULT_CONFIG = Path(__file__).parent / "config_default.toml"


def _load_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge *override* into *base* recursively. Override wins on conflict."""
    result = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class Config:
    """Configuration container for MOSAIC.

    Loads defaults from ``mosaic/config_default.toml``. If *user_config* is
    provided, its values are merged over the defaults — user values win and
    missing keys fall back to defaults.

    Parameters
    ----------
    smoke : bool, optional
        If ``True``, use reduced CV settings and single-value hyperparameter
        grids for lightweight runs. Default is ``False``.
    user_config : pathlib.Path or None, optional
        Path to a user-supplied TOML file. Only the keys present in this file
        override the defaults. Default is ``None``.

    Attributes
    ----------
    SMOKE : bool
        Whether the instance was created in smoke mode.
    PATHS : dict
        Dictionary with keys ``"INPUT"``, ``"DATASET"``, and ``"OUTPUT"``
        mapping to the corresponding directory paths as strings.
    RESULTS_FILE : str
        Full path to the TXT results file (``output/results.txt`` by default).
    RANDOM_STATE : int
        Global random seed. Default is ``42``.
    REDUCTION_TYPES : list of str
        Reduction methods to benchmark (e.g. ``["PCA", "UMAP"]``).
    REDUCTION_LEVELS : list of int
        Variance retention levels to benchmark (e.g. ``[70, 75, 80, 85, 90, 95]``).
    ACTIVE_MODELS : list of str
        Model keys to include in the benchmark (e.g. ``["svc", "rf", "xgb"]``).
    CV_CONFIG : dict
        Cross-validation settings with keys ``n_splits``, ``n_repeats``, and
        ``random_state``.
    PARAM_GRID : list of dict
        Raw hyperparameter grid definitions, one entry per model configuration.
    PARAM_GRID_BY_MODEL : dict
        Compiled :class:`~sklearn.model_selection.ParameterGrid` objects keyed
        by model name (``"svc"``, ``"rf"``, ``"xgb"``).

    Examples
    --------
    Default configuration:

    >>> cfg = Config()
    >>> cfg.RANDOM_STATE
    42

    Smoke mode with a user override:

    >>> cfg = Config(smoke=True, user_config=Path("my_config.toml"))
    >>> cfg.CV_CONFIG["n_splits"]
    3
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
        """Return the cross-validation configuration dictionary.

        Returns
        -------
        dict
            Dictionary with keys ``n_splits``, ``n_repeats``, and
            ``random_state``.
        """
        return self.CV_CONFIG

    def get_param_grid(self, model: str) -> ParameterGrid:
        """Return the compiled hyperparameter grid for a given model.

        Parameters
        ----------
        model : str
            Model key. One of ``"svc"``, ``"rf"``, or ``"xgb"``.

        Returns
        -------
        sklearn.model_selection.ParameterGrid
            Iterable of hyperparameter combinations for the requested model.
        """
        return self.PARAM_GRID_BY_MODEL[model]

    def setup(self) -> None:
        """Create output directories and set random seeds.

        This method must be called once from the pipeline entry point before
        any data is loaded or models are trained. It is intentionally separated
        from ``__init__`` so that :class:`Config` can be instantiated in tests
        without creating directories or modifying global random state.

        Notes
        -----
        Directories are created with ``exist_ok=True``, so calling ``setup``
        multiple times is safe.
        """
        for path in self.PATHS.values():
            os.makedirs(path, exist_ok=True)
        random.seed(self.RANDOM_STATE)
        np.random.seed(self.RANDOM_STATE)
