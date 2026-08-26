# SPDX-License-Identifier: LGPL-3.0-or-later
"""Configuration loader for MELITE.

Reads ``melite/config_default.toml`` as the base configuration.
An optional user-supplied TOML file can override any key via deep merge.
Hyperparameter grids are internal implementation details and are not part of
the public :class:`Config` API.

The :class:`Config` object is the single entry point for all runtime
settings. It is designed to be instantiated without filesystem or global RNG
side effects; workflow orchestration performs operational setup separately.
"""

import os
import random
import tomllib
from pathlib import Path

import numpy as np

from melite.optimization_policy import OPTIMIZATION_POLICY

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
    """Configuration container for loading, merging, normalizing, and inspecting
    MELITE runtime settings.

    Loads defaults from ``melite/config_default.toml``. If *user_config* is
    provided, its values are merged over the defaults — user values win and
    missing keys fall back to defaults.

    Parameters
    ----------
    smoke : bool, optional
        Whether to use reduced CV/search settings for lightweight execution
        checks. Default is ``False``.
    user_config : pathlib.Path or None, optional
        Optional TOML configuration file merged over packaged defaults.
        Default is ``None``.

    Attributes
    ----------
    SMOKE : bool
        Whether the instance was created in smoke mode.
    PATHS : dict
        Dictionary with keys ``"INPUT"``, ``"DATASET"``, and ``"OUTPUT"``
        mapping to the corresponding directory paths as strings.
    RESULTS_FILE : str
        Path to the TXT results file (``output/results.txt`` by default).
    RANDOM_STATE : int
        Canonical global random seed used by MELITE runtime and evaluation
        components. Default is ``42``.
    N_TRIALS : int
        Optimization trial budget. Normal mode uses the effective
        ``[optimization].n_trials`` value, which defaults to ``100``. Smoke
        mode always uses MELITE's internal, non-configurable budget of ``5``.
    DATASETS : dict
        Normalized dataset registry keyed by user-defined dataset id. Each
        entry contains ``path`` and ``metadata`` plus ``label_path`` for NPZ
        datasets or ``label_column`` for CSV datasets.
    ACTIVE_CLASSIFIERS : list of str
        Classifier keys to include in the evaluation (e.g. ``["svc", "rf", "xgb"]``;
        add ``"stack"`` to opt in to stacking).
    CV_CONFIG : dict
        Cross-validation settings with keys ``n_splits``, ``n_repeats``, and
        ``inner_n_splits``.

    Raises
    ------
    FileNotFoundError
        If the supplied user configuration file does not exist.
    ValueError
        If a user configuration uses the obsolete ``[models]`` section,
        defines the unsupported ``[optimization_smoke]`` section,
        specifies ``random_state`` under ``[cv]`` or ``[cv_smoke]`` instead of
        ``[benchmark]``, contains an unsupported ``[optimization]`` key, sets
        an invalid ``[optimization].n_trials`` value, or defines an invalid
        registered dataset. Registered datasets support only ``.npz`` files
        with ``label_path`` and ``.csv`` files with a non-empty
        ``label_column``.

    Examples
    --------
    Default configuration:

    >>> from melite import Config
    >>> cfg = Config()
    >>> cfg.RANDOM_STATE
    42

    Smoke mode:

    >>> cfg = Config(smoke=True)
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
            if "models" in user_cfg:
                raise ValueError(
                    "The [models] configuration section was renamed to "
                    "[classifiers] in MELITE v0.2.4. Rename [models] to "
                    "[classifiers] in your configuration file."
                )
            if "optimization_smoke" in user_cfg:
                raise ValueError(
                    "[optimization_smoke] is unsupported. The smoke optimization "
                    "budget is internal, non-configurable MELITE policy."
                )
            for section in ("cv", "cv_smoke"):
                if "random_state" in user_cfg.get(section, {}):
                    raise ValueError(
                        f"[{section}].random_state is not supported. Configure "
                        "the canonical random seed through "
                        "[benchmark].random_state."
                    )
            optimization_cfg = user_cfg.get("optimization")
            if optimization_cfg is not None:
                if not isinstance(optimization_cfg, dict):
                    raise ValueError("[optimization] must be a TOML table.")
                unknown_keys = sorted(set(optimization_cfg) - {"n_trials"})
                if unknown_keys:
                    raise ValueError(
                        "[optimization] contains unsupported key(s): "
                        f"{', '.join(unknown_keys)}. The only supported key in "
                        "MELITE v0.3.0 is n_trials."
                    )
            cfg = _deep_merge(cfg, user_cfg)

        # Paths
        self.PATHS = {
            "INPUT": cfg["paths"]["input"],
            "DATASET": cfg["paths"]["dataset"],
            "OUTPUT": cfg["paths"]["output"],
        }
        self.RESULTS_FILE = os.path.join(self.PATHS["OUTPUT"], "results.txt")

        # Evaluation settings
        self.RANDOM_STATE = cfg["benchmark"]["random_state"]
        normal_n_trials = cfg["optimization"]["n_trials"]
        if (
            isinstance(normal_n_trials, bool)
            or not isinstance(normal_n_trials, int)
            or normal_n_trials < 1
        ):
            raise ValueError(
                "[optimization].n_trials must be an integer greater than or equal to 1."
            )
        self.N_TRIALS = OPTIMIZATION_POLICY.smoke_n_trials if smoke else normal_n_trials
        self.ACTIVE_CLASSIFIERS = cfg["classifiers"]["active"]
        self.DATASETS = self._build_dataset_registry(cfg)

        # Cross-validation
        cv_section = cfg["cv_smoke"] if smoke else cfg["cv"]
        self.CV_CONFIG = {
            "n_splits": cv_section["n_splits"],
            "n_repeats": cv_section["n_repeats"],
            "inner_n_splits": cv_section["inner_n_splits"],
        }

        # Hyperparameter grids — developer-facing, defined in Python
        self._param_grid = self._build_param_grid()

    # ------------------------------------------------------------------ #
    # Hyperparameter grids
    # ------------------------------------------------------------------ #

    def _build_dataset_registry(self, cfg: dict) -> dict:
        datasets = cfg.get("datasets")
        if datasets:
            return self._normalize_user_datasets(datasets)
        benchmark = cfg["benchmark"]
        return self._synthesize_legacy_datasets(
            benchmark["reduction_types"], benchmark["levels"]
        )

    @staticmethod
    def _normalize_user_datasets(datasets: dict) -> dict:
        metadata_keys = ("family", "method", "variant", "level", "description")
        allowed_keys = (
            "path",
            "label_path",
            "label_column",
            *metadata_keys,
        )
        normalized = {}
        for dataset_id, entry in datasets.items():
            unknown_keys = sorted(set(entry) - set(allowed_keys))
            if unknown_keys:
                raise ValueError(
                    f"Dataset '{dataset_id}' contains unknown field(s): "
                    f"{', '.join(unknown_keys)}. Allowed fields: "
                    f"{', '.join(allowed_keys)}."
                )

            if "path" not in entry:
                raise ValueError(
                    f"Dataset '{dataset_id}' is missing required field: path."
                )

            has_label_path = "label_path" in entry
            has_label_column = "label_column" in entry
            if has_label_path and has_label_column:
                raise ValueError(
                    f"Dataset '{dataset_id}' specifies both label_path and "
                    "label_column; configure exactly one for its registered format."
                )

            suffix = Path(entry["path"]).suffix.lower()
            if suffix not in {".npz", ".csv"}:
                raise ValueError(
                    f"Dataset '{dataset_id}' has unsupported extension "
                    f"'{suffix or '<none>'}'. .npz and .csv are the supported "
                    "registered-dataset formats."
                )

            if suffix == ".npz":
                if has_label_column:
                    raise ValueError(
                        f"Dataset '{dataset_id}' is an NPZ dataset; label_column "
                        "is forbidden. Configure label_path instead."
                    )
                if not has_label_path:
                    raise ValueError(
                        f"Dataset '{dataset_id}' is missing required field: label_path."
                    )
                label_spec = {"label_path": entry["label_path"]}
            else:
                if has_label_path:
                    raise ValueError(
                        f"Dataset '{dataset_id}' is a CSV dataset; label_path is "
                        "forbidden. Configure label_column instead."
                    )
                if not has_label_column:
                    raise ValueError(
                        f"Dataset '{dataset_id}' is missing required field: "
                        "label_column."
                    )
                label_column = entry["label_column"]
                if not isinstance(label_column, str) or not label_column.strip():
                    raise ValueError(
                        f"Dataset '{dataset_id}' label_column must be a non-empty "
                        "string."
                    )
                label_spec = {"label_column": label_column}

            metadata = {
                key: value for key, value in entry.items() if key in metadata_keys
            }
            normalized[dataset_id] = {
                "path": entry["path"],
                **label_spec,
                "metadata": metadata,
            }
        return normalized

    def _synthesize_legacy_datasets(self, reduction_types, levels) -> dict:
        datasets = {}
        for reduction_type in reduction_types:
            for level in levels:
                dataset_id = f"{reduction_type}{level}"
                datasets[dataset_id] = {
                    "path": os.path.join(self.PATHS["DATASET"], f"{dataset_id}.npz"),
                    "label_path": os.path.join(self.PATHS["INPUT"], "labels.npy"),
                    "metadata": {
                        "family": "dimensionality",
                        "method": reduction_type,
                        "level": level,
                    },
                }
        return datasets

    def _build_param_grid(self) -> list:
        if self.SMOKE:
            return [
                {
                    "model": ["svc"],
                    "svc__kernel": ["linear"],
                    "svc__C": [1],
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
                {
                    "model": ["stack"],
                },
            ]
        return [
            {
                "model": ["svc"],
                "svc__kernel": ["linear"],
                "svc__C": [0.01, 0.1, 1, 10],
            },
            {
                "model": ["svc"],
                "svc__kernel": ["poly"],
                "svc__C": [0.01, 0.1, 1, 10],
                "svc__coef0": [0.0, 0.1, 0.2, 0.6, 0.8, 1],
                "svc__gamma": [
                    0.001,
                    0.002,
                    0.004,
                    0.008,
                    0.01,
                    0.02,
                    0.04,
                    0.08,
                    0.1,
                    0.2,
                ],
                "svc__degree": [3, 4, 5],
            },
            {
                "model": ["svc"],
                "svc__kernel": ["rbf"],
                "svc__C": [0.01, 0.02, 0.1, 0.2, 1, 2, 10, 20],
                "svc__gamma": [
                    0.001,
                    0.002,
                    0.004,
                    0.008,
                    0.01,
                    0.02,
                    0.04,
                    0.08,
                    0.1,
                    0.2,
                ],
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
            {
                "model": ["stack"],
            },
        ]

    def _setup(self) -> None:
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
