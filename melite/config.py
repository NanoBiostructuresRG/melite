# SPDX-License-Identifier: LGPL-3.0-or-later
"""Configuration loader for MELITE classification and regression workflows.

Reads ``melite/config_default.toml`` as the base configuration.
An optional user-supplied TOML file can override any key via deep merge.

The :class:`Config` object is the single entry point for all runtime
settings. It is designed to be instantiated without filesystem or global RNG
side effects; workflow orchestration performs operational setup separately.
"""

import os
import random
import tomllib
from pathlib import Path

import numpy as np

from melite.model_catalog import MODEL_CATALOG
from melite.optimization_policy import OPTIMIZATION_POLICY

__all__ = ["Config"]

# Path to the default configuration file bundled with the package
_DEFAULT_CONFIG = Path(__file__).parent / "config_default.toml"
_REGRESSION_MODEL_KEYS = frozenset(
    model.key for model in MODEL_CATALOG if model.prediction_task == "regression"
)


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
        datasets or ``label_column`` for CSV datasets. Every entry also contains
        ``task``, which is exactly ``"classification"`` or ``"regression"``.
        CSV classification entries may additionally contain a declared
        ``categorical_features`` list.
        Registered datasets default to classification when ``task`` is omitted;
        the legacy synthesized registry is also classification-only.
    ACTIVE_CLASSIFIERS : list of str
        Classifier keys from ``[classifiers].active`` in configured evaluation
        order. The packaged default is ``["svc", "rf", "xgb"]``.
    ACTIVE_REGRESSORS : list of str
        Validated regressor keys from ``[regressors].active`` in configured
        evaluation order. Supported values are ``"linear"``, ``"poly"``,
        ``"ridge"``, and ``"lasso"``. The packaged default is an empty list;
        at least one regressor is required when any registered dataset has
        ``task = "regression"``.
    CATBOOST_CONFIG : dict
        Validated optional settings from ``[catboost]``. Empty when the section
        or ``boosting_type`` is omitted; otherwise contains ``boosting_type``
        set to exactly ``"Plain"`` or ``"Ordered"``.
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
        registered dataset or active-regressor list, or configures a regression
        dataset without an active regressor. Registered datasets support only
        ``.npz`` files with ``label_path`` and ``.csv`` files with a non-empty
        ``label_column``; ``task`` accepts exactly ``"classification"`` or
        ``"regression"``. ``[catboost].boosting_type`` accepts exactly
        ``"Plain"`` or ``"Ordered"``.

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
        self.ACTIVE_REGRESSORS = self._validate_active_regressors(
            cfg["regressors"]["active"]
        )
        self.CATBOOST_CONFIG = self._validate_catboost_config(cfg.get("catboost"))
        self.DATASETS = self._build_dataset_registry(cfg)
        regression_datasets = [
            dataset_id
            for dataset_id, spec in self.DATASETS.items()
            if spec["task"] == "regression"
        ]
        if regression_datasets and not self.ACTIVE_REGRESSORS:
            raise ValueError(
                "Regression dataset(s) require at least one selected regressor in "
                "[regressors].active: "
                f"{', '.join(regression_datasets)}."
            )

        # Cross-validation
        cv_section = cfg["cv_smoke"] if smoke else cfg["cv"]
        self.CV_CONFIG = {
            "n_splits": cv_section["n_splits"],
            "n_repeats": cv_section["n_repeats"],
            "inner_n_splits": cv_section["inner_n_splits"],
        }

    @staticmethod
    def _validate_active_regressors(active_regressors) -> list[str]:
        if not isinstance(active_regressors, list) or not all(
            isinstance(key, str) for key in active_regressors
        ):
            raise ValueError("[regressors].active must be a list of model keys.")

        unknown = sorted(set(active_regressors) - _REGRESSION_MODEL_KEYS)
        if unknown:
            valid = ", ".join(sorted(_REGRESSION_MODEL_KEYS))
            raise ValueError(
                "Unknown active regressor(s): "
                f"{', '.join(unknown)}. Valid regressor keys are: {valid}."
            )
        return list(active_regressors)

    @staticmethod
    def _validate_catboost_config(catboost_config) -> dict:
        if catboost_config is None:
            return {}
        if not isinstance(catboost_config, dict):
            raise ValueError("[catboost] must be a TOML table.")

        unknown_keys = sorted(set(catboost_config) - {"boosting_type"})
        if unknown_keys:
            raise ValueError(
                "[catboost] contains unsupported key(s): "
                f"{', '.join(unknown_keys)}. The only supported key is "
                "boosting_type."
            )

        if "boosting_type" not in catboost_config:
            return {}
        boosting_type = catboost_config["boosting_type"]
        if not isinstance(boosting_type, str) or boosting_type not in {
            "Plain",
            "Ordered",
        }:
            raise ValueError(
                '[catboost].boosting_type must be exactly "Plain" or "Ordered".'
            )
        return {"boosting_type": boosting_type}

    def _build_dataset_registry(self, cfg: dict) -> dict:
        datasets = cfg.get("datasets")
        if datasets:
            return self._normalize_user_datasets(
                datasets, active_classifiers=self.ACTIVE_CLASSIFIERS
            )
        benchmark = cfg["benchmark"]
        return self._synthesize_legacy_datasets(
            benchmark["reduction_types"], benchmark["levels"]
        )

    @staticmethod
    def _normalize_user_datasets(datasets: dict, active_classifiers) -> dict:
        metadata_keys = ("family", "method", "variant", "level", "description")
        allowed_keys = (
            "path",
            "label_path",
            "label_column",
            "task",
            "categorical_features",
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

            task = entry.get("task", "classification")
            if not isinstance(task, str) or task not in {
                "classification",
                "regression",
            }:
                raise ValueError(
                    f"Dataset '{dataset_id}' task must be exactly "
                    "'classification' or 'regression'."
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

            categorical_spec = {}
            if "categorical_features" in entry:
                categorical_features = entry["categorical_features"]
                if (
                    not isinstance(categorical_features, list)
                    or not categorical_features
                    or not all(
                        isinstance(column, str) and column.strip()
                        for column in categorical_features
                    )
                ):
                    raise ValueError(
                        f"Dataset '{dataset_id}' categorical_features must be a "
                        "non-empty list of non-empty column names."
                    )
                if len(categorical_features) != len(set(categorical_features)):
                    raise ValueError(
                        f"Dataset '{dataset_id}' categorical_features must contain "
                        "unique column names."
                    )
                if suffix != ".csv" or task != "classification":
                    raise ValueError(
                        f"Dataset '{dataset_id}' categorical_features is supported "
                        "only for CSV classification datasets."
                    )
                if label_column in categorical_features:
                    raise ValueError(
                        f"Dataset '{dataset_id}' categorical_features must not "
                        f"contain its label_column '{label_column}'."
                    )
                if active_classifiers != ["catboost"]:
                    raise ValueError(
                        f"Dataset '{dataset_id}' categorical_features requires "
                        '[classifiers].active = ["catboost"].'
                    )
                categorical_spec = {"categorical_features": list(categorical_features)}

            metadata = {
                key: value for key, value in entry.items() if key in metadata_keys
            }
            normalized[dataset_id] = {
                "path": entry["path"],
                **label_spec,
                "task": task,
                **categorical_spec,
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
                    "task": "classification",
                    "metadata": {
                        "family": "dimensionality",
                        "method": reduction_type,
                        "level": level,
                    },
                }
        return datasets

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
