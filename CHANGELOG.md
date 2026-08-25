# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.5] - 2026-08-25

### Added
- Added registered CSV dataset input using a configured `label_column`.

### Changed
- Migrated the bundled example from separate NPZ and NPY resources to a single
  synthetic numeric CSV dataset.
- Updated the Quick Start and public input documentation to present CSV as the
  recommended registered-dataset path.
- Updated installed-wheel smoke validation to exercise the bundled CSV example
  end to end.
- Updated public discovery keywords to emphasize general tabular and CSV usage
  instead of PCA/UMAP-specific representation terminology.
- Registered dataset configuration now validates supported extensions
  explicitly (`.npz` and `.csv`).
- Unknown fields in `[datasets.*]` are now rejected instead of silently
  ignored. These validation changes may reject configurations that were
  previously accepted accidentally.

---

## [0.2.4] - 2026-08-25

### Added
- Added the packaged `melite example` Quick Start with deterministic synthetic
  numeric tabular data, available directly from an installed distribution and
  verified through installed-wheel smoke testing.
- Added Ruff linting and formatting plus Mypy type checking as permanent CI
  quality gates.
- Added README contract tests covering canonical CLI commands, configuration
  terminology, local links, and version synchronization.
- Added persistent classifier evaluation evidence through `evaluations.csv`,
  with one aggregate row per dataset and active classifier.
- Added `evaluation_folds.csv` with one row per dataset, classifier, and
  outer-CV split, preserving the raw evidence used for classifier selection.
- Added one dataset-level F1-macro evidence figure under `output/figures/`,
  showing the existing outer-CV scores for every evaluated classifier and
  marking the selected classifier without running additional cross-validation.

### Changed
- Stabilized the public Python API around `Config`, `load_datasets`,
  `plot_f1_macro_evidence`, `predict`, and `__version__`; complete evaluation
  remains available through the `melite run` CLI workflow.
- Consolidated the public Quick Start around the packaged example and removed
  the redundant repository-level `examples/` workflow and `environment.yml`;
  `pyproject.toml` is now the dependency source of truth.
- Adopted the public product identity
  **MELITE — Multi-Model Classifier Evaluator**.
- Classifier selection is now based on mean outer-CV F1-macro from the
  evaluation evidence retained for every active classifier.
- Stacking graduated from experimental status to a stable opt-in classifier;
  it remains disabled by default.
- Standalone SVC evaluation now avoids probability fitting, while exported SVC
  artifacts and the SVC base estimator inside stacking retain probability
  support where required.
- Evaluation parallelism now avoids nested estimator parallelism: inner grid
  search or stacking owns parallel execution while contained estimators run
  single-threaded.
- Evaluation and smoke-mode terminology was aligned throughout the package,
  documentation, CLI, and generated reports.
- Renamed `[models]` / `ACTIVE_MODELS` to `[classifiers]` /
  `ACTIVE_CLASSIFIERS`; existing configuration users must rename the
  `[models]` section to `[classifiers]`.
- Renamed the evaluation CSV field `model_name` to `classifier_name` in
  `results.csv`, `evaluations.csv`, and `evaluation_folds.csv`; external CSV
  consumers must update that column name.
- Narrowed the public `Config` surface: legacy-reduction state, hyperparameter-
  grid internals, redundant accessors, and runtime setup are no longer public
  API. `RANDOM_STATE` is now the single public random-seed value, and
  `CV_CONFIG` contains only `n_splits`, `n_repeats`, and `inner_n_splits`.
- Explicit `[cv].random_state` and `[cv_smoke].random_state` settings are now
  rejected with a clear configuration error; the canonical location is
  `[benchmark].random_state`.
- Replaced the former three-metric CV distribution plot API with
  `plot_f1_macro_evidence()`, focused on the outer-CV F1-macro evidence used
  for classifier selection.

### Fixed
- Result and evaluation persistence failures now propagate to callers instead
  of being silently printed and ignored.
- Corrected tunable-classifier evaluation so hyperparameter search occurs
  inside each outer cross-validation split, separating hyperparameter tuning
  from classifier evaluation.
- Aligned stacking export reconstruction with the configured internal
  stratified CV strategy used during training.
- Removed the post-selection cross-validation step from `melite export`.
  Export now reconstructs the selected model, fits it on all available data,
  and saves the final `.pkl` artifact without producing a second evaluation.
- Preserved the role of `results.csv` as the selected-classifier summary while
  separating full evaluation evidence into dedicated output files.

### Notes
- The legacy `[benchmark]` configuration section remains supported for backward
  compatibility.
- Stacking remains opt-in; the default active classifiers are SVC, Random Forest,
  and XGBoost.
- Optuna and MLflow remain outside the scope of v0.2.4.

---

## [0.2.3] - 2026-06-01

### Added
- Added opt-in experimental stacking via the `stack` model key in
  `[models].active`.
- Added sklearn `StackingClassifier` support using `stack_method="predict_proba"`,
  `passthrough=False`, and `LogisticRegression` as the final estimator.
- Added export and prediction coverage for saved stacking `.pkl` artifacts.

### Changed
- Standalone SVC remains a `StandardScaler` -> `SVC` pipeline.
- The SVC base estimator inside stacking is also scaled and uses
  `SVC(probability=True)` so stacking combines probability outputs.
- Random Forest and XGBoost remain unscaled direct estimators because they are
  tree-based models and do not require feature scaling by default.
- Stacking-internal CV uses the configured split count and random state with
  one repeat to satisfy sklearn's out-of-fold prediction requirements.

### Notes
- Stacking is experimental and disabled by default.
- Export remains `.pkl` via `joblib`.
- Optuna and MLflow are intentionally out of scope for v0.2.3.

---

## [0.2.2] - 2026-05-28

### Changed
- SVC is now trained as a `StandardScaler` -> `SVC` sklearn `Pipeline`.
- Scaling is applied only to SVC; `RandomForestClassifier` and
  `XGBClassifier` remain unscaled direct estimators.
- The full SVC search grid now includes a linear kernel with `svc__C` values
  `[0.01, 0.1, 1, 10]`.
- Exported SVC models now preserve the same `StandardScaler` -> `SVC`
  pipeline used during benchmarking.

### Compatibility
- Legacy export compatibility is preserved for older SVC parameter
  dictionaries that use unprefixed keys such as `C` and `kernel` instead of
  pipeline keys such as `svc__C` and `svc__kernel`.

---

## [0.2.1] - 2026-05-27

### Changed
- `[models].active` / `ACTIVE_MODELS` now controls which model families are
  trained during benchmarking.
- `melite export` now uses strict dataset loading for registry-based datasets
  and no longer falls back to `arr.files[0]` when an `.npz` file lacks `X`.
- Added an installed-wheel smoke test that builds the wheel, installs it
  outside the repository checkout, runs a toy `[datasets.toy]` smoke benchmark,
  exports row 0 non-interactively, and verifies generated artifacts.

### Compatibility
- The top-level public API remains unchanged:
  `Config`, `load_datasets`, `plot_cv_distributions`, `predict`, and
  `__version__`.
- Legacy `reduction_type` + `level` export rows remain supported, but
  individual legacy `.npz` files must now contain an explicit `X` array.

---

## [0.2.0] - 2026-05-26

### Added
- Added canonical `[datasets.<dataset_id>]` TOML registry entries for
  user-defined numeric tabular datasets.
- Added strict generalized dataset loading through `load_datasets(config)`.
- Added dataset-aware benchmark result rows with `dataset`, `family`,
  `method`, `variant`, `level`, and `description` fields.
- Added dataset-aware final export naming, such as
  `Model_SVC_morgan_r2_2048.pkl` and `SVC_morgan_r2_2048.png`.

### Changed
- `melite run` now consumes `cfg.DATASETS` as the canonical execution path.
- PCA and UMAP inputs are treated as ordinary dataset registry entries.
- Legacy `[benchmark].reduction_types` and `levels` are normalized into
  dataset entries when `[datasets]` is absent.
- `melite export` prefers the new `dataset` column and falls back to legacy
  `reduction_type` + `level` rows for older CSV files.

### Fixed
- Registered datasets now fail clearly on missing files, missing `X`,
  non-2D or non-numeric `X`, X/y length mismatch, and embedded-y mismatch.

---

## [0.1.11] - 2026-05-26

### Changed
- Prepared documentation and package metadata for the first PyPI publication as
  `melite`.
- Updated release metadata to use final version `0.1.11` instead of development
  or previous release references.
- Clarified that MELITE is tabular at the modeling level: learning algorithms
  consume numeric `X` and `y` arrays that may come from PCA, UMAP, fingerprints,
  descriptors, clinical variables, experimental measurements, industrial
  features, or manually selected numeric features.
- Clarified that the current dataset orchestration still reflects MELITE's
  PCA/UMAP origin and uses concepts such as reduction type and level.
- Documented generalized `[datasets.*]` definitions as a future direction, not
  current behavior.

### Notes
- No functional behavior changes were made to training, model selection,
  cross-validation, export, prediction, dataset loading, or CLI commands.

---

## [0.1.10] - 2026-05-25

### Changed
- Renamed the project identity from MOSAIC to MELITE.
- Renamed the Python import package from `mosaic` to `melite`.
- Renamed the CLI command from `mosaic` to `melite`.
- Renamed the PyPI distribution target from `mosaic-tabular` to `melite`.
- Updated repository, documentation, CI, and publishing metadata for
  `NanoBiostructuresRG/melite`.
- Updated public API imports to remain the same symbols under `melite`.
- Updated result report headers to use MELITE branding and current CLI commands.

### Fixed
- Replaced the duplicated SVC RBF `C = 0.02` grid value with `C = 0.2`.

### Validation
- `python -m pytest tests/ -v --basetemp=.review_pytest_tmp -o cache_dir=.review_pytest_cache` — 82 passed, 1 warning, 0 failed.
- `melite --help`, `melite run --help`, `melite export --help`, and `melite --version` — passed.
- Public API smoke test — `OK: 0.1.10`.
- `python -m build` — `melite-0.1.10.tar.gz` and `.whl` built successfully.
- `python -m twine check dist/*` — PASSED.
- `mkdocs build --strict` — build succeeded with no errors.
- Wheel includes `melite/`, `melite/config_default.toml`, metadata, and license files.
- Sdist inspected; local virtual environments, caches, generated review/docs
  artifacts, build folders, raw/data/output folders, egg-info, `.pkl`, and
  `.joblib` artifacts were not present.

---

## [0.1.9] - 2026-05-25

### Added
- Added dedicated MkDocs pages for installation, quick start, CLI reference,
  configuration, and release notes.
- Added a home-page workflow diagram:
  `X / y -> mosaic run -> results.csv -> mosaic export -> .pkl -> predict()`.
- Added a "MOSAIC does / does not" scope table to clarify project boundaries.

### Changed
- Reworked `docs/index.md` into a concise project overview with links to the
  dedicated documentation pages.
- Updated `mkdocs.yml` navigation to expose the expanded documentation set.
- Updated `README.md` for `dev/v0.1.9`, test-suite status, live documentation
  links, and planned PyPI publication as `mosaic-tabular`.
- Bumped version to `0.1.9` in `version.py` and `CITATION.cff`.

### Validation
- `mkdocs build --strict` — build succeeded with no errors.
- `pytest tests/ -v` — 82 passed, 1 warning, 0 failed.
- `python -m build` — `mosaic_tabular-0.1.9.tar.gz` and `.whl` built successfully.
- `python -m twine check dist/*` — PASSED.
- `mosaic --version` -> `MOSAIC 0.1.9`.
- Wheel and sdist contents inspected; local virtual environments, caches,
  generated review artifacts, build folders, raw/data/output folders,
  egg-info, `.pkl`, and `.joblib` artifacts were not present.

---

## [0.1.8] - 2026-05-25

### Added
- Added `.github/workflows/publish-to-pypi.yml` — manual PyPI publish workflow
  triggered via `workflow_dispatch`. Validates tag format, verifies package
  version matches tag, builds wheel and sdist, checks distributions, and
  publishes via trusted publishing (OIDC, no API token required).
- Added `dist/` to `.gitignore`.

### Changed
- Renamed PyPI package from `mosaic-ml` to `mosaic-tabular` in `pyproject.toml`.
- Updated `description` in `pyproject.toml` to:
  `"Tabular classification benchmarking toolkit for model selection, repeated
  stratified cross-validation, final model export, and artifact-based inference."`
- Updated `docs/api.md` to include `predict` in the public API reference.
- Updated `docs/index.md`: version badge, pre-stable note, four-card panel,
  `predict` tab in Python API examples, `predict` in public API table.
- Updated `docs/stylesheets/extra.css`: added `.ms-grid--four` layout class.
- Updated `README.md`: corrected all references from `mosaic-ml` to
  `mosaic-tabular`, replaced "inference" with "artifact-based inference"
  throughout, updated roadmap and notes.
- Bumped version to `0.1.8` in `version.py` and `CITATION.cff`.

### Validation
- `python -m build` — `mosaic_tabular-0.1.8.tar.gz` and `.whl` built successfully.
- `python -m twine check dist/*` — PASSED.
- `mkdocs build --strict` — build succeeded with no errors.
- `pytest tests/ -v` — 80 passed, 1 warning, 0 failed.
- `mosaic --version` → `MOSAIC 0.1.8`.

---

## [0.1.7] - 2026-05-25

### Added
- Added `mosaic/predict.py` — inference module exposing `predict()` as a stable
  public API symbol. Loads a `.pkl` artifact, returns predictions and class
  probabilities for a new feature matrix.
- Added `predict` to `mosaic/__init__.py` and `__all__`.
- Added `.github/workflows/ci.yml` — CI workflow: tests on Python 3.11 and 3.12,
  public import boundary check, wheel and sdist smoke installs, docs build.
- Added `.github/workflows/docs.yml` — deploys MkDocs site to GitHub Pages on
  push to `main`.
- Added synthetic example dataset under `examples/`:
  - `sample_labels.npy` — 100 binary labels, balanced classes.
  - `sample_PCA70.npz` — 100×37 feature matrix with `X` and `y` keys.
  - `example_config.toml` — TOML config pointing to example files.
  - `generate_sample_data.py` — reproducible generation script (seed=42).
- Added `tests/test_predict.py` — 11 tests for the inference module.
- Added `tests/test_examples.py` — 11 tests for example dataset integrity.
- Added `tmp_model` fixture to `tests/conftest.py` — trained SVC saved as `.pkl`.
- Added `.pytest_tmp` to `.gitignore`.

### Changed
- Updated `tests/test_public_api.py` — added `predict` import and `__all__` tests.
  Total test count: **80 tests**.
- Rewrote `README.md` combining outside-in user perspective with full developer
  documentation.
- Updated `mosaic/__init__.py` public API docstring to include `predict`.
- Bumped version to `0.1.7` in `version.py` and `CITATION.cff`.

### Validation
- `pytest tests/ -v` — 80 passed, 1 warning, 0 failed.
- `mosaic run --smoke --config examples/example_config.toml` — completed successfully.
- `predict()` returns correct shapes for predictions and probabilities.
- CI workflow passes on Python 3.11 and 3.12 (GitHub Actions).
- `mkdocs build --strict` — build succeeded.

---

## [0.1.6] - 2026-05-23

### Added
- Added NumPy-style docstrings to all 10 modules: `__init__.py`, `version.py`,
  `config.py`, `load_dataset.py`, `result_manager.py`, `plot_metrics.py`,
  `model_training.py`, `export_best_model.py`, `main.py`, and `cli.py`.
- Added MkDocs documentation site with `mkdocs-material` theme:
  - `mkdocs.yml` — site configuration with MOSAIC steel-blue + amber palette.
  - `docs/index.md` — home page with hero section, quick start, and Python API examples.
  - `docs/api.md` — API Reference generated from docstrings via `mkdocstrings`.
  - `docs/changelog.md` — changelog page (includes `CHANGELOG.md`).
  - `docs/stylesheets/extra.css` — custom MOSAIC theme styles.

### Changed
- Bumped version to `0.1.6` in `version.py` and `CITATION.cff`.

### Validation
- `pytest tests/ -v` — 57 passed, 1 warning, 0 failed.
- `mkdocs build --strict` — build succeeded with no errors.
- `mkdocs serve` — site renders correctly locally.
- `python -c "help(mosaic.Config)"` — docstring visible.

---

## [0.1.5] - 2026-05-23

### Added
- Defined stable public API in `mosaic/__init__.py`: `Config`, `load_dataset`,
  `ResultManager`, `plot_cv_distributions`, and `__version__` are now importable
  directly from `mosaic`.
- Added `__all__` to all modules: `config.py`, `load_dataset.py`,
  `result_manager.py`, `plot_metrics.py`, `model_training.py`,
  `export_best_model.py`, and `cli.py`.
- Added `tests/test_version.py` — 6 tests for version metadata.
- Added `tests/test_public_api.py` — 7 tests for public API imports and `__all__`.
- Added `tests/test_plot_metrics.py` — 5 tests for PNG output and directory creation.
- Added `tests/test_cli.py` — 8 tests for CLI help and version output.
- Added `pytest` to `environment.yml`.

### Changed
- `mosaic/__init__.py` now exposes the stable public API with explicit imports
  and `__all__`. Users can do `from mosaic import Config, load_dataset`.
- Bumped version to `0.1.5` in `version.py` and `CITATION.cff`.

### Validation
- `pytest tests/ -v` — 57 passed, 1 warning (expected Agg backend warning), 0 failed.
- `from mosaic import Config, load_dataset, ResultManager, plot_cv_distributions, __version__` ✓
- `mosaic.__all__` contains exactly the 5 expected public symbols ✓
- CLI help and version tests pass via subprocess ✓

---

## [0.1.4] - 2026-05-22

### Added
- Added formal `pytest` suite under `tests/` with 31 tests covering
  `Config`, `load_dataset`, `ResultManager`, and `Finalizer`.
- Added `tests/conftest.py` with shared synthetic fixtures: labels, valid/invalid
  `.npz` files, minimal `results.csv`, and a base `Config` instance.
- Added `tests_output.txt` to `.gitignore`.

### Fixed
- Removed redundant `print()` progress calls from `main.py`; all progress now
  goes through `logger.*` only. Smoke banner remains as `print()` (user-facing UI).
- Changed smoke-mode `logger.warning()` to `logger.info()` in `main.py` to avoid
  spurious output when `--verbose` is not passed.
- Removed `logging.basicConfig(level=logging.INFO)` from `load_dataset.py` —
  module-level `basicConfig` was overriding the package logger configuration.
- Improved error message for missing `.npz` file: now includes full expected path
  and an actionable hint.
- Improved error message for missing `X` key in `.npz`: now raises `ValueError`
  with filename and list of available keys.
- Improved label mismatch error: now includes both array shapes and differing
  element count.
- Added guard in `Finalizer.__init__` for missing `results.csv`: raises
  `FileNotFoundError` with path and actionable hint before attempting to read.
- Bumped version to `0.1.4` in `version.py` and `CITATION.cff`.

### Validation
- `pytest tests/ -v` — 31 passed, 0 failed.
- `mosaic run --smoke` — silent output (smoke banner only, no logger noise).
- `mosaic run --smoke --verbose` — clean INFO logs, no duplicates.
- `mosaic export --row 0` (missing CSV) — descriptive `FileNotFoundError`.

---

## [0.1.3] - 2026-05-22

### Added
- Added `pyproject.toml` with `hatchling` build backend and `[project.scripts]`
  entry point: `mosaic = "mosaic.cli:main"`. Package is now installable via
  `pip install -e .`.
- Added unified CLI `mosaic/cli.py` with subcommands `mosaic run` and
  `mosaic export`, replacing the two separate entry points.
- Added global CLI flags: `--verbose` (INFO-level logging), `--config PATH`
  (user TOML override), `--version`.
- Added `mosaic/config_default.toml` with externalized configuration: paths,
  reduction types, levels, random state, CV settings, and active models.
- Added `Config.setup()` method to separate filesystem side effects from object
  instantiation, enabling safe use in tests.
- Added `ResultManager.write_csv()` method, consolidating all file I/O under
  `ResultManager`.
- Added `smoke` column to `results.csv` to mark runs generated in smoke mode.
- Added smoke-mode export guard: `mosaic export` blocks export of smoke results
  with exit code 1. Use `mosaic export --force` to override with a visible
  warning.
- Added package-level `NullHandler` logger in `mosaic/__init__.py`.

### Changed
- `Config` now reads `config_default.toml` as the base configuration.
- `Config.__init__` no longer creates directories or sets random seeds.
- `Config.PARAM_GRID` is now built by `_build_param_grid()`.
- `main.py` now iterates over `config.REDUCTION_TYPES` instead of a hardcoded list.
- `main.py` delegates CSV writing to `ResultManager.write_csv()`.
- CLI logic centralized in `mosaic/cli.py`.
- Bumped version to `0.1.3` in `version.py` and `CITATION.cff`.

### Validation
- `pip install -e .` succeeded in `mosaic_env`.
- `mosaic --help`, `mosaic run --help`, `mosaic export --help` verified.
- `mosaic --version` returns `MOSAIC 0.1.3`.
- `mosaic run --smoke --verbose` completed full PCA + UMAP run with INFO logging.
- `mosaic export --row 0 --force` exported with warning, `.pkl` artifact created.

---

## [0.1.2] - 2026-05-22

### Changed
- Moved all source modules into a `mosaic/` package directory.
- Updated all intra-package imports to use the `mosaic.*` namespace.
- `result_manager.py` now reads `__version__` from `mosaic.version`.
- Updated `README.md` to reflect the `mosaic/` package structure.

### Fixed
- Bumped version string in `version.py` to `0.1.2`.

### Added
- Added `--smoke` flag to `mosaic.main` for lightweight benchmarking.
- Added `argparse` CLI to `mosaic.main` with `--smoke` flag and `--help` support.

### Validation
- Python syntax validation passed for all modules under `mosaic/`.
- Dataset loading smoke test passed: PCA70 labels and features loaded correctly.
- SVC, Random Forest and XGBoost smoke tests passed; scores match v0.1.1 baseline.
- PNG figure generation smoke test passed.

---

## [0.1.1] - 2025-05-22

### Changed
- Rewrote `README.md` to reflect the current pre-stable repository workflow.
- Changed project license from MIT to GNU LGPL v3.0 or later.
- Added GNU license files: `COPYING` and `COPYING.LESSER`.
- Added SPDX license metadata: `LGPL-3.0-or-later`.

### Fixed
- Added validation to ensure embedded `.npz` labels match `raw/labels.npy`.
- Ensured plot output directories are created automatically before saving figures.

### Added
- Added `version.py` with centralized project version metadata.
- Added `CITATION.cff` for software citation metadata.
- Added non-interactive model export support with `python export_best_model.py --row <index>`.

### Validation
- Python syntax validation passed.
- PCA70 dataset loading passed.
- Export CLI help validation passed.
- Minimal SVC, Random Forest and XGBoost smoke tests passed in `mosaic_env`.

---

## [0.1.0] - 2025-05-22

### Added
- Initial corrected pre-stable version of MOSAIC.
- Tabular classification benchmarking with PCA/UMAP reduced datasets.
- Grid search support for SVC, Random Forest and XGBoost.
- Repeated Stratified K-Fold evaluation.
- TXT/CSV result export.
- Final model export as `.pkl`.
- Three-panel metric plotting for F1, Accuracy and AUC-ROC.

---

[0.2.5]: https://github.com/NanoBiostructuresRG/melite/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/NanoBiostructuresRG/melite/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/NanoBiostructuresRG/melite/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/NanoBiostructuresRG/melite/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/NanoBiostructuresRG/melite/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/NanoBiostructuresRG/melite/compare/v0.1.11...v0.2.0
[0.1.11]: https://github.com/NanoBiostructuresRG/melite/compare/v0.1.10...v0.1.11
[0.1.10]: https://github.com/NanoBiostructuresRG/melite/compare/v0.1.9...v0.1.10
[0.1.9]: https://github.com/NanoBiostructuresRG/melite/compare/v0.1.8...v0.1.9
[0.1.8]: https://github.com/NanoBiostructuresRG/melite/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/NanoBiostructuresRG/melite/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/NanoBiostructuresRG/melite/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/NanoBiostructuresRG/melite/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/NanoBiostructuresRG/melite/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/NanoBiostructuresRG/melite/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/NanoBiostructuresRG/melite/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/NanoBiostructuresRG/melite/releases/tag/v0.1.1
[0.1.0]: https://github.com/NanoBiostructuresRG/melite/releases/tag/v0.1.0
