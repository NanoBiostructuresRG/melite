# Changelog

All notable changes to MOSAIC will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
