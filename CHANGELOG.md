# Changelog

All notable changes to MOSAIC will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.2] - 2026-05-22

### Changed
- Moved all source modules into a `mosaic/` package directory.
- Updated all intra-package imports to use the `mosaic.*` namespace.
- `result_manager.py` now reads `__version__` from `mosaic.version` instead of
  using a hardcoded string.
- Updated `README.md` to reflect the `mosaic/` package structure, corrected CLI
  commands, documented `--smoke` mode, and updated validation section.

### Fixed
- Bumped version string in `version.py` to `0.1.2`.

### Added
- Added `--smoke` flag to `mosaic.main` for lightweight benchmarking with
  single-value hyperparameter grids and 3-fold CV (no repeats).
- Added `argparse` CLI to `mosaic.main` with `--smoke` flag and `--help` support.

### Validation
- Python syntax validation passed for all modules under `mosaic/`.
- Dataset loading smoke test passed: PCA70 labels and features loaded correctly.
- SVC, Random Forest and XGBoost smoke tests passed; scores match v0.1.1 baseline.
- PNG figure generation smoke test passed.
- `--smoke` mode completed full PCA + UMAP benchmark run successfully.

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
