# Changelog

All notable changes to MOSAIC will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [dev/0.1.1] - Unreleased

### Changed
- Changed project license from MIT to GNU LGPL v3.0 or later.
- Added GNU license files: `COPYING` and `COPYING.LESSER`.
- Added SPDX license metadata: `LGPL-3.0-or-later`.

### Fixed
- Added validation to ensure embedded `.npz` labels match `raw/labels.npy`.
- Ensured plot output directories are created automatically before saving figures.

### Added
- Added non-interactive model export support with `python export_best_model.py --row <index>`.

### Validation
- Python syntax validation passed.
- PCA70 dataset loading passed.
- Export CLI help validation passed.
- Minimal SVC, Random Forest and XGBoost smoke tests passed in `mosaic_env`.

## [0.1.0] - 2025-05-22

### Added
- Initial corrected pre-stable version of MOSAIC.
- Tabular classification benchmarking with PCA/UMAP reduced datasets.
- Grid search support for SVC, Random Forest and XGBoost.
- Repeated Stratified K-Fold evaluation.
- TXT/CSV result export.
- Final model export as `.pkl`.
- Three-panel metric plotting for F1, Accuracy and AUC-ROC.