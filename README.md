# MOSAIC: Modular Multi-Model Selection and Cross-Validation

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL_v3-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-v0.1.5-blue.svg)]()

---

## Description

**MOSAIC** is a lightweight benchmarking toolkit for tabular classification.

It evaluates machine learning workflows based on PCA- or UMAP-reduced feature
matrices, performs grid search over SVC, Random Forest and XGBoost classifiers,
evaluates model configurations with repeated stratified cross-validation,
exports TXT/CSV performance summaries, supports final model retraining on all
available data, saves deployable `.pkl` model artifacts, and generates
three-panel metric plots for F1, Accuracy and AUC-ROC.

---

## Development Status

MOSAIC is currently in **pre-stable development**.

The current development version is:

```text
0.1.5
```

The active development branch is:

```text
dev/v0.1.5
```

MOSAIC is not yet published on PyPI. It can be installed in editable mode
directly from the repository.

---

## Purpose

The primary objective of **MOSAIC** is to provide a reproducible benchmarking
workflow for supervised tabular classification.

MOSAIC is especially useful when feature matrices have already been generated
by an upstream workflow, for example molecular fingerprints, PCA-reduced
representations, or UMAP-reduced representations.

The toolkit enables:

- Standardized comparison of tabular classification models.
- Repeated stratified cross-validation for robust performance estimation.
- Grid-search model selection across SVC, Random Forest and XGBoost.
- Export of benchmarking results as human-readable TXT and structured CSV.
- Final retraining of a selected model on the full available dataset.
- Serialization of trained models as `.pkl` artifacts.
- Visualization of cross-validation fold distributions for F1, Accuracy and
  AUC-ROC.

---

## Current Scope

MOSAIC currently assumes that feature matrices and labels have already been
created before execution.

It does **not** currently generate PCA or UMAP reductions from raw molecular
data. Instead, it consumes local `.npy` and `.npz` files prepared by an upstream
step.

Expected local inputs:

```text
raw/labels.npy
data/PCA70.npz
data/PCA75.npz
data/PCA80.npz
data/PCA85.npz
data/PCA90.npz
data/PCA95.npz
data/UMAP70.npz
data/UMAP75.npz
data/UMAP80.npz
data/UMAP85.npz
data/UMAP90.npz
data/UMAP95.npz
```

Each `.npz` file must contain an `X` array.

If an embedded `y` array is present inside the `.npz` file, MOSAIC validates it
against `raw/labels.npy` to avoid silent feature-label mismatches.

---

## Installation for Development

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate mosaic_env
```

Install MOSAIC in editable mode:

```bash
pip install -e .
```

Install development dependencies (includes pytest):

```bash
pip install -e ".[dev]"
```

Verify the main dependencies:

```bash
python -c "import numpy, pandas, sklearn, xgboost, joblib, matplotlib; print('dependencies OK')"
```

---

## Public API

MOSAIC exposes a stable public API for use as a Python library:

```python
from mosaic import Config
from mosaic import load_dataset
from mosaic import ResultManager
from mosaic import plot_cv_distributions
from mosaic import __version__
```

Modules not listed above are importable directly but are not part of the
stable API and may change between versions.

---

## Quick Start

### 1. Run the full benchmarking phase

```bash
mosaic run
```

This runs the configured benchmarking workflow and writes:

```text
output/results.txt
output/results.csv
```

### 2. Run a lightweight smoke test

```bash
mosaic run --smoke
```

Uses single-value hyperparameter grids and 3-fold CV (no repeats) for fast
validation. Results are not benchmark-quality and are marked in `results.csv`.

### 3. Export a selected model interactively

```bash
mosaic export
```

MOSAIC will display the available rows from `output/results.csv` and ask which
row should be exported.

### 4. Export a selected model non-interactively

```bash
mosaic export --row 0
```

This selects row `0` from `output/results.csv`, retrains the corresponding
model on all available data, saves a `.pkl` artifact, and generates a metric
plot.

### 5. Export with verbose logging

```bash
mosaic run --verbose
mosaic export --row 0 --verbose
```

### 6. Override smoke-mode export guard

```bash
mosaic export --row 0 --force
```

Smoke-mode results are blocked from export by default. Use `--force` to
override with a visible warning.

### 7. Use a custom configuration file

```bash
mosaic run --config my_config.toml
```

Only the keys present in `my_config.toml` override the defaults. All other
settings fall back to `mosaic/config_default.toml`.

### 8. Run the test suite

```bash
pytest tests/ -v
```

---

## Input Format

### Labels

```text
raw/labels.npy
```

This file must contain the target vector `y`.

Example shape:

```text
(182,)
```

### Reduced Feature Matrices

Each reduced dataset must be stored as an `.npz` file in `data/`.

Example:

```text
data/PCA70.npz
```

Required key:

```text
X
```

Optional key:

```text
y
```

If `y` is present, MOSAIC checks that it matches `raw/labels.npy`.

Expected array relationship:

```text
X.shape = (n_samples, n_features)
y.shape = (n_samples,)
```

The number of rows in `X` must match the number of labels in `y`.

---

## Configuration

MOSAIC reads its default configuration from `mosaic/config_default.toml`.
To customize paths, reduction levels, CV settings, or active models, create
a TOML file and pass it with `--config`:

```toml
[paths]
output = "my_output/"

[benchmark]
levels = [70, 85, 95]

[models]
active = ["svc", "rf"]
```

```bash
mosaic run --config my_config.toml
```

Hyperparameter grids are defined in `mosaic/config.py` and are intended for
developer-level customization only.

---

## Outputs

MOSAIC writes outputs under the local `output/` directory.

```text
output/
├── results.txt
├── results.csv
├── Model_<model>_<reduction><level>.pkl
└── figures/
    └── <model>_<reduction><level>.png
```

### `results.txt`

Human-readable report summarizing the selected model and metrics for each
configuration.

### `results.csv`

Structured table containing model performance, selected hyperparameters, and
a `smoke` column indicating whether the run was generated in smoke mode.

### `.pkl` Model Artifact

Final model retrained on all available data for the selected configuration.

### PNG Figure

Three-panel plot showing cross-validation fold distributions for:

- F1
- Accuracy
- AUC-ROC

---

## Project Structure

```text
MOSAIC/
├── mosaic/                       # Python package
│   ├── __init__.py               # Public API
│   ├── cli.py                    # Unified CLI entry point (mosaic run / mosaic export)
│   ├── config.py                 # Configuration loader and hyperparameter grids
│   ├── config_default.toml       # Default user-facing configuration
│   ├── load_dataset.py           # Dataset loading and label consistency validation
│   ├── model_training.py         # GridSearchCV, repeated CV and model selection
│   ├── main.py                   # Main benchmarking pipeline
│   ├── result_manager.py         # TXT and CSV result writer
│   ├── export_best_model.py      # Final model export workflow
│   ├── plot_metrics.py           # CV metric distribution plots
│   └── version.py                # Package version metadata
│
├── tests/                        # pytest suite (57 tests)
│   ├── conftest.py               # Shared synthetic fixtures
│   ├── test_config.py
│   ├── test_load_dataset.py
│   ├── test_result_manager.py
│   ├── test_export.py
│   ├── test_version.py
│   ├── test_public_api.py
│   ├── test_plot_metrics.py
│   └── test_cli.py
│
├── raw/                          # Local input labels and upstream feature data; ignored by Git
├── data/                         # Local PCA/UMAP reduced matrices; ignored by Git
├── output/                       # Local generated reports, figures and models; ignored by Git
│
├── pyproject.toml                # Package metadata and build system
├── environment.yml               # Conda development environment
├── CHANGELOG.md                  # Version history
├── CITATION.cff                  # Citation metadata
├── COPYING                       # GNU GPL v3 license text
├── COPYING.LESSER                # GNU LGPL v3 license text
├── LICENSE                       # Project license summary
└── README.md
```

---

## Models

MOSAIC currently benchmarks:

- Support Vector Classifier (`SVC`)
- Random Forest Classifier
- XGBoost Classifier

The configured hyperparameter grids are defined in `mosaic/config.py`.
Active models can be set in `config_default.toml` or a user config file.

---

## Cross-Validation

MOSAIC uses repeated stratified K-fold cross-validation.

The default configuration in `mosaic/config_default.toml`:

```toml
[cv]
n_splits = 10
n_repeats = 5
```

This gives:

```text
N × M = 10 × 5 = 50 validation folds
```

per evaluated configuration.

---

## Example Console Output

```text
# Full benchmarking phase — verbose
$ mosaic run --verbose
INFO:mosaic.main:Running with PCA...
INFO:mosaic.load_dataset:Labels loaded: raw/labels.npy (shape=(182,))
INFO:mosaic.main:Training with PCA85 (level=85).
INFO:mosaic.main:Final report written to output/results.txt
INFO:mosaic.main:CSV file written to output/results.csv

# Smoke test
$ mosaic run --smoke
[SMOKE TEST] Using reduced grid and CV. Results are not benchmark-quality.

# Export phase — interactive
$ mosaic export
  reduction_type  level              model_name  f1_macro  accuracy  auc_roc
0            PCA     85                     SVC    0.8336    0.8408   0.8802
1           UMAP     85  RandomForestClassifier    0.7041    0.7097   0.7855

Enter the row number to keep: 0

Training SVC on PCA85 using all available data...

# Export phase — non-interactive
$ mosaic export --row 0

# Smoke guard
$ mosaic export --row 0
[ERROR] This result was generated in smoke mode and is not benchmark-quality.
        Run 'mosaic run' (without --smoke) to generate valid results,
        or use 'mosaic export --force' to override this guard.

# Test suite
$ pytest tests/ -v
57 passed, 1 warning in 58.31s
```

---

## Notes

- Full benchmarking can be computationally expensive because MOSAIC performs
  grid search and repeated cross-validation. Use `mosaic run --smoke` for fast
  validation during development.
- `output/`, `data/`, and `raw/` are local working directories and are ignored
  by Git.
- Model artifacts such as `.pkl` and `.joblib` files are ignored by Git.
- PyPI publishing is planned for a future development phase.

---

## Validation

The current `dev/v0.1.5` branch has been validated with:

```bash
pytest tests/ -v
```

57 tests passed covering `Config`, `load_dataset`, `ResultManager`, `Finalizer`,
`plot_metrics`, `version`, public API, and CLI.

CLI help smoke tests:

```bash
mosaic --help
mosaic run --help
mosaic export --help
mosaic --version
```

Public API smoke test:

```bash
python -c "from mosaic import Config, load_dataset, ResultManager, plot_cv_distributions, __version__; print('OK:', __version__)"
```

---

## Roadmap

Near-term development goals:

- ~~Add formal tests with `pytest`.~~ ✓ Done in v0.1.4
- ~~Define stable public API.~~ ✓ Done in v0.1.5
- Publish to PyPI as `mosaic-ml`.
- Add continuous integration.
- Add documented example datasets.
- Add a prediction/inference module for exported `.pkl` artifacts.

---

## Citation

If you use MOSAIC in your research, please cite it using the metadata in
[CITATION.cff](CITATION.cff).

Suggested citation format:

```text
Contreras-Torres, F. F., & Murrieta, A. C. (2026). MOSAIC: Modular
Multi-Model Selection and Cross-Validation (0.1.5). Tecnologico de
Monterrey. https://github.com/NanoBiostructuresRG/mosaic
```

---

## Authors

Developed by **Flavio F. Contreras-Torres**
Tecnologico de Monterrey

Co-author: **Ana C. Murrieta**
Tecnologico de Monterrey

---

## License

This project is licensed under the terms of the
[GNU Lesser General Public License v3.0 or later](LICENSE).

SPDX identifier:

```text
LGPL-3.0-or-later
```
