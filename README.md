# MOSAIC: Modular Multi-Model Selection and Cross-Validation

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL_v3-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-v0.1.1-blue.svg)]()

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
0.1.1-dev
```

The active development branch is:

```text
dev/v0.1.1
```

MOSAIC is not yet packaged for PyPI. At this stage, it is intended to be used
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

Alternatively, if the environment already exists:

```bash
conda activate mosaic_env
```

Verify the main dependencies:

```bash
python -c "import numpy, pandas, sklearn, xgboost, joblib, matplotlib; print('dependencies OK')"
```

---

## Quick Start

### 1. Run the benchmarking phase

```bash
python main.py
```

This runs the configured benchmarking workflow and writes:

```text
output/results.txt
output/results.csv
```

### 2. Export a selected model interactively

```bash
python export_best_model.py
```

MOSAIC will display the available rows from `output/results.csv` and ask which
row should be exported.

### 3. Export a selected model non-interactively

```bash
python export_best_model.py --row 0
```

This selects row `0` from `output/results.csv`, retrains the corresponding
model on all available data, saves a `.pkl` artifact, and generates a metric
plot.

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

Structured table containing model performance and selected hyperparameters.

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
├── raw/                    # Local input labels and upstream feature data; ignored by Git
├── data/                   # Local PCA/UMAP reduced matrices; ignored by Git
├── output/                 # Local generated reports, figures and models; ignored by Git
│
├── config.py               # Paths, random seed, reduction levels, CV and hyperparameter grids
├── load_dataset.py         # Dataset loading and label consistency validation
├── model_training.py       # GridSearchCV, repeated CV and model selection
├── main.py                 # Main benchmarking pipeline
├── result_manager.py       # Human-readable TXT report writer
├── export_best_model.py    # Final model export workflow
├── plot_metrics.py         # CV metric distribution plots
│
├── environment.yml         # Conda development environment
├── CHANGELOG.md            # Version history
├── CITATION.cff            # Citation metadata
├── COPYING                 # GNU GPL v3 license text
├── COPYING.LESSER          # GNU LGPL v3 license text
├── LICENSE                 # Project license summary
└── README.md
```

---

## Models

MOSAIC currently benchmarks:

- Support Vector Classifier (`SVC`)
- Random Forest Classifier
- XGBoost Classifier

The configured hyperparameter grids are defined in `config.py`.

---

## Cross-Validation

MOSAIC uses repeated stratified K-fold cross-validation.

The current configuration is defined in `config.py`:

```python
CV_CONFIG = {
    "n_splits": 10,
    "n_repeats": 5,
    "random_state": 42,
}
```

This gives:

```text
N × M = 10 × 5 = 50 validation folds
```

per evaluated configuration.

---

## Example Console Output

```text
# Training phase
Running with PCA...
INFO:load_dataset:Labels
Training with PCA85 (level=85).
Running with UMAP...
INFO:load_dataset:Labels loaded:
Training with UMAP85 (level=85).
Final report written to output/results.txt
CSV file written to output/results.csv

# Export phase
$ python export_best_model.py
--------------------------------------------
  reduction_type  level              model_name  f1_macro  accuracy  auc_roc
0            PCA     85                     SVC    0.8336    0.8408   0.8802
1           UMAP     85  RandomForestClassifier    0.7041    0.7097   0.7855


Enter the row number to keep: 0

Training SVC on PCA85 using all available data...

```

---

## Notes

- Full benchmarking can be computationally expensive because MOSAIC performs
  grid search and repeated cross-validation.
- `output/`, `data/`, and `raw/` are local working directories and are ignored
  by Git.
- Model artifacts such as `.pkl` and `.joblib` files are ignored by Git.
- The current workflow is script-based and not yet a packaged Python API.
- PyPI packaging is planned for a future development phase.

---

## Validation

The current `dev/v0.1.1` branch has been validated with:

```bash
python -m py_compile config.py export_best_model.py load_dataset.py main.py model_training.py plot_metrics.py result_manager.py
```

Dataset loading smoke test:

```bash
python -c "from config import Config; from load_dataset import load_dataset; c=Config(); d=load_dataset(c,'PCA',[70]); print(d['PCA70'][0].shape, d['PCA70'][1].shape)"
```

CLI help smoke test:

```bash
python export_best_model.py --help
```

Minimal smoke tests were also performed for:

- SVC
- Random Forest
- XGBoost
- PNG figure generation
- `.pkl` model serialization

---

## Roadmap

Near-term development goals:

- Add formal tests with `pytest`.
- Introduce a package structure suitable for PyPI.
- Add a public command-line entry point.
- Improve configuration handling.
- Add documented example datasets.
- Add a prediction/inference module for exported `.pkl` artifacts.
- Add continuous integration.

---

## Citation

If you use MOSAIC in your research, please cite it using the metadata in
[CITATION.cff](CITATION.cff).

Suggested citation format:

```text
Contreras-Torres, F. F., & Murrieta, A. C. (2026). MOSAIC: Modular
Multi-Model Selection and Cross-Validation (0.1.1-dev). Tecnologico de
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
