# MELITE — Multi-Model Classifier Evaluator

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL_v3-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-v0.2.4-blue.svg)]()
[![PyPI](https://img.shields.io/pypi/v/melite.svg)](https://pypi.org/project/melite/)
[![Python](https://img.shields.io/pypi/pyversions/melite.svg)](https://pypi.org/project/melite/)
[![CI](https://github.com/NanoBiostructuresRG/melite/actions/workflows/ci.yml/badge.svg)](https://github.com/NanoBiostructuresRG/melite/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-teal.svg)](https://nanobiostructuresrg.github.io/melite/)


## Description

**MELITE** is a Python package and command-line tool for evaluating and
comparing classifiers on numeric tabular datasets. It separates hyperparameter
tuning from classifier evaluation, preserves the evidence used for selection, and
exports the selected model as a reusable artifact for downstream inference.

MELITE operates at the tabular modeling level. Its learning algorithms consume
numeric feature matrices (`X`) and target labels (`y`), regardless of how those
features were produced. Inputs may therefore originate from fingerprints,
descriptors, dimensionality-reduction methods, clinical variables, experimental
measurements, industrial features, or other numeric representations.

## Purpose

MELITE is designed to make classifier comparison and selection explicit,
reproducible, and auditable. Its workflow separates stages
that are often mixed together in small classification workflows:

- hyperparameter tuning;
- classifier evaluation;
- comparison and selection;
- final fitting on all available data;
- model export and inference.

This separation ensures that, within each outer cross-validation split, the
data used to evaluate a tuned classifier are held out from the hyperparameter
search that produced it, while preserving the evidence needed to understand
how the competing classifiers performed.


## Why Use MELITE?

- **Controlled evaluation.** Hyperparameter tuning is kept separate from the
  evidence used to compare classifiers.
- **Evidence preservation.** Aggregate and fold-level evaluation results are
  retained for every evaluated classifier, not only for the selected one.
- **Explicit selection.** Classifier selection follows a predefined criterion
  based on cross-validation evidence rather than an informal choice after
  training.
- **Domain-agnostic inputs.** MELITE works with numeric tabular data without
  assuming how the features were generated.
- **Reusable artifacts.** After selection, the chosen classifier can be fitted
  on all available data and saved as a model artifact for prediction.
- **CLI and Python interfaces.** MELITE can be used through its command-line
  workflow and through a focused public Python API.



## What MELITE Does

| MELITE does | MELITE does not |
|---|---|
| Evaluate multiple classifiers on prepared numeric `X` and `y`. | Generate domain-specific features or descriptors. |
| Tune supported classifiers within the evaluation design. | Act as a general AutoML framework. |
| Preserve aggregate and fold-level evaluation evidence. | Generate PCA, UMAP, fingerprints, or other feature representations. |
| Select the best active classifier by mean outer-CV F1-macro. | Process raw domain-specific inputs. |
| Fit and export the selected model as a `.pkl` artifact. | Perform automatic feature engineering or feature selection. |
| Run inference from exported model artifacts. | Guarantee a stable 1.0 API yet. |


## Evaluation Contract

For a registered dataset, MELITE follows the contract below:

1. `X` is a two-dimensional numeric feature matrix and `y` provides the target
   labels for the same samples.
2. Each active classifier is evaluated under the configured outer
   cross-validation design.
3. For tunable classifiers, hyperparameter search occurs only within the
   training portion of each outer split.
4. Evaluation evidence is obtained from the held-out folds of repeated
   stratified outer cross-validation.
5. Mean outer-CV F1-macro is used to select the best active classifier for each
   dataset.
6. Aggregate and per-fold evidence are preserved for every evaluated
   classifier.
7. After selection, the chosen classifier is fitted using all available data.
   If it is tunable, MELITE performs a final full-data hyperparameter search to
   determine the exported configuration.
8. `melite export` does not run a second post-selection evaluation. It fits the
   selected classifier on all available data and serializes the final model
   artifact.
9. Smoke mode is intended for fast execution checks, not final classifier selection.


## Installation

### Package Users


Install **MELITE** in a supported Python environment:

```bash
python -m pip install melite
```

Verify the installation:

```bash
melite --version
```

### Contributors and Developers

Clone the repository and install in editable mode with development dependencies:

```bash
git clone https://github.com/NanoBiostructuresRG/melite.git
cd melite
conda create -n melite_env python=3.11
conda activate melite_env
python -m pip install -e ".[dev]"
```

To build the documentation locally, install the `docs` extra as well:

```bash
python -m pip install -e ".[dev,docs]"
mkdocs serve
```

## Quick Start

### Command-Line Interface

Run a fast smoke evaluation with the bundled synthetic example dataset:

```bash
melite run --smoke --config examples/example_config.toml
```

Run a configured evaluation:

```bash
melite run --config my_config.toml
```

Export a selected model artifact from an existing results table:

```bash
melite export --config examples/example_config.toml --row 0 --csv examples/output/results.csv --outdir examples/output/
```

### Python API

Use an exported artifact for inference:

```python
import numpy as np
from melite import predict

X_new = np.load("examples/sample_PCA70.npz")["X"]
result = predict("examples/output/Model_SVC_sample_pca70.pkl", X_new)

print(result["predictions"])
print(result["probabilities"])
```

The current public API also exposes:

```python
from melite import Config
from melite import load_datasets
from melite import plot_f1_macro_evidence
from melite import predict
from melite import __version__
```

## Workflow


### CLI Workflow

The command-line interface provides the canonical end-to-end MELITE workflow:

1. Register one or more numeric datasets in a TOML configuration file.
2. Choose the active classifiers.
3. Run `melite run` to generate evaluation evidence and selected results.
4. Inspect `results.csv`, `evaluations.csv`, `evaluation_folds.csv`, and the
   dataset-level F1-macro evidence figures.
5. Run `melite export` for the selected result you want to preserve as a model
   artifact.
6. Use the exported `.pkl` artifact through `melite.predict()` for inference.


### Python Workflow

The Python API is intentionally component-oriented. It exposes configuration,
dataset loading, evaluation-evidence plotting, artifact-based prediction, and
version metadata as public symbols.

MELITE does not expose the full evaluation orchestration as a stable
high-level Python workflow API. For reproducible end-to-end execution, use the
CLI and a version-controlled TOML configuration.


## Supported Classifiers

**MELITE** currently supports four classifier keys:

| Key | Classifier | Active by default |
|---|---|---|
| `svc` | Support Vector Classifier | Yes |
| `rf` | Random Forest | Yes |
| `xgb` | XGBoost | Yes |
| `stack` | Stacking classifier | No |

The default configuration is:

```toml
[classifiers]
active = ["svc", "rf", "xgb"]
```

Add `"stack"` to evaluate Stacking alongside the default classifiers.

Standalone SVC is evaluated as a `StandardScaler` -> `SVC` pipeline, with
probability fitting disabled during standalone evaluation. Exported SVC
artifacts retain probability support for inference. Random Forest and XGBoost
remain unscaled. The opt-in Stacking classifier combines a scaled probabilistic SVC
with Random Forest and XGBoost base estimators and uses logistic regression as
the final estimator.

## Input Format

### Dataset Registry

Datasets are registered under user-defined `[datasets.<dataset_id>]` entries.
For example:

```toml
[datasets.morgan_r2_2048]
path = "data/morgan_r2_2048.npz"
label_path = "raw/labels.npy"
family = "fingerprints"
method = "Morgan"
variant = "r2_2048"
description = "Morgan fingerprints, radius 2, 2048 bits"
```

Here, `family` is optional **dataset metadata** used to describe the feature
representation. It is unrelated to the classifier selected or evaluated by
MELITE.

Each dataset must define `path` and `label_path`. Optional metadata fields such
as `family`, `method`, `variant`, `level`, and `description` are preserved for
traceability and do not trigger dataset-specific execution logic.

The legacy `[benchmark]` configuration section remains supported for backward
compatibility. New configurations should use the dataset registry.

### Array Requirements

A registered `.npz` dataset must contain an `X` array. MELITE validates that:

- `X` is two-dimensional;
- `X` is numeric;
- the number of rows in `X` matches the number of labels in `y`;
- an embedded `y` array, when present, matches the configured label vector.

A typical input layout is:

```text
raw/
└── labels.npy

data/
├── morgan_r2_2048.npz
├── rdkit_descriptors.npz
├── PCA85.npz
└── UMAP90.npz
```

The filenames and feature families are examples only. MELITE does not require
PCA, UMAP, fingerprints, descriptors, or any other specific feature-generation
method.

## Main Outputs

A standard **MELITE** workflow produces evaluation artifacts and, when requested,
a final exported model:

```text
output/
├── results.txt
├── results.csv
├── evaluations.csv
├── evaluation_folds.csv
├── figures/
│   └── evaluation_f1_macro_<dataset>.png
└── Model_<classifier>_<dataset>.pkl
```

The artifacts have distinct roles:

- `results.txt` — human-readable summary of the selected results.
- `results.csv` — selected classifier result for each dataset.
- `evaluations.csv` — aggregate evaluation evidence for every active classifier.
- `evaluation_folds.csv` — outer-CV evidence for every dataset, classifier, and
  outer split.
- `figures/evaluation_f1_macro_<dataset>.png` — visualization of the outer-CV
  F1-macro evidence used for classifier selection.
- `Model_<classifier>_<dataset>.pkl` — final full-data fitted model created by
  `melite export`.


The evaluation figure is generated from already-computed outer-CV evidence. It
does not trigger additional fitting, tuning, cross-validation, or selection.


## Configuration

MELITE uses TOML configuration files to keep execution choices explicit and
reproducible. Configuration controls, among other settings:

- registered datasets and their metadata;
- active classifiers;
- random state;
- inner and outer cross-validation settings;
- input and output paths.

Use `--config` to supply a project-specific configuration:

```bash
melite run --config my_config.toml
melite export --config my_config.toml --row 0
```

Smoke mode can be requested independently from the configuration:

```bash
melite run --smoke --config my_config.toml
```

See the full configuration reference in the project documentation.

## Development

### Project Structure

```text
MELITE/
|-- melite/
|   |-- __init__.py             # Public API
|   |-- cli.py                  # Command-line interface
|   |-- config.py               # Configuration loading and normalization
|   |-- config_default.toml     # Default configuration
|   |-- export_best_model.py    # Final model fitting and export
|   |-- load_dataset.py         # Dataset loading and validation
|   |-- main.py                 # Evaluation workflow orchestration
|   |-- model_training.py       # Classifier tuning, evaluation, and selection
|   |-- plot_metrics.py         # Evaluation evidence figures
|   |-- predict.py              # Artifact-based inference
|   |-- result_manager.py       # Results and artifact management
|   `-- version.py              # Package version metadata
|-- tests/                      # Test suite (pytest)
|-- examples/                   # Example dataset, configuration, and generator
|-- docs/                       # MkDocs documentation sources
|-- scripts/
|   `-- smoke_install_wheel.py  # Installed-wheel smoke validation
|-- .github/
|   `-- workflows/              # CI, documentation, and PyPI publishing
|-- pyproject.toml              # Build, package, and dependency metadata
|-- environment.yml             # Conda development environment
|-- mkdocs.yml                  # Documentation site configuration
|-- CHANGELOG.md
|-- CITATION.cff
|-- CODE_OF_CONDUCT.md
|-- CONTRIBUTING.md
|-- COPYING
|-- COPYING.LESSER
|-- LICENSE
`-- README.md
```

### Running Tests

Run the test suite:

```bash
python -m pytest tests -q
```

Build the documentation in strict mode:

```bash
mkdocs build --strict
```

Build and check the distributions:

```bash
python -m build --no-isolation
python -m twine check dist/*
```

Run the installed-wheel smoke test:

```bash
python scripts/smoke_install_wheel.py
```

## Contributing

Contributions are welcome. Please open an issue before submitting a pull
request. Follow the existing code style: NumPy-style docstrings, type hints,
and SPDX license headers in all source files.

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines, including the
development setup and the pull request target branch.
Please also read our [Code of Conduct](CODE_OF_CONDUCT.md).


## Documentation

The full documentation is published at:

https://nanobiostructuresrg.github.io/melite/


## Citation

If you use MELITE in your research, please cite it using the metadata in
[CITATION.cff](CITATION.cff) or the format below:


```text
Contreras-Torres, F. F., & Murrieta, A. C. (2026). MELITE: Multi-model Evaluation and Learning for Inference-ready Tabular Experiments. Zenodo. https://doi.org/10.5281/zenodo.20382752
```

## Authors

- **Flavio F. Contreras-Torres** — Tecnológico de Monterrey
- **Ana C. Murrieta** — Tecnológico de Monterrey


## License

This project is licensed under the terms of the
[GNU Lesser General Public License v3.0 or later](LICENSE).
SPDX identifier: `LGPL-3.0-or-later`.
