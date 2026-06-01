# MELITE: Multi-model Evaluation and Learning for Inference-ready Tabular Experiments

[![CI](https://github.com/NanoBiostructuresRG/melite/actions/workflows/ci.yml/badge.svg)](https://github.com/NanoBiostructuresRG/melite/actions/workflows/ci.yml)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL_v3-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-v0.2.3-blue.svg)]()
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)]()

**MELITE** is a pre-stable Python toolkit for tabular classification
benchmarking, model selection, repeated stratified cross-validation, final
model export, and artifact-based inference.

MELITE is tabular at the modeling level. The learning algorithms consume
numeric `X` and `y` arrays, so the feature matrix may come from PCA, UMAP,
fingerprints, descriptors, clinical variables, experimental measurements,
industrial features, or manually selected numeric features.

## Project Identity

```text
Project: MELITE
PyPI distribution: melite
Import package: melite
CLI: melite
Version: 0.2.3
License: LGPL-3.0-or-later
Status: alpha / pre-stable
```

## Documentation

The live documentation is published at:

https://nanobiostructuresrg.github.io/melite/

Key pages:

- [Installation](https://nanobiostructuresrg.github.io/melite/installation/)
- [Quick Start](https://nanobiostructuresrg.github.io/melite/quickstart/)
- [CLI Reference](https://nanobiostructuresrg.github.io/melite/cli/)
- [Configuration](https://nanobiostructuresrg.github.io/melite/configuration/)
- [API Reference](https://nanobiostructuresrg.github.io/melite/api/)

## Installation

After PyPI publication:

```bash
python -m pip install melite
```

For local development:

```bash
git clone https://github.com/NanoBiostructuresRG/melite.git
cd melite
python -m pip install -e .
```

For development and documentation tools:

```bash
python -m pip install -e ".[dev]"
python -m pip install -e ".[docs]"
```

## Quick Start

Run a fast smoke benchmark with the bundled synthetic example dataset:

```bash
melite run --smoke --config examples/example_config.toml
```

Export a selected model artifact:

```bash
melite export --row 0 --csv examples/output/results.csv --outdir examples/output/
```

Run artifact-based inference:

```python
import numpy as np
from melite import predict

X_new = np.load("examples/sample_PCA70.npz")["X"]
result = predict("examples/output/Model_SVC_sample_pca70.pkl", X_new)
print(result["predictions"])
print(result["probabilities"])
```

## Scope

| MELITE does | MELITE does not |
|-------------|-----------------|
| Accept prepared `X` and `y` arrays. | Generate fingerprints. |
| Benchmark SVC, Random Forest, XGBoost, and opt-in experimental stacking classifiers. | Process SMILES. |
| Select the best row by F1-macro. | Generate PCA or UMAP reductions from raw data. |
| Export a final retrained `.pkl` model. | Act as a general AutoML framework. |
| Run artifact-based inference through `predict()`. | Promise a stable 1.0 API yet. |
| Handle any numeric tabular matrix. | Generate or validate domain-specific descriptors. |

Datasets are registered as concrete tabular matrix candidates under
`[datasets.<dataset_id>]`. The `dataset_id` is user-defined and is used in
`results.csv`, figures, and exported model filenames.

```toml
[datasets.morgan_r2_2048]
path = "data/morgan_r2_2048.npz"
label_path = "raw/labels.npy"
family = "fingerprints"
method = "Morgan"
variant = "r2_2048"

[datasets.rdkit_descriptors]
path = "data/rdkit_descriptors.npz"
label_path = "raw/labels.npy"
family = "descriptors"
method = "RDKit"

[datasets.pca85]
path = "data/PCA85.npz"
label_path = "raw/labels.npy"
family = "dimensionality"
method = "PCA"
level = 85
```

Each registered dataset must define `path` and `label_path`. Optional metadata
fields are `family`, `method`, `variant`, `level`, and `description`; they are
reported for traceability and do not drive special-case model execution.
Registered datasets are loaded strictly: missing files, missing `X`, non-2D or
non-numeric `X`, length mismatches, and embedded `y` mismatches fail the run.
Legacy `[benchmark].reduction_types` and `levels` configs are still accepted
and are normalized into equivalent dataset entries such as `PCA70` and `UMAP90`.

Model families are controlled by `[models].active`:

```toml
[models]
active = ["svc", "rf", "xgb"]
```

Remove a key to skip that family during training. Valid keys are `svc`, `rf`,
`xgb`, and experimental `stack`. Stacking is opt-in; add `"stack"` to
`active` to evaluate an sklearn `StackingClassifier` alongside the default
families.

Standalone SVC is trained and exported as a `StandardScaler` -> `SVC` sklearn
pipeline because SVM/kernel-based methods are sensitive to feature scale.
Random Forest and XGBoost are tree-based models and remain unscaled by
default. Experimental stacking uses `stack_method="predict_proba"` with a
scaled probabilistic SVC base estimator, unscaled RF/XGBoost base estimators,
and a logistic regression final estimator. Its internal stacking CV uses the
configured split count and random state with one repeat to satisfy sklearn's
out-of-fold prediction requirements. Final exports remain `.pkl` artifacts
serialized with `joblib`; Optuna and MLflow are not part of v0.2.3.
    
## CLI

```bash
melite --help
melite run --help
melite export --help
melite --version
```

Common commands:

```bash
melite run
melite run --smoke
melite run --config my_config.toml
melite export --row 0
melite export --config my_config.toml --row 0
melite export --row 0 --force
```

## Public API

```python
from melite import Config
from melite import load_datasets
from melite import plot_cv_distributions
from melite import predict
from melite import __version__
```

Modules not listed above are importable directly but are not part of the public
contract and may change before 1.0.

## Input Format

```text
raw/labels.npy          <- target vector y, shape (n_samples,)
data/morgan_r2_2048.npz <- required key: X, optional key: y
data/rdkit_descriptors.npz
data/PCA85.npz
data/UMAP90.npz
```

Each `.npz` file must contain an `X` array. If an embedded `y` array is present,
MELITE validates it against the configured `label_path`.

## Outputs

```text
output/
|-- results.txt
|-- results.csv
|-- Model_<model>_<dataset>.pkl
`-- figures/
    `-- <model>_<dataset>.png
```

Local inputs and generated artifacts such as `raw/`, `data/`, `output/`,
`.pkl`, and `.joblib` files are intentionally ignored by Git.

## Validation

The current `dev/v0.2.3` branch targets:

```bash
python -m pytest tests/ -v --basetemp=.review_pytest_tmp -o cache_dir=.review_pytest_cache
mkdocs build --strict
python -m build --no-isolation
python -m twine check dist/*
python scripts/smoke_install_wheel.py
melite --help
melite run --help
melite export --help
melite --version
```

## Citation

If you use MELITE in your research, please cite it using the metadata in
[CITATION.cff](CITATION.cff).

```text
Contreras-Torres, F. F., & Murrieta, A. C. (2026). MELITE: Multi-model Evaluation and Learning for Inference-ready Tabular Experiments. Zenodo. https://doi.org/10.5281/zenodo.20382752
```

## Authors

Developed by **Flavio F. Contreras-Torres**. Tecnologico de Monterrey

Co-author: **Ana C. Murrieta**. Tecnologico de Monterrey

## License

This project is licensed under the terms of the
[GNU Lesser General Public License v3.0 or later](LICENSE).

SPDX identifier: `LGPL-3.0-or-later`
