# MELITE: Multi-model Evaluation and Learning for Inference-ready Tabular Experiments

[![CI](https://github.com/NanoBiostructuresRG/melite/actions/workflows/ci.yml/badge.svg)](https://github.com/NanoBiostructuresRG/melite/actions/workflows/ci.yml)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL_v3-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-v0.1.10-blue.svg)]()
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)]()

**MELITE** is a pre-stable Python toolkit for tabular classification
benchmarking, model selection, repeated stratified cross-validation, final
model export, and artifact-based inference.

MELITE is especially useful when feature matrices already exist, for example
molecular fingerprints, PCA-reduced descriptors, or UMAP-reduced
representations produced by an upstream workflow.

## Project Identity

```text
Project: MELITE
PyPI distribution: melite
Import package: melite
CLI: melite
Version: 0.1.10
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
result = predict("examples/output/Model_SVC_PCA70.pkl", X_new)
print(result["predictions"])
print(result["probabilities"])
```

## Scope

| MELITE does | MELITE does not |
|-------------|-----------------|
| Accept prepared `X` and `y` arrays. | Generate fingerprints. |
| Benchmark SVC, Random Forest, and XGBoost classifiers. | Process SMILES. |
| Select the best row by F1-macro. | Generate PCA or UMAP reductions from raw data. |
| Export a final retrained `.pkl` model. | Act as a general AutoML framework. |
| Run artifact-based inference through `predict()`. | Promise a stable 1.0 API yet. |

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
from melite import load_dataset
from melite import ResultManager
from melite import plot_cv_distributions
from melite import predict
from melite import __version__
```

Modules not listed above are importable directly but are not part of the public
contract and may change before 0.2.0.

## Input Format

```text
raw/labels.npy          <- target vector y, shape (n_samples,)
data/PCA70.npz          <- required key: X, optional key: y
data/PCA85.npz
data/UMAP70.npz
data/UMAP85.npz
```

Each `.npz` file must contain an `X` array. If an embedded `y` array is present,
MELITE validates it against `raw/labels.npy`.

## Outputs

```text
output/
|-- results.txt
|-- results.csv
|-- Model_<model>_<reduction><level>.pkl
`-- figures/
    `-- <model>_<reduction><level>.png
```

Local inputs and generated artifacts such as `raw/`, `data/`, `output/`,
`.pkl`, and `.joblib` files are intentionally ignored by Git.

## Validation

The current `dev/v0.1.10` branch targets:

```bash
python -m pytest tests/ -v --basetemp=.review_pytest_tmp -o cache_dir=.review_pytest_cache
mkdocs build --strict
python -m build
python -m twine check dist/*
melite --help
melite run --help
melite export --help
melite --version
```

## Citation

If you use MELITE in your research, please cite it using the metadata in
[CITATION.cff](CITATION.cff).

```text
Contreras-Torres, F. F., & Murrieta, A. C. (2026). MELITE: Multi-model
Evaluation and Learning for Inference-ready Tabular Experiments (0.1.10).
Tecnologico de Monterrey. https://github.com/NanoBiostructuresRG/melite
```

## Authors

Developed by **Flavio F. Contreras-Torres**

Tecnologico de Monterrey

Co-author: **Ana C. Murrieta**

Tecnologico de Monterrey

## License

This project is licensed under the terms of the
[GNU Lesser General Public License v3.0 or later](LICENSE).

SPDX identifier: `LGPL-3.0-or-later`
