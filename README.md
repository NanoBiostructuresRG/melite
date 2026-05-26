# MOSAIC: Modular Multi-Model Selection and Cross-Validation

[![CI](https://github.com/NanoBiostructuresRG/mosaic/actions/workflows/ci.yml/badge.svg)](https://github.com/NanoBiostructuresRG/mosaic/actions/workflows/ci.yml)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL_v3-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-v0.1.9-blue.svg)]()
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)]()

**MOSAIC** is a lightweight benchmarking toolkit for prepared tabular
classification datasets. It compares SVC, Random Forest, and XGBoost model
configurations with repeated stratified cross-validation, selects the best row
by F1-macro, exports a final retrained `.pkl` artifact, and supports
artifact-based inference through `predict()`.

MOSAIC is especially useful when feature matrices already exist, for example
molecular fingerprints, PCA-reduced descriptors, or UMAP-reduced
representations produced by an upstream workflow.

## Documentation

The live documentation is published at:

https://nanobiostructuresrg.github.io/mosaic/

Key pages:

- [Installation](https://nanobiostructuresrg.github.io/mosaic/installation/)
- [Quick Start](https://nanobiostructuresrg.github.io/mosaic/quickstart/)
- [CLI Reference](https://nanobiostructuresrg.github.io/mosaic/cli/)
- [Configuration](https://nanobiostructuresrg.github.io/mosaic/configuration/)
- [API Reference](https://nanobiostructuresrg.github.io/mosaic/api/)

## Development Status

MOSAIC is currently in **alpha-stage, pre-stable development**.

```text
Version: 0.1.9
Branch:  dev/v0.1.9
```

MOSAIC is being prepared for publication on PyPI as `mosaic-tabular`. Until
that package is published, install from the repository in editable mode.

## Installation

```bash
git clone https://github.com/NanoBiostructuresRG/mosaic.git
cd mosaic
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
mosaic run --smoke --config examples/example_config.toml
```

Export a selected model artifact:

```bash
mosaic export --row 0 --csv examples/output/results.csv --outdir examples/output/
```

Run artifact-based inference:

```python
import numpy as np
from mosaic import predict

X_new = np.load("examples/sample_PCA70.npz")["X"]
result = predict("examples/output/Model_SVC_PCA70.pkl", X_new)
print(result["predictions"])
print(result["probabilities"])
```

## Scope

| MOSAIC does | MOSAIC does not |
|-------------|-----------------|
| Accept prepared `X` and `y` arrays. | Generate PCA or UMAP representations. |
| Benchmark SVC, Random Forest, and XGBoost classifiers. | Engineer molecular fingerprints or descriptors. |
| Select the best row by F1-macro. | Handle raw molecular data directly. |
| Export a final retrained `.pkl` model. | Require internet access at runtime. |
| Run artifact-based inference through `predict()`. | Train deep learning models. |

## CLI

```bash
mosaic --help
mosaic run --help
mosaic export --help
mosaic --version
```

Common commands:

```bash
mosaic run
mosaic run --smoke
mosaic run --config my_config.toml
mosaic export --row 0
mosaic export --config my_config.toml --row 0
mosaic export --row 0 --force
```

## Public API

```python
from mosaic import Config
from mosaic import load_dataset
from mosaic import ResultManager
from mosaic import plot_cv_distributions
from mosaic import predict
from mosaic import __version__
```

Modules not listed above are importable directly but are not part of the public
contract and may change before 1.0.

## Input Format

```text
raw/labels.npy          <- target vector y, shape (n_samples,)
data/PCA70.npz          <- required key: X, optional key: y
data/PCA85.npz
data/UMAP70.npz
data/UMAP85.npz
```

Each `.npz` file must contain an `X` array. If an embedded `y` array is present,
MOSAIC validates it against `raw/labels.npy`.

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

The current `dev/v0.1.9` branch targets:

```bash
pytest tests/ -v
mkdocs build --strict
python -m build
python -m twine check dist/*
mosaic --help
mosaic run --help
mosaic export --help
mosaic --version
```

The test suite currently covers 82 tests across configuration loading, dataset
loading, result writing, final model export, plotting, public API imports, CLI
behavior, artifact-based inference, and example dataset integrity.

## Citation

If you use MOSAIC in your research, please cite it using the metadata in
[CITATION.cff](CITATION.cff).

```text
Contreras-Torres, F. F., & Murrieta, A. C. (2026). MOSAIC: Modular
Multi-Model Selection and Cross-Validation (0.1.9). Tecnologico de
Monterrey. https://github.com/NanoBiostructuresRG/mosaic
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
