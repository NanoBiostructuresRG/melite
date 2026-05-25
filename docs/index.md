# MOSAIC

<section class="ms-hero">
  <div class="ms-hero__content">
    <p class="ms-eyebrow">Tabular classification benchmarking</p>
    <div class="ms-brand" aria-label="MOSAIC">
      <span class="ms-dotmark" aria-hidden="true">
        <span></span><span></span><span></span>
        <span></span><span></span><span></span>
        <span></span><span></span><span></span>
      </span>
      <span class="ms-wordmark">MOSAIC</span>
    </div>
    <p class="ms-subtitle">
      Model selection, repeated stratified cross-validation, final model export,
      and artifact-based inference for prepared tabular feature matrices.
    </p>
    <div class="ms-actions">
      <a class="md-button md-button--primary" href="#installation">Install</a>
      <a class="md-button" href="#quick-start">Quick start</a>
      <a class="md-button" href="api.md">API Reference</a>
    </div>
    <div class="ms-badges" aria-label="Project badges">
      <img alt="CI" src="https://github.com/NanoBiostructuresRG/mosaic/actions/workflows/ci.yml/badge.svg">
      <img alt="Version" src="https://img.shields.io/badge/version-v0.1.8-blue.svg">
      <img alt="Python versions" src="https://img.shields.io/badge/python-3.11%20%7C%203.12-blue">
      <img alt="License: LGPL v3+" src="https://img.shields.io/badge/License-LGPL_v3%2B-blue.svg">
    </div>
  </div>
</section>

!!! note "Pre-stable"
    MOSAIC is currently in alpha-stage development (`v0.1.x`). It is being
    prepared for first PyPI publication as `mosaic-tabular`, but is not yet
    published on PyPI. Public APIs may change before 1.0.

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/NanoBiostructuresRG/mosaic.git
cd mosaic
python -m pip install -e .
```

MOSAIC requires Python 3.11 or 3.12 and depends on NumPy, pandas,
scikit-learn, XGBoost, Matplotlib, and joblib.

<section class="ms-panel">
  <div class="ms-grid ms-grid--four">
    <article class="ms-card">
      <span class="ms-card__icon">01</span>
      <h3>Run</h3>
      <p>Evaluate configured feature matrices with SVC, Random Forest, and
      XGBoost model grids.</p>
    </article>
    <article class="ms-card">
      <span class="ms-card__icon">02</span>
      <h3>Select</h3>
      <p>Compare configurations with repeated stratified cross-validation and
      choose the best-performing model.</p>
    </article>
    <article class="ms-card">
      <span class="ms-card__icon">03</span>
      <h3>Export</h3>
      <p>Retrain the selected model on all available data and save a
      deployable <code>.pkl</code> artifact.</p>
    </article>
    <article class="ms-card">
      <span class="ms-card__icon">04</span>
      <h3>Predict</h3>
      <p>Load the exported artifact and run inference on new matrices with the
      same feature representation.</p>
    </article>
  </div>
</section>

## Quick Start

The fastest way to try MOSAIC is with the included synthetic example dataset:

```bash
git clone https://github.com/NanoBiostructuresRG/mosaic.git
cd mosaic
python -m pip install -e .
```

Run a lightweight benchmark:

```bash
mosaic run --smoke --config examples/example_config.toml
```

Export a selected model:

```bash
mosaic export --row 0 --csv examples/output/results.csv --outdir examples/output/
```

Run artifact-based inference on new data:

```python
import numpy as np
from mosaic import predict

X_new = np.load("examples/sample_PCA70.npz")["X"]
result = predict("examples/output/Model_SVC_PCA70.pkl", X_new)
print(result["predictions"])    # shape (n_samples,)
print(result["probabilities"])  # shape (n_samples, n_classes)
```

## Why MOSAIC?

Many tabular classification projects already have feature matrices prepared
before model selection begins. In cheminformatics workflows, those matrices may
come from molecular fingerprints, PCA-reduced descriptors, or UMAP-reduced
representations. The difficult part is then making model comparison
reproducible: every feature representation, model family, hyperparameter
setting, and validation split must be evaluated consistently.

**MOSAIC** was created for that benchmarking step. It takes prepared tabular
matrices and labels, runs configured model grids, evaluates each configuration
with repeated stratified cross-validation, writes traceable result files, and
exports the selected final model as a reusable artifact.

!!! tip "Scope contract"
    MOSAIC does not generate PCA, UMAP, molecular fingerprints, or raw
    molecular descriptors. It consumes the feature matrices produced by an
    upstream workflow and focuses on benchmarking, selection, export, and
    artifact-based inference.

## What You Provide and Receive

| Stage | User provides | MOSAIC does | User receives |
|-------|---------------|-------------|---------------|
| Inputs | `raw/labels.npy` and `.npz` feature matrices with an `X` array. | Loads matrices and validates optional embedded `y` labels. | Aligned `X, y` datasets for each configured representation. |
| Configuration | Defaults from `mosaic/config_default.toml` plus optional TOML overrides. | Merges user settings over defaults and applies smoke mode when requested. | A reproducible run profile. |
| Benchmark | Prepared matrices such as `data/PCA70.npz` or `data/UMAP85.npz`. | Runs grid search and repeated stratified cross-validation. | `output/results.txt` and `output/results.csv`. |
| Export | A selected row from `results.csv`. | Retrains that model on all available data and plots CV metric distributions. | A `.pkl` model artifact and PNG metric figure. |
| Inference | A `.pkl` model artifact and a new 2-D NumPy feature matrix. | Loads the model and calls `predict` and, when available, `predict_proba`. | A prediction dictionary with labels, probabilities, model path, and sample count. |

## Input Format

MOSAIC consumes pre-computed feature matrices and labels. A typical project
layout looks like this:

```text
raw/labels.npy          <- target vector y, shape (n_samples,)
data/PCA70.npz          <- required key: X, optional key: y
data/PCA85.npz
data/UMAP70.npz
...
```

Each `.npz` file must contain an `X` array. If an embedded `y` array is present,
MOSAIC validates it against `raw/labels.npy` to avoid silent feature-label
mismatches.

## Configuration

MOSAIC reads defaults from `mosaic/config_default.toml`. Override only the
settings you need with a user TOML file:

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
mosaic export --config my_config.toml --row 0
```

## CLI Reference

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
mosaic export
mosaic export --row 0
mosaic export --config my_config.toml --row 0
mosaic export --row 0 --force
```

Smoke-mode results are marked in `results.csv` and are blocked from export by
default. Use `--force` only when you intentionally want to export a
non-benchmark-quality smoke result.

## Outputs

MOSAIC writes outputs under the local `output/` directory by default:

```text
output/
|-- results.txt
|-- results.csv
|-- Model_<model>_<reduction><level>.pkl
`-- figures/
    `-- <model>_<reduction><level>.png
```

| Output | Purpose |
|--------|---------|
| `results.txt` | Human-readable report with selected model and metrics for each configuration. |
| `results.csv` | Structured table with model performance, parameters, and smoke-mode marker. |
| `.pkl` artifact | Final model retrained on all available data for the selected row. |
| PNG figure | Three-panel distribution plot for F1, Accuracy, and AUC-ROC across CV folds. |

Local inputs and generated artifacts such as `raw/`, `data/`, `output/`,
`.pkl`, and `.joblib` files are intentionally ignored by Git.

## Python API Examples

=== "Config"

    ```python
    from pathlib import Path
    from mosaic import Config

    cfg = Config(user_config=Path("my_config.toml"))
    cfg.setup()
    ```

=== "load_dataset"

    ```python
    from mosaic import Config, load_dataset

    cfg = Config()
    cfg.setup()
    dataset = load_dataset(cfg, "PCA", [70, 85])
    X, y = dataset["PCA70"]
    print(f"X: {X.shape}, y: {y.shape}")
    ```

=== "ResultManager"

    ```python
    from mosaic import ResultManager

    rm = ResultManager("output/results.txt")
    rm.write_results("Model: SVC\nF1: 0.85")
    rm.write_csv(rows, "output/results.csv", smoke=False)
    ```

=== "plot_cv_distributions"

    ```python
    from pathlib import Path
    from mosaic import plot_cv_distributions

    plot_cv_distributions(
        f1=[0.76, 0.90, 0.82],
        acc=[0.77, 0.90, 0.82],
        auc=[0.83, 0.95, 0.89],
        model_name="SVC",
        params="{'kernel': 'linear', 'C': 1}",
        save_to=Path("output/figures/SVC_PCA70.png"),
    )
    ```

=== "predict"

    ```python
    import numpy as np
    from mosaic import predict

    X_new = np.load("data/PCA85.npz")["X"]
    result = predict("output/Model_SVC_PCA85.pkl", X_new)
    print(result["predictions"])    # shape (n_samples,)
    print(result["probabilities"])  # shape (n_samples, n_classes)
    ```

## Public API

| Symbol | Description |
|--------|-------------|
| [`Config`](api.md#config) | Runtime configuration object |
| [`load_dataset`](api.md#load_dataset) | Load reduced feature matrices and labels |
| [`ResultManager`](api.md#resultmanager) | Write TXT and CSV benchmark results |
| [`plot_cv_distributions`](api.md#plot_cv_distributions) | Generate CV metric distribution plots |
| [`predict`](api.md#predict) | Artifact-based inference using exported `.pkl` models |
| `__version__` | Package version string |

See the [API Reference](api.md) for the intended public API documentation
generated from docstrings.

## Citation

If you use MOSAIC in your research, please cite it using the metadata in
[CITATION.cff](https://github.com/NanoBiostructuresRG/mosaic/blob/main/CITATION.cff).

```text
Contreras-Torres, F. F., & Murrieta, A. C. (2026). MOSAIC: Modular
Multi-Model Selection and Cross-Validation (0.1.8). Tecnologico de
Monterrey. https://github.com/NanoBiostructuresRG/mosaic
```

## License

This project is licensed under the terms of the
[GNU Lesser General Public License v3.0 or later](https://github.com/NanoBiostructuresRG/mosaic/blob/main/LICENSE).
SPDX identifier: `LGPL-3.0-or-later`.
