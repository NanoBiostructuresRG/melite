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
      Multi-model selection, cross-validation and export toolkit for
      reduced-feature tabular classification.
    </p>
    <div class="ms-actions">
      <a class="md-button md-button--primary" href="#installation">Install</a>
      <a class="md-button" href="#quick-start">Quick start</a>
      <a class="md-button" href="api.md">API Reference</a>
    </div>
    <div class="ms-badges" aria-label="Project badges">
      <img alt="Version" src="https://img.shields.io/badge/version-v0.1.8-blue.svg">
      <img alt="Python versions" src="https://img.shields.io/badge/python-3.11%20%7C%203.12-blue">
      <img alt="License: LGPL v3+" src="https://img.shields.io/badge/License-LGPL_v3%2B-blue.svg">
    </div>
  </div>
</section>

!!! note "Pre-stable"
    MOSAIC is currently in Alpha-stage development (`v0.1.x`). The public API
    is being hardened before stability is declared.


## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/NanoBiostructuresRG/mosaic.git
cd mosaic
conda env create -f environment.yml
conda activate mosaic_env
pip install -e .
```

MOSAIC requires Python 3.11 or 3.12 and depends on NumPy, pandas,
scikit-learn, XGBoost, Matplotlib, and joblib.

<section class="ms-panel">
  <div class="ms-grid ms-grid--four">
    <article class="ms-card">
      <span class="ms-card__icon">01</span>
      <h3>Run</h3>
      <p>Grid search over SVC, Random Forest and XGBoost with repeated
      stratified K-fold cross-validation across all configured datasets.</p>
    </article>
    <article class="ms-card">
      <span class="ms-card__icon">02</span>
      <h3>Evaluate</h3>
      <p>Export TXT and CSV performance summaries with F1-macro, Accuracy,
      and AUC-ROC. Visualise CV fold distributions as three-panel PNG figures.</p>
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
      <p>Load a saved <code>.pkl</code> artifact and run artifact-based
      inference on new feature matrices.</p>
    </article>
  </div>
</section>

## Quick Start

=== "CLI"

    ```bash
    # Full benchmark
    mosaic run

    # Lightweight smoke test
    mosaic run --smoke

    # Export selected model non-interactively
    mosaic export --row 0

    # Verbose logging
    mosaic run --verbose

    # Custom configuration
    mosaic run --config my_config.toml
    ```

=== "Python API"

    ```python
    from mosaic import Config, load_dataset

    cfg = Config()
    cfg.setup()
    dataset = load_dataset(cfg, "PCA", [70, 85, 95])
    X, y = dataset["PCA85"]
    print(X.shape, y.shape)
    ```

## Input Format

MOSAIC consumes pre-computed feature matrices. Place your files as:

```text
raw/labels.npy          ← target vector y, shape (n_samples,)
data/PCA70.npz          ← required key: X, optional key: y
data/PCA85.npz
data/UMAP70.npz
...
```

If a `.npz` file contains an embedded `y` array, MOSAIC validates it against
`raw/labels.npy` and raises a descriptive error on mismatch.

## Configuration

Override any default setting with a TOML file:

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

## Python API Examples

=== "Config"

    ```python
    from mosaic import Config

    # Default configuration
    cfg = Config()
    cfg.setup()  # creates directories, sets random seeds

    # Smoke mode
    cfg_smoke = Config(smoke=True)

    # User overrides
    from pathlib import Path
    cfg_custom = Config(user_config=Path("my_config.toml"))
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

See the [API Reference](api.md) for full documentation generated from docstrings.

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
