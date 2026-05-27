# MELITE

<section class="ms-hero">
  <div class="ms-hero__content">
    <p class="ms-eyebrow">Tabular classification benchmarking</p>
    <div class="ms-brand" aria-label="MELITE">
      <span class="ms-dotmark" aria-hidden="true">
        <span></span><span></span><span></span>
        <span></span><span></span><span></span>
        <span></span><span></span><span></span>
      </span>
      <span class="ms-wordmark">MELITE</span>
    </div>
    <p class="ms-subtitle">
      Tabular classification benchmarking toolkit for model selection with repeated stratified cross-validation.
    </p>
    <div class="ms-actions">
      <a class="md-button md-button--primary" href="installation/">Install</a>
      <a class="md-button" href="quickstart/">Quick start</a>
      <a class="md-button" href="api/">API Reference</a>
    </div>
    <div class="ms-badges" aria-label="Project badges">
      <img alt="CI" src="https://github.com/NanoBiostructuresRG/melite/actions/workflows/ci.yml/badge.svg">
      <img alt="Version" src="https://img.shields.io/badge/version-v0.1.11-blue.svg">
      <img alt="Python versions" src="https://img.shields.io/badge/python-3.11%20%7C%203.12-blue">
      <img alt="License: LGPL v3+" src="https://img.shields.io/badge/License-LGPL_v3%2B-blue.svg">
    </div>
  </div>
</section>

!!! note "Pre-stable"
    MELITE is currently in alpha-stage development (`v0.1.x`). Publication on
    PyPI is prepared under the package name `melite`. Public APIs may
    change before 0.2.0.

## Workflow

<section class="ms-workflow" aria-label="MELITE workflow">
  <div class="ms-flow">
    <div class="ms-flow__item">
      <span class="ms-flow__kicker">Input</span>
      <strong>X / y</strong>
      <small>prepared arrays</small>
    </div>
    <div class="ms-flow__item">
      <span class="ms-flow__kicker">Benchmark</span>
      <strong>melite run</strong>
      <small>cross-validation</small>
    </div>
    <div class="ms-flow__item ms-flow__item--artifact">
      <span class="ms-flow__kicker">Results</span>
      <strong>results.csv</strong>
      <small>ranked rows</small>
    </div>
    <div class="ms-flow__item">
      <span class="ms-flow__kicker">Export</span>
      <strong>melite export</strong>
      <small>final retraining</small>
    </div>
    <div class="ms-flow__item ms-flow__item--artifact">
      <span class="ms-flow__kicker">Artifact</span>
      <strong>.pkl</strong>
      <small>saved model</small>
    </div>
    <div class="ms-flow__item">
      <span class="ms-flow__kicker">Inference</span>
      <strong>predict()</strong>
      <small>new matrices</small>
    </div>
  </div>
</section>

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
      select by F1-macro.</p>
    </article>
    <article class="ms-card">
      <span class="ms-card__icon">03</span>
      <h3>Export</h3>
      <p>Retrain the selected model on all available data and save a reusable
      <code>.pkl</code> artifact.</p>
    </article>
    <article class="ms-card">
      <span class="ms-card__icon">04</span>
      <h3>Predict</h3>
      <p>Load the exported artifact and run inference on new matrices with the
      same feature representation.</p>
    </article>
  </div>
</section>

## Scope

MELITE is tabular at the modeling level. The learning algorithms only consume
numeric `X` and `y` arrays, so the feature matrix may come from PCA, UMAP,
fingerprints, descriptors, clinical variables, experimental measurements,
industrial features, or manually selected numeric features.

| MELITE does | MELITE does not |
|-------------|-----------------|
| Accept prepared `X` and `y` arrays. | Generate PCA or UMAP representations. |
| Benchmark SVC, Random Forest, and XGBoost classifiers. | Engineer molecular fingerprints or descriptors. |
| Select the best row by F1-macro. | Handle raw molecular data directly. |
| Export a final retrained `.pkl` model. | Require internet access at runtime. |
| Run artifact-based inference through `predict()`. | Train deep learning models. |
| Handle any numeric tabular matrix. | Generate descriptors or reductions from raw data. |

MELITE uses a dataset registry under `[datasets.<dataset_id>]`. Each
`dataset_id` names one concrete numeric `X` matrix candidate.

```toml
[datasets.morgan_r2_2048]
path = "data/morgan_r2_2048.npz"
label_path = "raw/labels.npy"
family = "fingerprints"
method = "Morgan"

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

Required fields are `path` and `label_path`; optional metadata fields are
`family`, `method`, `variant`, `level`, and `description`. Legacy
`[benchmark].reduction_types` and `levels` configs are still normalized into
dataset entries when `[datasets]` is absent.

## Quick Example

```bash
python -m pip install melite
melite run --smoke --config examples/example_config.toml
melite export --row 0 --csv examples/output/results.csv --outdir examples/output/
```

```python
import numpy as np
from melite import predict

X_new = np.load("examples/sample_PCA70.npz")["X"]
result = predict("examples/output/Model_SVC_sample_pca70.pkl", X_new)
print(result["predictions"])
```

## Documentation

| Page | Purpose |
|------|---------|
| [Installation](installation.md) | Supported Python versions, local install, and optional dependencies. |
| [Quick Start](quickstart.md) | Minimal CLI and Python workflow using the bundled example data. |
| [CLI Reference](cli.md) | `melite run`, `melite export`, smoke mode, config files, and version checks. |
| [Configuration](configuration.md) | Default TOML settings, user overrides, inputs, and outputs. |
| [API Reference](api.md) | Public Python API generated from docstrings. |
| [Release Notes](release.md) | Version history and validation notes. |

## Citation

If you use MELITE in your research, please cite it using the metadata in
[CITATION.cff](https://github.com/NanoBiostructuresRG/melite/blob/main/CITATION.cff).

```text
Contreras-Torres, F. F., & Murrieta, A. C. (2026). MELITE: Multi-model
Evaluation and Learning for Inference-ready Tabular Experiments (0.1.11). Tecnologico de
Monterrey. https://github.com/NanoBiostructuresRG/melite
```

## License

This project is licensed under the terms of the
[GNU Lesser General Public License v3.0 or later](https://github.com/NanoBiostructuresRG/melite/blob/main/LICENSE).
SPDX identifier: `LGPL-3.0-or-later`.
