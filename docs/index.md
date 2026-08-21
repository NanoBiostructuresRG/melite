# MELITE — Multi-Model Classifier Evaluator

<section class="ms-hero">
  <div class="ms-hero__content">
    <p class="ms-eyebrow">Multi-model classifier evaluation</p>
    <div class="ms-brand" aria-label="MELITE">
      <span class="ms-dotmark" aria-hidden="true">
        <span></span><span></span><span></span>
        <span></span><span></span><span></span>
        <span></span><span></span><span></span>
      </span>
      <span class="ms-wordmark">MELITE</span>
    </div>
    <p class="ms-subtitle">
      Evaluate and compare classifiers on numeric tabular data with nested
      cross-validation, explicit model selection, persistent evaluation
      evidence, final model export, and artifact-based inference.
    </p>
    <div class="ms-actions">
      <a class="md-button md-button--primary" href="usage/">Usage</a>
      <a class="md-button" href="api/">API Reference</a>
      <a class="md-button" href="changelog/">Changelog</a>
    </div>
    <div class="ms-badges" aria-label="Project badges">
      <a href="https://github.com/NanoBiostructuresRG/melite/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-LGPL_v3-blue.svg" alt="License: LGPL v3+"></a>
      <a href="https://pypi.org/project/melite/"><img src="https://img.shields.io/pypi/pyversions/melite.svg" alt="Python versions"></a>
      <a href="https://pypi.org/project/melite/"><img src="https://img.shields.io/pypi/v/melite.svg" alt="PyPI"></a>
      <a href="https://github.com/NanoBiostructuresRG/melite/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/NanoBiostructuresRG/melite/actions/workflows/ci.yml/badge.svg"></a>
    </div>
  </div>
</section>

!!! note "Pre-stable"
    MELITE is currently in alpha-stage development (`v0.2.x`). Public
    interfaces may evolve before version 1.0.

## Overview

MELITE is a Python package and command-line tool for evaluating and comparing
classifiers on prepared numeric tabular datasets. It separates hyperparameter
tuning from model evaluation, preserves the evidence used for selection, and
exports the selected model as a reusable artifact.

MELITE operates at the tabular modeling level. Feature matrices may originate
from fingerprints, descriptors, dimensionality-reduction methods, clinical
variables, experimental measurements, industrial features, or other numeric
representations. MELITE evaluates the supplied matrices; it does not generate
those representations itself.

## Evaluation Workflow

<section class="ms-workflow" aria-label="MELITE evaluation workflow">
  <div class="ms-flow">
    <div class="ms-flow__item">
      <span class="ms-flow__kicker">Input</span>
      <strong>X / y</strong>
      <small>numeric tabular data</small>
    </div>
    <div class="ms-flow__item">
      <span class="ms-flow__kicker">Evaluate</span>
      <strong>melite run</strong>
      <small>nested evaluation</small>
    </div>
    <div class="ms-flow__item ms-flow__item--artifact">
      <span class="ms-flow__kicker">Evidence</span>
      <strong>CSV + figure</strong>
      <small>outer-CV results</small>
    </div>
    <div class="ms-flow__item">
      <span class="ms-flow__kicker">Select</span>
      <strong>F1-macro</strong>
      <small>mean outer-CV score</small>
    </div>
    <div class="ms-flow__item">
      <span class="ms-flow__kicker">Export</span>
      <strong>melite export</strong>
      <small>full-data fit</small>
    </div>
    <div class="ms-flow__item ms-flow__item--artifact">
      <span class="ms-flow__kicker">Inference</span>
      <strong>predict()</strong>
      <small>saved model artifact</small>
    </div>
  </div>
</section>

## Evaluation Contract

For each registered dataset, MELITE follows a reproducible evaluation contract:

1. `X` is validated as a two-dimensional numeric feature matrix and `y`
   provides labels for the same samples.
2. Each active classifier is evaluated under the configured outer
   cross-validation design.
3. For tunable classifiers, hyperparameter search occurs only within the
   training portion of each outer split.
4. Evaluation evidence comes from the held-out folds of repeated stratified
   outer cross-validation.
5. Mean outer-CV F1-macro is used to select the best active classifier for each
   dataset.
6. Aggregate and per-fold evidence are preserved for every evaluated
   classifier.
7. After selection, the chosen classifier is fitted using all available data.
   If it is tunable, MELITE performs a final full-data hyperparameter search to
   determine the exported configuration.
8. `melite export` does not run a second post-selection evaluation.

Smoke mode is intended for fast execution checks, not final model selection.

## Documentation

| Page | Purpose |
|---|---|
| [Usage](usage.md) | Installation, quick start, CLI, configuration, inputs, supported classifiers, evaluation settings, and outputs. |
| [API Reference](api.md) | Public Python API generated from package docstrings. |
| [Changelog](changelog.md) | Complete project history from the repository changelog. |

## Citation

For the current software metadata, see
[CITATION.cff](https://github.com/NanoBiostructuresRG/melite/blob/main/CITATION.cff).

The existing Zenodo record was published under MELITE's previous formal title
and should be cited as:

```text
Contreras-Torres, F. F., & Murrieta, A. C. (2026). MELITE: Multi-model Evaluation and Learning for Inference-ready Tabular Experiments. Zenodo. https://doi.org/10.5281/zenodo.20382752
```

## License

MELITE is licensed under the
[GNU Lesser General Public License v3.0 or later](https://github.com/NanoBiostructuresRG/melite/blob/main/LICENSE).

SPDX identifier: `LGPL-3.0-or-later`.
