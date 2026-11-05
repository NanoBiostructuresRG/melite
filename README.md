# MELITE — Multi-Model Supervised Learning Evaluator

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL_v3-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-v0.3.3-blue.svg)](https://pypi.org/project/melite/)
[![PyPI](https://img.shields.io/pypi/v/melite.svg)](https://pypi.org/project/melite/)
[![Python](https://img.shields.io/pypi/pyversions/melite.svg)](https://pypi.org/project/melite/)
[![CI](https://github.com/NanoBiostructuresRG/melite/actions/workflows/ci.yml/badge.svg)](https://github.com/NanoBiostructuresRG/melite/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-teal.svg)](https://nanobiostructuresrg.github.io/melite/)


## Description

**MELITE** is a Python package and command-line application for evaluating and
comparing supervised classification and regression models on prepared tabular
datasets. It separates hyperparameter tuning from model evaluation, preserves
the evidence used for selection, and exports the selected model as a reusable
artifact for downstream inference.

MELITE operates at the tabular modeling level. Its established input contract
uses numeric feature matrices (`X`) and targets (`y`), regardless of how those
features were produced. CSV classification datasets evaluated exclusively with
CatBoost may instead declare categorical columns and use categorical-only or
mixed numeric/categorical predictors. Inputs may originate from fingerprints,
descriptors, dimensionality-reduction methods, clinical variables, experimental
measurements, industrial features, or other prepared representations.

Classification targets with two distinct labels use the binary evaluation
contract; targets with three or more distinct labels use the multiclass
contract. MELITE determines this from target cardinality and preserves the
original class identities through evaluation, export, and inference.

## Purpose

MELITE is designed to make supervised-model comparison and selection explicit,
reproducible, and auditable. Its workflow separates stages that are often mixed
together in machine-learning workflows:

- hyperparameter tuning;
- model evaluation;
- comparison and selection;
- final fitting on all available data;
- model export and inference.

This separation ensures that, within each outer cross-validation split, the
data used to evaluate a tuned model are held out from the hyperparameter
search that produced it, while preserving the evidence needed to understand
how the competing models performed.


## Why Use MELITE?

- **Controlled evaluation.** Hyperparameter tuning is kept separate from the
  evidence used to compare models.
- **Evidence preservation.** Aggregate and fold-level evaluation results are
  retained for every evaluated model, not only for the selected one.
- **Explicit selection.** Model selection follows a predefined, task-specific
  criterion based on cross-validation evidence rather than an informal choice
  after training.
- **Domain-agnostic inputs.** MELITE works with prepared tabular data without
  assuming how the features were generated; native categorical predictors are
  available for CatBoost-only CSV classification.
- **Reusable artifacts.** After selection, the chosen model can be fitted
  on all available data and saved as a model artifact for prediction.
- **CLI and Python interfaces.** MELITE can be used through its command-line
  workflow and through a focused public Python API.



## What MELITE Does

| MELITE does | MELITE does not |
|---|---|
| Evaluate supported classification and regression models on prepared tabular `X` and `y`. | Generate domain-specific features or descriptors. |
| Tune supported models within the evaluation design. | Act as a general AutoML framework. |
| Preserve aggregate and fold-level evaluation evidence. | Generate PCA, UMAP, fingerprints, or other feature representations. |
| Select the best active model using the task-specific primary metric. | Process raw domain-specific inputs. |
| Fit and export the selected model as a `.pkl` artifact. | Perform automatic feature engineering or feature selection. |
| Run inference from exported model artifacts. | Guarantee a stable 1.0 API yet. |

MELITE records model-evaluation evidence and supports reproducible execution;
it does not by itself establish external scientific validity or predictive
generalizability.


## Evaluation Contract

For each registered dataset, MELITE uses the configured `task` to apply a
task-specific evaluation contract:

1. Classification uses repeated stratified outer cross-validation and
   stratified inner cross-validation. Models are selected by highest mean outer
   F1-macro; accuracy and ROC-AUC are also reported. Binary ROC-AUC retains its
   binary definition, while multiclass ROC-AUC uses probability-based
   One-vs-One macro averaging.
2. Regression uses `RepeatedKFold` for outer evaluation and shuffled `KFold`
   for inner optimization. Models are selected by lowest mean outer RMSE; MAE
   and R² are also reported.
3. Inner hyperparameter optimization is confined to the training portion of
   each outer split. LinearRegression, PolynomialRegression, and
   StackingClassifier are evaluated without a separate Optuna search.
4. Exact selection ties preserve the configured model order. Aggregate and
   per-fold outer evidence are retained for every evaluated model. Classification
   additionally retains held-out precision, recall, F1-score, and support for
   every original class in every outer fold.
5. After selection, tunable models receive a fresh full-data optimization;
   non-tunable models are fitted directly. The resulting final parameters are
   persisted in the task-specific results CSV.
6. `melite export` reconstructs the selected model from those parameters, fits
   it on all available data, and serializes the artifact without another
   optimization, cross-validation, or selection step.
7. Smoke mode is intended for execution checks, not final model selection.

Outer cross-validation estimates performance under the configured evaluation
design; it does not establish external predictive generalizability. In repeated
cross-validation, the same observation can occur in more than one held-out
fold across repeats, so those records are not an independent test population.


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

The bundled example creates a ready-to-run synthetic numeric CSV classification
dataset and example configuration for a short classification workflow, so you
can verify that MELITE is installed and working. Cloning the repository is
**not** required.

```bash
melite example
melite run --smoke --config melite_example/config.toml
```

`melite example` creates `./melite_example/` in the current directory.

### Next steps

- [Prepare and use your own datasets](docs/usage.md#data-preparation).
- [Configure MELITE](docs/usage.md#configuration).
- [Run a full evaluation](docs/usage.md#melite-run).
- [Export a model](docs/usage.md#melite-export) and use the
  [Python prediction API](docs/usage.md#python-api).

## Workflow


### CLI Workflow

The command-line interface provides the canonical end-to-end MELITE workflow:

1. Register one or more prepared tabular datasets in a TOML configuration file.
2. Assign each dataset to classification or regression and choose the active
   models for the configured tasks.
3. Run `melite run` to generate evaluation evidence and selected results.
4. Inspect the classification and/or regression artifact family generated for
   the registered tasks.
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

MELITE supports the following 11 classifier keys. The public names and model
families follow the canonical model catalog.

| Key | Public name | Model family | Tunable | Active by default |
|---|---|---|---|---|
| `svc` | SVC | Individual — support-vector machine | Yes | Yes |
| `lr` | LogisticRegression | Individual — logistic regression | Yes | No |
| `gnb` | GaussianNB | Individual — Gaussian naive Bayes | Yes | No |
| `knn` | KNeighborsClassifier | Individual — nearest neighbors | Yes | No |
| `dt` | DecisionTreeClassifier | Individual — decision tree | Yes | No |
| `rf` | RandomForestClassifier | Ensemble — bagging-derived | Yes | Yes |
| `ada` | AdaBoostClassifier | Ensemble — adaptive boosting | Yes | No |
| `xgb` | XGBClassifier | Ensemble — gradient boosting | Yes | Yes |
| `lgbm` | LGBMClassifier | Ensemble — gradient boosting | Yes | No |
| `catboost` | CatBoostClassifier | Ensemble — gradient boosting | Yes | No |
| `stack` | StackingClassifier | Ensemble — stacking | No | No |

The ten individual and ensemble classifiers other than `stack` have
model-specific Optuna search spaces. `stack` is a fixed, opt-in composite and
is fitted without a separate Optuna search.

The default configuration is:

```toml
[classifiers]
active = ["svc", "rf", "xgb"]
```

The `[classifiers].active` list accepts any supported subset in evaluation
order, including a single classifier. Classifiers other than `svc`, `rf`, and
`xgb` remain inactive unless explicitly selected.

Every exported classifier artifact supports both `predict()` and
`predict_proba()`. SVC remains one classifier family: MELITE preserves direct
SVC class predictions and its decision function while supplying calibrated
probabilities internally for exported inference. The other supported
classifiers provide their estimator-native probability outputs. SVC,
LogisticRegression, and KNeighborsClassifier use scaling pipelines;
GaussianNB, DecisionTreeClassifier, and the standalone bagging and boosting
classifiers operate directly on the supplied numeric features. The opt-in
StackingClassifier combines SVC, RandomForestClassifier, and XGBClassifier as
base estimators with LogisticRegression as the final estimator and uses
probability-based base-estimator outputs.

CatBoost additionally supports native categorical predictors for CSV
classification datasets. Declare those columns with `categorical_features` and
select only `catboost`; undeclared predictors must remain numeric. With no
`categorical_features` declaration, CatBoost retains the same numeric NumPy
input path as the other numeric classifiers.

Binary and multiclass classification use this same classifier inventory. For
multiclass artifacts, `classes_` retains the original labels and defines the
column order returned by `predict_proba()` and `melite.predict()`.

Tunable classifiers use Optuna with its TPE sampler for hyperparameter
optimization. The public `n_trials` setting explicitly declares the budget per
search, with a normal default of 100 trials. Searches are sequential and seeded
from the canonical `RANDOM_STATE`, supporting reproducible optimization under
the same configuration and software environment.

## Supported Regressors

MELITE supports the following four regressor keys:

| Key | Public name | Estimator pipeline | Parameter optimization |
|---|---|---|---|
| `linear` | LinearRegression | `LinearRegression` | No |
| `poly` | PolynomialRegression | `StandardScaler` → `PolynomialFeatures(degree=2, include_bias=False)` → `LinearRegression` | No |
| `ridge` | Ridge | `StandardScaler` → `Ridge` | Yes — `alpha` |
| `lasso` | Lasso | `StandardScaler` → `Lasso(max_iter=10000)` | Yes — `alpha` |

Ridge and Lasso optimize `alpha` over the continuous logarithmic range from
`1e-4` to `1e2`. PolynomialRegression has a fixed degree of two. Before fitting,
MELITE calculates the resulting feature count and rejects an expansion above
10,000 features, reporting both the original and expanded dimensions.

## Input Format

### Dataset Registry

Datasets are registered under user-defined `[datasets.<dataset_id>]` entries.
For example:

```toml
[datasets.morgan_r2_2048]
path = "data/morgan_r2_2048.npz"
label_path = "raw/labels.npy"
task = "classification"
family = "fingerprints"
method = "Morgan"
variant = "r2_2048"
description = "Morgan fingerprints, radius 2, 2048 bits"
```

The optional `task` field accepts exactly `classification` or `regression` and
defaults to `classification` when omitted. Regression requires a
one-dimensional, finite quantitative target and at least one model in
`[regressors].active`. Classification-only, regression-only, and mixed-task
registries are supported.

Here, `family` is optional **dataset metadata** used to describe the feature
representation. It is unrelated to the model selected or evaluated by MELITE.

Each dataset must define `path` and the label source required by its format:
`label_column` for CSV datasets or `label_path` for NPZ datasets.
Optional metadata fields such
as `family`, `method`, `variant`, `level`, and `description` are preserved for
traceability and do not trigger dataset-specific execution logic.

The legacy `[benchmark]` configuration section remains supported for backward
compatibility. New configurations should use the dataset registry.

Numeric CSV datasets omit `categorical_features`; every predictor must then be
numeric. CatBoost-only categorical or mixed input is configured explicitly:

```toml
[datasets.customer_segments]
path = "data/customer_segments.csv"
label_column = "segment"
categorical_features = ["region", "channel"]

[classifiers]
active = ["catboost"]

[catboost]
boosting_type = "Ordered"
```

`categorical_features` must be a non-empty list of unique predictor column
names in a CSV classification dataset; it cannot contain `label_column` and
requires `[classifiers].active = ["catboost"]`. Declared categorical values
must be non-empty strings, while all remaining predictors must be numeric.
Categorical-only input declares every predictor; mixed input declares only its
categorical predictors. `[catboost].boosting_type` accepts `"Plain"` or
`"Ordered"`; omitting it preserves the `"Plain"` behavior.

Exported categorical and mixed CatBoost artifacts retain their feature names,
column order, and categorical schema. Inference with those artifacts requires
a pandas `DataFrame` with exactly the same columns in the same order and
compatible categorical/numeric types; missing, extra, or reordered columns are
rejected. The schema is stored in the artifact, so the original TOML is not
needed for inference. Numeric artifacts retain the existing two-dimensional
NumPy inference contract.

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

Classification datasets produce `results.txt`, `results.csv`,
`evaluations.csv`, `evaluation_folds.csv`, `optimization_searches.csv`,
`optimization_provenance.json`, `evaluation_class_folds.csv`,
`evaluation_class_summary.csv`, `classification_metric_provenance.json`, and a
dataset-level F1-macro evidence figure.
Regression datasets produce the separate `regression_results.txt`,
`regression_results.csv`, `regression_evaluations.csv`,
`regression_evaluation_folds.csv`, `regression_optimization_searches.csv`, and
`regression_optimization_provenance.json` family. Mixed-task runs produce both
families. Regression currently produces no F1-macro figure.

The classification artifacts have distinct roles:

- `results.txt` — human-readable summary of the selected results.
- `results.csv` — selected classifier result for each dataset and the persisted
  parameter source used by `melite export`.
- `evaluations.csv` — aggregate evaluation evidence for every active classifier.
- `evaluation_folds.csv` — outer-CV evidence for every dataset, classifier, and
  outer split.
- `evaluation_class_folds.csv` — held-out precision, recall, F1-score, and
  support for every original class in every outer split.
- `evaluation_class_summary.csv` — per-class means and population standard
  deviations over outer folds, including per-fold support summaries.
- `optimization_searches.csv` — one row per completed outer or final
  optimization search.
- `optimization_provenance.json` — the effective optimization and evaluation
  contract for the run.
- `classification_metric_provenance.json` — classification metric definitions
  and ordered original class identities for each executed dataset.
- `figures/evaluation_f1_macro_<dataset>.png` — visualization of the outer-CV
  F1-macro evidence used for classifier selection.
- `Model_<classifier>_<dataset>.pkl` — final full-data fitted model created by
  `melite export`.

The corresponding regression CSV files preserve selected results, aggregate
and fold-level RMSE/MAE/R² evidence, completed Ridge/Lasso searches, and the
effective regression evaluation and optimization contract. Exported regression
models use the same `Model_<model>_<dataset>.pkl` naming pattern.

See the [Output Data Contract](docs/usage.md#output-data-contract) for the
column-level schemas and semantics.

The evaluation figure is generated from already-computed outer-CV evidence. It
does not trigger additional fitting, tuning, cross-validation, or selection.


## Configuration

MELITE uses TOML configuration files to keep execution choices explicit and
reproducible. Configuration controls, among other settings:

- registered datasets and their metadata;
- dataset prediction tasks;
- active classifiers and regressors;
- random state;
- inner and outer cross-validation settings;
- optimization trial budget (`n_trials`);
- input and output paths.

Use `--config` to supply a project-specific configuration:

```bash
melite run --config my_config.toml
melite export --config my_config.toml --row 0
melite export --config my_config.toml --csv output/regression_results.csv --row 0
```

A complete configuration template is available at
`examples/melite_run.toml`.

Smoke mode can be requested independently from the configuration:

```bash
melite run --smoke --config my_config.toml
```

See the full configuration reference in the project documentation.

## Development

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
Contreras-Torres, F. F., & Murrieta, A. C. (2026). MELITE — Multi-Model Classifier Evaluator. Zenodo. https://doi.org/10.5281/zenodo.20382752
```

## License

This project is licensed under the terms of the
[GNU Lesser General Public License v3.0 or later](LICENSE).
SPDX identifier: `LGPL-3.0-or-later`.
