# Usage

MELITE supports supervised classification and regression on prepared tabular
datasets. The established input path is numeric, with native categorical and
mixed CSV classification available when CatBoost is the sole active
classifier. The bundled quick-start project remains a numeric classification
example.

## Installation

Install MELITE in a supported Python environment:

```bash
python -m pip install melite
```

Verify the installation:

```bash
melite --version
```

## Quick Start

MELITE includes a bundled synthetic numeric CSV classification dataset and
example configuration for verifying the installed classification workflow
without cloning the repository or preparing input files.

Create the example project in the current directory:

```bash
melite example
```

This creates:

```text
melite_example/
├── config.toml
└── data/
    └── sample_tabular.csv
```

The bundled configuration uses paths rooted at `melite_example/`. Run the
following command from the directory that contains `melite_example/`:

```bash
melite run --smoke --config melite_example/config.toml
```

A successful run creates `melite_example/output/` with `results.txt`,
`results.csv`, `evaluations.csv`, `evaluation_folds.csv`,
`evaluation_class_folds.csv`, `evaluation_class_summary.csv`,
`optimization_searches.csv`, `optimization_provenance.json`,
`classification_metric_provenance.json`, and the F1-macro evidence figure. For
the bundled example, `results.csv` contains one selected result for
`sample_tabular`.

Smoke mode uses reduced evaluation settings to verify the execution workflow.
The resulting evaluation evidence is explicitly marked as smoke and is not
intended for final classifier selection or model export.

## Data Preparation

MELITE evaluates prepared tabular feature representations. Its established
contract is numeric; CSV classification datasets evaluated exclusively with
CatBoost may additionally declare categorical predictors. Users are responsible
for supplying analysis-ready features compatible with the selected model.

### CSV

CSV is the recommended low-friction registered input path. A single table
contains both feature columns and labels, and the configured `label_column`
identifies the target column. All remaining columns become `X` in exactly the
order in which they appear in the CSV. Labels may be categorical.

When `categorical_features` is omitted, all feature columns must be numeric and
MELITE preserves the existing NumPy loading path. This is the input contract
for every regressor and for all classifiers on numeric datasets.

For a CSV classification dataset evaluated only with CatBoost,
`categorical_features` may explicitly list categorical predictor columns.
Declared columns are loaded as strings and must contain only non-empty string
values; all remaining predictors must be numeric. A categorical-only dataset
declares every predictor, while a mixed dataset declares only its categorical
predictors. MELITE preserves the DataFrame column names and order so CatBoost
can process categories natively inside each training fold.

Numeric artifacts do not persist semantic feature names, so later inference
must preserve the same representation, feature count, and order. Exported
categorical or mixed CatBoost artifacts do persist their DataFrame schema; their
inference requirements are described under [Python API](#python-api).

A column whose name starts with `Unnamed:` is rejected because it commonly
represents an accidentally exported pandas index. When creating a CSV with
pandas, omit that index explicitly:

```python
dataframe.to_csv("data/features.csv", index=False)
```

MELITE does not:

- impute missing values;
- provide general categorical preprocessing or encoding for other models;
- infer identifier columns;
- select feature columns automatically;
- reorder feature columns;
- perform feature engineering.

Registered input validation does not reject non-finite numeric values merely
at the loader boundary. This does not imply that every downstream classifier
accepts `NaN` or positive or negative infinity.

### NPZ

NPZ remains a supported alternative. The archive contains an explicit `X`
array, and the configured `label_path` supplies the authoritative `y` vector.
When the NPZ also contains an embedded `y`, MELITE uses it only as a
consistency check against `label_path`.

Relative paths for both formats are resolved from the directory in which
`melite` is executed.

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
is fitted without a separate Optuna search. The default active classifiers are:

```toml
[classifiers]
active = ["svc", "rf", "xgb"]
```

The `[classifiers].active` list accepts any supported subset in evaluation
order, including a single classifier. Classifiers other than `svc`, `rf`, and
`xgb` are opt-in.

MELITE determines binary or multiclass classification from the number of
distinct target labels: two labels select the binary metric contract, while
three or more select the multiclass contract. The target is not recoded in
persisted evidence or user-facing predictions; original class identities are
retained.

Standalone SVC evaluation uses a `StandardScaler` → `SVC` pipeline and retains
its decision function without probability fitting during evaluation.

LogisticRegression and KNeighborsClassifier also use `StandardScaler`
pipelines. GaussianNB, DecisionTreeClassifier, and the boosting and
bagging-derived classifiers operate directly on the supplied numeric features.

StackingClassifier is stable but opt-in. It combines SVC,
RandomForestClassifier, and XGBClassifier as base estimators with
LogisticRegression as the final estimator and uses probability-based
base-estimator outputs.

### Prediction and probabilities

Classifier artifacts exported for all 11 supported keys provide both
`predict()` and `predict_proba()`.

For binary and multiclass artifacts, the fitted model's `classes_` attribute is
the authoritative class order. Column `i` of `predict_proba()` and of the
`probabilities` array returned by `melite.predict()` corresponds to
`classes_[i]`.

For an exported SVC, class predictions and the decision function come directly
from SVC, while MELITE supplies calibrated probabilities through an internal
layer. SVC remains a single public classifier family and is selected normally:

```toml
[classifiers]
active = ["svc"]
```

The other supported classifiers provide their estimator-native probabilities.
The exported StackingClassifier also provides predictions and probabilities;
internally, it uses probability-based outputs from its SVC,
RandomForestClassifier, and XGBClassifier base estimators before the final
LogisticRegression estimator. It can be selected by itself:

```toml
[classifiers]
active = ["stack"]
```

CatBoost supports numeric, categorical-only, and mixed numeric/categorical CSV
classification inputs. Native categorical processing is opt-in through the
dataset's `categorical_features` declaration and remains inside each training
fold. It does not add categorical preprocessing to any other classifier.

Hyperparameter search spaces for tunable classifiers are defined by MELITE. User
configuration determines which supported classifiers are active rather than
defining individual hyperparameter search spaces.

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

## Configuration

MELITE reads its default settings from `melite/config_default.toml`. A user
TOML file overrides only the settings that need to change.

A complete configuration template for a normal run is available at
`examples/melite_run.toml`.

Relative dataset, label, and output paths are resolved from the working
directory in which `melite` is executed, not from the location of the TOML
file.

For modern registered datasets, the actual dataset file is determined by
`[datasets.*].path`. `[paths].input` remains part of MELITE's historical path
contract and is not used to locate a registered CSV dataset. The bundled CSV
example deliberately points both `input` and `dataset` to
`melite_example/data/` so it does not create an unused `raw/` directory.

### Minimal Configuration

Each registered dataset may set `task` to exactly `classification` or
`regression`. When `task` is omitted, MELITE uses `classification`, preserving
the historical behavior and the configuration produced by `melite example`.

A minimal classification-only configuration is:

```toml
[paths]
output = "output/"

[datasets.tabular_a]
path = "data/features.csv"
label_column = "label"
description = "Prepared numeric feature matrix."
# task is omitted, so this dataset is classified.

[classifiers]
active = ["svc", "rf", "xgb"]
```

For a concise reproducible multiclass run, prepare a numeric CSV whose
`class_label` column contains the original labels `1`, `2`, and `3`, then use:

```toml
[paths]
output = "output/"

[benchmark]
random_state = 42

[datasets.three_class]
path = "data/three_class.csv"
label_column = "class_label"
task = "classification"

[cv]
n_splits = 2
n_repeats = 1
inner_n_splits = 2

[classifiers]
active = ["lr", "xgb", "stack"]

[optimization]
n_trials = 5
```

Run it with `melite run --config config.toml`. The small fold and trial counts
make this suitable for workflow verification, not for a final scientific model
comparison.

A minimal regression-only configuration is:

```toml
[paths]
output = "output/"

[datasets.quantitative_a]
path = "data/quantitative.csv"
label_column = "target"
task = "regression"

[regressors]
active = ["ridge"]
```

The regression target must be a one-dimensional, finite, real numerical vector
representing the quantitative response. A configuration containing any
regression dataset must select at least one of `linear`, `poly`, `ridge`, or
`lasso` in `[regressors].active`. The default active-regressor list is empty, so
classification-only configurations remain backward compatible.

A mixed-task configuration registers both kinds of dataset and activates both
model groups:

```toml
[paths]
output = "output/"

[datasets.classification_a]
path = "data/classification.csv"
label_column = "class_label"

[datasets.regression_a]
path = "data/regression.csv"
label_column = "quantitative_target"
task = "regression"

[classifiers]
active = ["svc", "rf", "xgb"]

[regressors]
active = ["ridge", "lasso"]
```

Classification-only, regression-only, and mixed-task runs use the same
`melite run` command. In a mixed run, each dataset is dispatched according to
its own normalized `task` value.

Run MELITE from the project directory expected by those paths:

```bash
melite run --config my_config.toml
```

### Dataset Registry

Each feature matrix is registered under a user-defined
`[datasets.<dataset_id>]` entry.

CSV example:

```toml
[datasets.my_dataset]
path = "data/my_dataset.csv"
label_column = "label"
task = "classification"
family = "tabular"
```

For CSV, `path` and `label_column` are required, and `label_path` is forbidden.
All columns except `label_column` become `X` in their existing column order.

NPZ example:

```toml
[datasets.my_dataset]
path = "data/my_dataset.npz"
label_path = "raw/labels.npy"
family = "descriptors"
```

For NPZ, `path` and `label_path` are required, and `label_column` is forbidden.
Specifying both label fields is invalid. The supported registered extensions
are exactly `.csv` and `.npz`, matched case-insensitively.

The accepted fields inside each `[datasets.*]` entry are exactly:

- `path`;
- `label_path`;
- `label_column`;
- `task`;
- `categorical_features`;
- `family`;
- `method`;
- `variant`;
- `level`;
- `description`.

Unknown fields are rejected rather than silently ignored.

`task` is optional and accepts only the lowercase values `classification` and
`regression`. Omitting it selects classification. The label source is chosen by
file format, not by task: CSV uses `label_column`, while NPZ uses `label_path`.

Optional metadata fields:

- `family`;
- `method`;
- `variant`;
- `level`;
- `description`.

These metadata fields are preserved primarily for reporting and traceability. Legacy dimensionality metadata may also be interpreted to preserve historical `reduction_type` compatibility.
Do not invent a semantic value merely
to populate optional metadata.

The `family` field describes the feature representation as dataset metadata. It
is unrelated to the model selected or evaluated by MELITE.

Each `dataset_id` identifies one concrete feature matrix evaluated as an
independent dataset.

`categorical_features` is optional. When present, it must be a non-empty list
of unique, non-empty column names, must not include `label_column`, and is valid
only for CSV classification with `[classifiers].active = ["catboost"]`. When it
is omitted, the existing numeric behavior is unchanged.

### CatBoost Configuration

The following example uses mixed input: `age` and `balance` are numeric, while
`region` and `channel` are categorical. A categorical-only dataset uses the same
form but lists every predictor in `categorical_features`.

```toml
[datasets.customer_segments]
path = "data/customer_segments.csv"
label_column = "segment"
task = "classification"
categorical_features = ["region", "channel"]

[classifiers]
active = ["catboost"]

[catboost]
boosting_type = "Ordered"
```

`[catboost].boosting_type` is independent of whether categorical features are
declared and accepts exactly `"Plain"` or `"Ordered"`. Omitting the setting
preserves the existing `"Plain"` behavior. A numeric CatBoost dataset therefore
may select either boosting type without declaring `categorical_features`.

### Evaluation Settings

The normal cross-validation defaults are:

```toml
[cv]
n_splits = 5
n_repeats = 3
inner_n_splits = 3
```

For each dataset and active model, the outer evaluation contains
`n_splits × n_repeats` held-out splits. With the defaults, this produces 15
outer evaluations per model.

Classification uses `RepeatedStratifiedKFold` for outer evaluation and shuffled
`StratifiedKFold` for the inner searches of tunable classifiers. F1-macro is
maximized during optimization and the highest mean outer F1-macro selects the
classifier. Accuracy and ROC-AUC are additional reported metrics. Stacking is
not optimized separately, but uses `inner_n_splits` for its internal
out-of-fold predictions.

Regression uses `RepeatedKFold` for outer evaluation and shuffled `KFold` for
the inner searches of Ridge and Lasso. RMSE is minimized during optimization
and the lowest mean outer RMSE selects the regressor. RMSE and MAE are reported
as nonnegative errors; R² is also reported. LinearRegression and
PolynomialRegression have no hyperparameter search and are fitted directly.

For either task, `n_splits` and `n_repeats` control the outer strategy, while
`inner_n_splits` controls inner optimization. Each tunable model receives a
separate inner search within each outer training split. The corresponding
held-out outer fold supplies evaluation evidence and is not used by that
search.

The public optimization budget for tunable models is configured as:

```toml
[optimization]
n_trials = 100
```

`n_trials` is the trial budget per Optuna search and is the primary public
control for optimization effort. The normal default is 100. Smoke mode instead
uses an internal 5-trial budget. It applies to tunable classifiers and to Ridge
and Lasso; direct-fitting models do not start an Optuna study.

The global random seed is configured as:

```toml
[benchmark]
random_state = 42
```

`random_state` remains an active configuration setting. It is located under
`[benchmark]` for historical compatibility and is not part of the legacy
reduction settings described below.

Smoke mode uses its own reduced settings:

```toml
[cv_smoke]
n_splits = 3
n_repeats = 1
inner_n_splits = 2
```

Use them with:

```bash
melite run --smoke --config my_config.toml
```

Selection compares aggregate outer evidence in configured model order. Exact
ties retain the first configured model: classification maximizes mean F1-macro,
whereas regression minimizes mean RMSE. After selection, a tunable winner
receives a fresh optimization using all available data with the configured
inner-CV strategy. MELITE then fits the winner on all data and records those
final parameters in the task-specific results CSV. A non-tunable winner is
fitted directly and records an empty parameter mapping.

Outer evaluation evidence, inner optimization results, final parameter
selection, and full-data fitting are distinct stages. The resulting evidence
characterizes the configured evaluation procedure; it does not establish
external predictive generalizability.

### Legacy Configuration

Legacy reduction-oriented settings under `[benchmark]` remain supported for
backward compatibility. When `[datasets]` is absent, those legacy reduction
settings are normalized into equivalent dataset entries.

Other active settings currently stored under `[benchmark]`, such as
`random_state`, are not legacy reduction settings.

New dataset definitions should use the dataset registry.

## Command-Line Interface

Display the main CLI help:

```bash
melite --help
```

Command-specific help is available with:

```bash
melite example --help
melite run --help
melite export --help
```

### `melite example`

Create the bundled example project in the current directory:

```bash
melite example
```

The command creates `./melite_example/` with a synthetic numeric CSV dataset
and example configuration:

```text
melite_example/config.toml
melite_example/data/sample_tabular.csv
```

The `output/` directory is created only when MELITE executes the workflow.

The generated `config.toml` assumes that MELITE is executed from the directory
containing `melite_example/`.

If `melite_example/` already exists, MELITE exits without overwriting or
merging its contents.

### `melite run`

Run an evaluation with a project-specific configuration:

```bash
melite run --config my_config.toml
```

Enable verbose logging:

```bash
melite run --verbose --config my_config.toml
```

Run with reduced smoke settings:

```bash
melite run --smoke --config my_config.toml
```

Evaluation time depends on the number and tasks of registered datasets, the
active classifiers and regressors, cross-validation settings, and the
configured `n_trials` budget for tunable models. A normal evaluation can
therefore take substantially longer than a smoke run.

### `melite export`

After a normal classification evaluation, inspect `results.csv` and export the
selected result for the desired dataset.

`--row` is the zero-based row index in `results.csv`. With multiple datasets,
identify the corresponding row in `results.csv` before exporting it.

Export a specific row:

```bash
melite export --config my_config.toml --row 0
```

When `--csv` and `--outdir` are omitted, MELITE reads `results.csv` from the
configured output directory and writes the `.pkl` artifact to that same
directory.

Explicit paths can be supplied when needed:

```bash
melite export \
  --row 0 \
  --csv output/results.csv \
  --outdir output/
```

`melite export` reconstructs the selected classifier and fits the final model
on all available data. It uses the classifier parameters persisted in
`results.csv` by `melite run`. It performs no additional hyperparameter search,
cross-validation, or classifier selection.

For CatBoost, `results.csv` also persists the effective boosting type and
categorical feature declaration used during evaluation. Export verifies these
values against the supplied configuration before fitting, so it cannot silently
change the evaluated CatBoost contract. The exported categorical or mixed model
stores its feature schema in the artifact; the original TOML is not required
for inference. Legacy v0.3.3 CatBoost result files without this persisted field
remain exportable only for numeric input with `Plain` boosting.

Regression export requires explicit selection of the regression results file:

```powershell
melite export --config config.toml --csv output/regression_results.csv --row 0
```

`--row` is also zero-based for `regression_results.csv`. MELITE validates the
regression schema, reconstructs the selected regressor with the final
parameters recorded by `melite run`, fits it on all available data, and writes
the `.pkl` artifact. Export does not repeat Ridge or Lasso optimization and does
not perform another evaluation or model-selection step.

Smoke results are blocked from export by default. The `--force` option exists
for deliberate testing or diagnostic use with either task. For example:

```powershell
melite export --force --config config.toml --csv output/regression_results.csv --row 0
```

## Outputs

A MELITE run writes only the artifact family required by the tasks that were
executed. A mixed-task run can produce:

```text
output/
├── results.txt
├── results.csv
├── evaluations.csv
├── evaluation_folds.csv
├── evaluation_class_folds.csv
├── evaluation_class_summary.csv
├── optimization_searches.csv
├── optimization_provenance.json
├── classification_metric_provenance.json
├── figures/
│   └── evaluation_f1_macro_<dataset>.png
├── regression_results.txt
├── regression_results.csv
├── regression_evaluations.csv
├── regression_evaluation_folds.csv
├── regression_optimization_searches.csv
├── regression_optimization_provenance.json
└── Model_<model>_<dataset>.pkl
```

The classification artifacts have distinct roles:

- `results.txt` — human-readable summary of selected results;
- `results.csv` — selected classifier result for each dataset and the sole
  operational persisted parameter input used by `melite export`;
- `evaluations.csv` — aggregate evaluation evidence for every active
  classifier;
- `evaluation_folds.csv` — outer-CV evidence for every dataset, classifier, and
  outer split;
- `evaluation_class_folds.csv` — held-out precision, recall, F1-score, and
  support for every original class in every outer split;
- `evaluation_class_summary.csv` — class-level means and population standard
  deviations over outer folds, including per-fold support summaries;
- `optimization_searches.csv` — one persisted row per optimization search,
  covering outer and optional final optimization evidence;
- `optimization_provenance.json` — effective optimization and evaluation
  contract for the run;
- `classification_metric_provenance.json` — metric definitions and ordered
  original class identities for every executed classification dataset;
- `figures/evaluation_f1_macro_<dataset>.png` — visualization of the outer-CV
  F1-macro evidence used for classifier selection;
- `Model_<classifier>_<dataset>.pkl` — final full-data fitted model created by
  `melite export`.

The separate regression family contains:

- `regression_results.txt` — human-readable selected-regressor summaries;
- `regression_results.csv` — one selected regressor per regression dataset and
  the operational parameter source for regression export;
- `regression_evaluations.csv` — aggregate RMSE, MAE, and R² evidence for every
  active regressor;
- `regression_evaluation_folds.csv` — held-out RMSE, MAE, and R² for every
  regressor and outer split;
- `regression_optimization_searches.csv` — completed outer and final Ridge or
  Lasso optimization searches;
- `regression_optimization_provenance.json` — the effective regression
  evaluation, optimization, and search-space contract.

Classification-only runs do not write regression artifacts, regression-only
runs do not write classification artifacts, and mixed-task runs write both
families. Regression currently does not produce an F1-macro or other evaluation
figure.

`results.csv`, `evaluations.csv`, `evaluation_folds.csv`,
`evaluation_class_folds.csv`, `evaluation_class_summary.csv`, and
`optimization_searches.csv` include a `smoke` column identifying whether their
evidence was produced in smoke mode. `optimization_provenance.json` records the
same condition in its top-level `smoke` key rather than a CSV column.

The four regression CSV files use the same `smoke` convention, and
`regression_optimization_provenance.json` records it as a top-level key.

### Output Data Contract

The classification and regression CSV files and their task-specific provenance
JSON artifacts have package-versioned schemas.
MELITE does not embed an independent machine-readable `schema_version`. A
separate schema version should be reconsidered only if output schemas need to
evolve independently of the MELITE package version.

Optional metadata that is absent is serialized as an empty CSV field.
`reduction_type` is retained for legacy compatibility and is normally empty
for modern registered datasets. `smoke` identifies rows produced under reduced
smoke-mode evaluation settings.

#### `results.csv`

`results.csv` contains one row per evaluated dataset, containing only the
classifier selected for that dataset. It is both a public selected-result
artifact and the persisted interface used by `melite export` to identify and
reconstruct a selected result produced by `melite run`.

<!-- melite-schema:results.csv -->

| Column | Meaning |
|---|---|
| `dataset` | User-defined registered dataset identifier. |
| `family` | Optional dataset-family metadata preserved for reporting and traceability. |
| `method` | Optional dataset-method metadata preserved for reporting and traceability. |
| `variant` | Optional dataset-variant metadata preserved for reporting and traceability. |
| `level` | Optional dataset-level metadata preserved for reporting and traceability. |
| `description` | Optional dataset description preserved for reporting and traceability. |
| `reduction_type` | Legacy compatibility field; normally empty for modern registered datasets. |
| `classifier_name` | Name of the classifier selected for the dataset. |
| `parameters` | Parameters recorded for the selected result produced by `melite run`. |
| `catboost_config` | Effective CatBoost `boosting_type` and ordered `categorical_features` serialized as JSON for a selected CatBoost result; empty for other classifiers. |
| `f1_macro` | Mean outer-CV F1-macro for the selected classifier. |
| `f1_std` | Population standard deviation of outer-CV F1-macro. |
| `accuracy` | Mean outer-CV accuracy for the selected classifier. |
| `acc_std` | Population standard deviation of outer-CV accuracy. |
| `auc_roc` | Mean outer-CV AUC-ROC when available. |
| `auc_std` | Population standard deviation of outer-CV AUC-ROC when available. |
| `smoke` | Whether the row was produced in smoke mode. |

The metric values in this selected-result summary are written rounded to four
decimal places by the current `melite run` workflow. For tunable classifiers,
`parameters` records the hyperparameters selected by the final full-data search
performed during `melite run`. `melite export` uses those persisted parameters
to reconstruct and fit the final model; it does not perform another
hyperparameter search.

#### `evaluations.csv`

`evaluations.csv` contains one row per dataset × evaluated classifier. It
preserves aggregate evaluation evidence for every active classifier, not only
the winner. `selected` identifies the classifier selected for that dataset.

<!-- melite-schema:evaluations.csv -->

| Column | Meaning |
|---|---|
| `dataset` | User-defined registered dataset identifier. |
| `family` | Optional dataset-family metadata preserved for reporting and traceability. |
| `method` | Optional dataset-method metadata preserved for reporting and traceability. |
| `variant` | Optional dataset-variant metadata preserved for reporting and traceability. |
| `level` | Optional dataset-level metadata preserved for reporting and traceability. |
| `description` | Optional dataset description preserved for reporting and traceability. |
| `reduction_type` | Legacy compatibility field; normally empty for modern registered datasets. |
| `classifier_name` | Name of the evaluated classifier. |
| `f1_macro` | Mean outer-CV F1-macro for the evaluated classifier. |
| `f1_std` | Population standard deviation of outer-CV F1-macro. |
| `accuracy` | Mean outer-CV accuracy for the evaluated classifier. |
| `acc_std` | Population standard deviation of outer-CV accuracy. |
| `auc_roc` | Mean outer-CV AUC-ROC when available. |
| `auc_std` | Population standard deviation of outer-CV AUC-ROC when available. |
| `selected` | Whether this classifier was selected for the dataset. |
| `smoke` | Whether the row was produced in smoke mode. |

Aggregate metric values are persisted from the evaluation evidence without
the four-decimal summary rounding applied when `results.csv` rows are
constructed. Metadata, `reduction_type`, and `smoke` have the same semantics
described above.

#### `evaluation_folds.csv`

`evaluation_folds.csv` contains one row per dataset × evaluated classifier ×
outer-CV split. It is the most granular persisted evaluation evidence used to
reconstruct and inspect the aggregate comparison.

<!-- melite-schema:evaluation_folds.csv -->

| Column | Meaning |
|---|---|
| `dataset` | User-defined registered dataset identifier. |
| `family` | Optional dataset-family metadata preserved for reporting and traceability. |
| `method` | Optional dataset-method metadata preserved for reporting and traceability. |
| `variant` | Optional dataset-variant metadata preserved for reporting and traceability. |
| `level` | Optional dataset-level metadata preserved for reporting and traceability. |
| `description` | Optional dataset description preserved for reporting and traceability. |
| `reduction_type` | Legacy compatibility field; normally empty for modern registered datasets. |
| `classifier_name` | Name of the evaluated classifier. |
| `outer_split` | Sequential identifier of the preserved outer evaluation split. |
| `outer_repeat` | Outer repeated-CV repeat identifier. |
| `outer_fold` | Fold identifier within the repeat. |
| `f1_macro` | Held-out F1-macro for this outer split. |
| `accuracy` | Held-out accuracy for this outer split. |
| `auc_roc` | Held-out AUC-ROC for this outer split when available. |
| `selected` | Whether this classifier was selected globally for the dataset after aggregate evaluation. |
| `smoke` | Whether the evidence was generated under smoke-mode settings. |

The `selected` field does not mean that classifier selection occurred
independently within that fold.

#### `evaluation_class_folds.csv`

`evaluation_class_folds.csv` contains one row per dataset × evaluated
classifier × outer split × original class. Metrics are computed only from that
split's held-out labels and predictions. Every original class is retained;
undefined precision, recall, or F1-score is reported as zero
(`zero_division=0`).

<!-- melite-schema:evaluation_class_folds.csv -->

| Column | Meaning |
|---|---|
| `dataset` | User-defined registered dataset identifier. |
| `family` | Optional dataset-family metadata preserved for reporting and traceability. |
| `method` | Optional dataset-method metadata preserved for reporting and traceability. |
| `variant` | Optional dataset-variant metadata preserved for reporting and traceability. |
| `level` | Optional dataset-level metadata preserved for reporting and traceability. |
| `description` | Optional dataset description preserved for reporting and traceability. |
| `reduction_type` | Legacy compatibility field; normally empty for modern registered datasets. |
| `classifier_name` | Name of the evaluated classifier. |
| `outer_split` | Sequential identifier of the preserved outer evaluation split. |
| `outer_repeat` | Outer repeated-CV repeat identifier. |
| `outer_fold` | Fold identifier within the repeat. |
| `class_label` | Original class identity in deterministic scoring order. |
| `precision` | Held-out precision for this class and outer split. |
| `recall` | Held-out recall for this class and outer split. |
| `f1_score` | Held-out F1-score for this class and outer split. |
| `support` | Number of held-out observations of this class in the outer split. |
| `selected` | Whether this classifier was selected globally for the dataset. |
| `smoke` | Whether the evidence was generated under smoke-mode settings. |

Within one dataset, classifier, and outer split, class supports sum to the
held-out fold size and F1-macro is the arithmetic mean of the class-level
F1-scores.

#### `evaluation_class_summary.csv`

`evaluation_class_summary.csv` summarizes each dataset × classifier × class
over outer folds. It does not pool predictions across repeats or present
repeated observations as an independent test population.

<!-- melite-schema:evaluation_class_summary.csv -->

| Column | Meaning |
|---|---|
| `dataset` | User-defined registered dataset identifier. |
| `family` | Optional dataset-family metadata preserved for reporting and traceability. |
| `method` | Optional dataset-method metadata preserved for reporting and traceability. |
| `variant` | Optional dataset-variant metadata preserved for reporting and traceability. |
| `level` | Optional dataset-level metadata preserved for reporting and traceability. |
| `description` | Optional dataset description preserved for reporting and traceability. |
| `reduction_type` | Legacy compatibility field; normally empty for modern registered datasets. |
| `classifier_name` | Name of the evaluated classifier. |
| `class_label` | Original class identity in deterministic scoring order. |
| `precision_mean` | Arithmetic mean of the class precision values over outer folds. |
| `precision_std` | Population standard deviation (`ddof=0`) of class precision over outer folds. |
| `recall_mean` | Arithmetic mean of the class recall values over outer folds. |
| `recall_std` | Population standard deviation (`ddof=0`) of class recall over outer folds. |
| `f1_score_mean` | Arithmetic mean of the class F1-scores over outer folds. |
| `f1_score_std` | Population standard deviation (`ddof=0`) of class F1-score over outer folds. |
| `support_mean` | Mean class support per outer fold; not a pooled repeated-sample total. |
| `support_std` | Population standard deviation (`ddof=0`) of class support over outer folds. |
| `n_outer_folds` | Number of outer-fold records contributing to the summary. |
| `selected` | Whether this classifier was selected globally for the dataset. |
| `smoke` | Whether the evidence was generated under smoke-mode settings. |

#### `optimization_searches.csv`

`optimization_searches.csv` contains one row per complete optimization search,
never one row per Optuna trial. It preserves both the inner optimization
associated with outer evaluation splits and the optional post-selection final
optimization.

<!-- melite-schema:optimization_searches.csv -->

| Column | Meaning |
|---|---|
| `dataset` | User-defined registered dataset identifier. |
| `family` | Optional dataset-family metadata preserved for reporting and traceability. |
| `method` | Optional dataset-method metadata preserved for reporting and traceability. |
| `variant` | Optional dataset-variant metadata preserved for reporting and traceability. |
| `level` | Optional dataset-level metadata preserved for reporting and traceability. |
| `description` | Optional dataset description preserved for reporting and traceability. |
| `reduction_type` | Legacy compatibility field; normally empty for modern registered datasets. |
| `classifier_name` | Name of the classifier whose optimization search completed. |
| `search_scope` | `outer` for an inner optimization associated with one outer evaluation split; `final` for post-selection full-data optimization. |
| `outer_split` | Sequential outer split identifier for outer rows; empty for final rows. |
| `outer_repeat` | Outer repeated-CV repeat identifier for outer rows; empty for final rows. |
| `outer_fold` | Fold identifier within the repeat for outer rows; empty for final rows. |
| `best_inner_f1_macro` | Best inner-CV F1-macro from the search, persisted at full precision without summary rounding. |
| `best_params` | Effective best parameters serialized as canonical JSON. |
| `n_trials_requested` | Number of trials requested for the search. |
| `n_trials_complete` | Number of successfully completed trials. |
| `n_trials_failed` | Number of failed trials. |
| `selected` | Dataset-level classifier-selection boolean for outer rows; empty and not applicable for final rows. |
| `smoke` | Whether the search was produced in smoke mode. |

Outer rows use `search_scope="outer"` and contain `True` or `False` in
`selected`. Final rows use `search_scope="final"`, leave all outer indices
empty, and leave `selected` empty because classifier selection has already
occurred. Generic dataframe readers may therefore infer a nullable or
object-like `selected` column; pandas dtype inference is not part of the CSV
contract. A Stack-only run creates a header-only `optimization_searches.csv`,
because zero optimization searches is a legitimate successful outcome.

`best_params` uses canonical JSON, while `results.csv.parameters` retains its
Python representation. For the selected tunable classifier, both encode the
same semantic parameter mapping. `results.csv` remains the sole operational
input used by `melite export`; `optimization_searches.csv` is methodological
evidence only.

The requested, complete, and failed counters summarize a search without
preserving trial ordering. When failed trials occur, those counters alone may
not establish retrospectively whether TPE entered model-based sampling. Full
trial traces, including individual configurations and objective values, are
deliberately not persisted.

#### `optimization_provenance.json`

`optimization_provenance.json` records the effective optimization and
evaluation contract under which the run evidence was produced.

<!-- melite-schema:optimization_provenance.json -->

| Key | Meaning |
|---|---|
| `melite_version` | MELITE package version governing the artifact contracts. |
| `optimization_backend` | Optimization backend name and runtime version; currently Optuna. |
| `smoke` | Whether the effective run used smoke settings. |
| `random_state` | Canonical MELITE random state used by evaluation and optimization. |
| `active_classifiers` | Active classifier keys in configured evaluation order. |
| `cv` | Effective outer split count, repeat count, and inner split count. |
| `optimization` | Effective trial budget and the complete fixed optimization policy. |
| `search_spaces` | Complete B1 search-space contracts for active tunable classifiers; active Stack is represented as `null`. |

The `optimization.policy` object records sampler, startup budget, smoke budget,
TPE options, pruning, storage, parallelism, direction, and objective. Only
active classifiers appear in `search_spaces`; inactive search spaces are not
included. The artifact deliberately excludes filesystem, dataset, user,
environment, estimator, Study, Trial, and per-trial runtime state. MELITE does
not introduce a separate formal JSON Schema or independent `schema_version`.

#### `classification_metric_provenance.json`

`classification_metric_provenance.json` records the effective classification
metric definitions and keeps each executed dataset's class identity separate.
It is methodological evidence and is not an input to `melite export`.

<!-- melite-schema:classification_metric_provenance.json -->

| Key | Meaning |
|---|---|
| `melite_version` | MELITE package version governing the artifact contract. |
| `prediction_task` | Fixed value `classification`. |
| `optimization_and_selection` | F1-macro objective and maximization direction. |
| `reported` | F1-macro, accuracy, binary ROC-AUC, and probability-based multiclass OvO macro ROC-AUC definitions. |
| `per_class` | Per-class metrics, held-out source, class order, `zero_division=0` convention, and fold-summary aggregation contract. |
| `datasets` | Ordered dataset records containing dataset id, class count, original labels in scoring order, evaluation mode, and applicable ROC-AUC definition. |

For multiclass datasets, `roc_auc_definition` is
`multiclass_ovo_macro_probability`; for binary datasets it is `binary`.
Probability columns follow the listed class-label order.

#### Regression artifact schemas

`regression_results.txt` is the human-readable regression report. It identifies
the selected regressor, final parameter mapping, aggregate outer-CV metrics,
effective CV strategies, random state, trial budget, and smoke status for each
executed regression dataset. It is not the machine-readable export interface;
that role belongs to `regression_results.csv`.

All four regression CSV files begin with this exact ordered identity prefix:

| Column | Meaning |
|---|---|
| `dataset` | User-defined registered dataset identifier. |
| `task` | Prediction task; always `regression` in these artifacts. |
| `family` | Optional dataset-family metadata. |
| `method` | Optional dataset-method metadata. |
| `variant` | Optional dataset-variant metadata. |
| `level` | Optional dataset-level metadata. |
| `description` | Optional dataset description. |
| `reduction_type` | Legacy compatibility field; normally empty for modern registered datasets. |
| `regressor_key` | Canonical regressor key: `linear`, `poly`, `ridge`, or `lasso`. |
| `regressor_name` | Canonical public regressor name. |

Optional metadata is serialized as an empty field when absent.

##### `regression_results.csv`

This file contains one row per regression dataset and only the selected
regressor. After the identity prefix, its exact columns are:

| Column | Meaning |
|---|---|
| `parameters` | Final full-data parameter mapping serialized as canonical JSON; empty mappings are written as `{}`. |
| `rmse` | Mean outer-CV RMSE for the selected regressor. |
| `rmse_std` | Population standard deviation of outer-CV RMSE. |
| `mae` | Mean outer-CV MAE for the selected regressor. |
| `mae_std` | Population standard deviation of outer-CV MAE. |
| `r2` | Mean outer-CV R² for the selected regressor. |
| `r2_std` | Population standard deviation of outer-CV R². |
| `smoke` | Whether the row was produced in smoke mode. |

For Ridge and Lasso, `parameters` contains the effective `ridge__alpha` or
`lasso__alpha` parameter selected by the fresh full-data search performed
during `melite run`. LinearRegression and PolynomialRegression record `{}`.
`melite export` uses this file and parameter mapping to reconstruct and fit the
final regressor without another optimization.

##### `regression_evaluations.csv`

This file contains one row per regression dataset × evaluated regressor. After
the identity prefix, its exact columns are:

| Column | Meaning |
|---|---|
| `rmse` | Mean outer-CV RMSE. |
| `rmse_std` | Population standard deviation of outer-CV RMSE. |
| `mae` | Mean outer-CV MAE. |
| `mae_std` | Population standard deviation of outer-CV MAE. |
| `r2` | Mean outer-CV R². |
| `r2_std` | Population standard deviation of outer-CV R². |
| `selected` | Whether this regressor was selected for the dataset. |
| `smoke` | Whether the row was produced in smoke mode. |

##### `regression_evaluation_folds.csv`

This file contains one row per regression dataset × evaluated regressor ×
outer split. After the identity prefix, its exact columns are:

| Column | Meaning |
|---|---|
| `outer_split` | Sequential identifier of the preserved outer split. |
| `outer_repeat` | Repeated-CV repeat identifier. |
| `outer_fold` | Fold identifier within the repeat. |
| `rmse` | Nonnegative held-out RMSE for this outer split. |
| `mae` | Nonnegative held-out MAE for this outer split. |
| `r2` | Held-out R² for this outer split. |
| `selected` | Whether this regressor was selected globally after aggregate evaluation. |
| `smoke` | Whether the evidence was produced in smoke mode. |

As with classification folds, `selected` does not indicate independent
selection within an outer fold.

##### `regression_optimization_searches.csv`

This file contains one row per completed Ridge or Lasso optimization search,
not one row per Optuna trial. After the identity prefix, its exact columns are:

| Column | Meaning |
|---|---|
| `search_scope` | `outer` for an inner search within one outer training split; `final` for the fresh full-data search after selection. |
| `outer_split` | Sequential outer split identifier for outer rows; empty for final rows. |
| `outer_repeat` | Outer repeat identifier for outer rows; empty for final rows. |
| `outer_fold` | Fold identifier for outer rows; empty for final rows. |
| `best_inner_rmse` | Lowest mean inner-CV RMSE found by the search. |
| `best_params` | Effective best parameters serialized as canonical JSON. |
| `n_trials_requested` | Number of Optuna trials requested. |
| `n_trials_complete` | Number of successfully completed trials. |
| `n_trials_failed` | Number of failed trials. |
| `selected` | Dataset-level selection status; final rows identify the selected tunable regressor. |
| `smoke` | Whether the search was produced in smoke mode. |

If a regression run evaluates only LinearRegression and/or
PolynomialRegression, the file is still created with its header and no search
rows.

##### `regression_optimization_provenance.json`

This file records the effective regression evaluation and optimization
contract with these exact top-level keys:

| Key | Meaning |
|---|---|
| `melite_version` | MELITE package version governing the artifact contract. |
| `task` | Fixed value `regression`. |
| `optimization_backend` | Optimization backend name and runtime version. |
| `smoke` | Whether the effective run used smoke settings. |
| `random_state` | Canonical MELITE random state. |
| `active_regressors` | Active regressor keys in configured evaluation order. |
| `cv` | Effective `RepeatedKFold` outer and shuffled `KFold` inner contracts. |
| `optimization` | RMSE objective, minimization direction, sampler settings, effective trial budget, pruning policy, and parallelism. |
| `search_spaces` | Effective search-space contracts; direct-fitting regressors are represented as `null`. |

## Python API

The public `melite.predict(model_path, X, return_proba=True)` function loads an
exported model artifact and runs inference on a new feature matrix. It
returns `predictions`, `probabilities`, `model_path`, and `n_samples`.

Classification example:

```python
import numpy as np
from melite import predict

X_new = np.load("data/new_samples.npz")["X"]

result = predict(
    "output/Model_SVC_my_dataset.pkl",
    X_new,
)

print(result["predictions"])
print(result["probabilities"])
```

`X_new` must be a two-dimensional NumPy array using the same feature
representation, feature order, and feature count expected by the exported
numeric model. This existing NumPy contract applies to numeric CatBoost models
as well as the other numeric classifiers and regressors.

Categorical and mixed CatBoost artifacts instead require a pandas DataFrame:

```python
import pandas as pd
from melite import predict

X_new = pd.DataFrame(
    {
        "age": [31.0, 47.0],
        "balance": [125.5, 810.0],
        "region": ["north", "south"],
        "channel": ["web", "store"],
    }
)

result = predict("output/Model_CatBoostClassifier_segments.pkl", X_new)
```

The DataFrame must contain exactly the feature names stored during export, in
the same order. Declared categorical columns must contain non-empty strings and
all other columns must be numeric. Missing, extra, reordered, or
type-incompatible columns are rejected explicitly. The exported artifact
contains this schema and does not need the original TOML at inference time.

All classifier artifacts exported by MELITE for the 11 supported keys provide
probability estimates. Exported regression artifacts return `probabilities` as
`None`. For classifier artifacts, that value is also `None` when
`return_proba=False`.

Classification predictions retain the original target labels. For binary and
multiclass artifacts, probability column `i` corresponds to `classes_[i]` on
the fitted model. MELITE validates numerical shape and feature count where the
estimator exposes it, but it does not infer semantic feature identity.

Regression example:

```python
import numpy as np
from melite import predict

X_new = np.array(
    [
        [0.2, 1.4, -0.3],
        [0.8, 0.1, 0.5],
    ],
    dtype=float,
)

result = predict("output/Model_Ridge_quantitative_a.pkl", X_new)
continuous_predictions = result["predictions"]
assert continuous_predictions.shape == (2,)
assert result["probabilities"] is None
```

Regression inference requires a two-dimensional NumPy array containing only
finite real numerical values. MELITE checks the feature count exposed by the
fitted regressor and returns one continuous quantitative prediction per input
row. It does not persist or validate feature names or semantic feature identity;
the caller must preserve the training representation and column order.

For the complete public Python interface, see the [API Reference](api.md).

## Evaluation Contract

For each registered dataset, MELITE follows a task-specific reproducible
evaluation contract:

1. `X` is a two-dimensional numeric matrix except for CSV classification
   datasets declaring native CatBoost categorical features, which retain a
   schema-preserving DataFrame. Classification uses a one-dimensional label
   target; regression requires a one-dimensional finite real quantitative
   target.
2. Classification determines binary or multiclass evaluation from target
   cardinality, then uses repeated stratified outer folds and stratified inner
   folds. Regression uses repeated `KFold` outer folds and shuffled `KFold`
   inner folds.
3. Every inner optimization is fitted only on the training portion of its
   corresponding outer split. Its best configuration is then evaluated on the
   held-out outer fold.
4. Classification selects the highest mean outer F1-macro and additionally
   reports accuracy and ROC-AUC. Binary ROC-AUC uses its established binary
   definition; multiclass ROC-AUC uses probabilities with One-vs-One macro
   averaging. Regression selects the lowest mean outer RMSE and additionally
   reports MAE and R².
5. Exact metric ties preserve configured model order. Aggregate and per-fold
   evidence is retained for all evaluated models. Classification also records
   held-out precision, recall, F1-score, and support for every original class;
   undefined values use `zero_division=0`.
6. After selection, Ridge, Lasso, and tunable classifiers receive a fresh
   full-data optimization using the configured inner strategy. LinearRegression,
   PolynomialRegression, and StackingClassifier are fitted directly.
7. Final parameters are persisted in `regression_results.csv` or `results.csv`.
   The selected model is fitted on all available data during `melite run`.
8. `melite export` reconstructs the selected model from those recorded final
   parameters and fits it on all available data to create the `.pkl` artifact.
   It does not repeat optimization, cross-validation, or model selection.
9. The classification F1-macro figure is generated from existing outer evidence
   without further fitting. No corresponding regression figure is currently
   produced.

Class-level summaries use arithmetic means and population standard deviations
(`ddof=0`) over outer folds. They report per-fold support mean and standard
deviation rather than pooling predictions across repeats. Since an observation
can appear in held-out folds in more than one repeat, repeated-CV records must
not be interpreted as an independent external test population.

These stages preserve the evidence for the configured software evaluation;
they do not establish performance on an external population or scientific
validity beyond the evaluated data and design.
