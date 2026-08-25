# Usage

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

MELITE includes a bundled synthetic numeric CSV dataset and example
configuration for verifying the installed workflow without cloning the
repository or preparing input files.

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
`results.csv`, `evaluations.csv`, `evaluation_folds.csv`, and the F1-macro
evidence figure. For the bundled example, `results.csv` contains one selected
result for `sample_tabular`.

Smoke mode uses reduced evaluation settings to verify the execution workflow.
The resulting evaluation evidence is explicitly marked as smoke and is not
intended for final classifier selection or model export.

## Data Preparation

MELITE evaluates prepared numeric tabular feature representations. Users are
responsible for supplying analysis-ready numeric features compatible with the
classifiers they evaluate.

### CSV

CSV is the recommended low-friction registered input path. A single table
contains both feature columns and labels, and the configured `label_column`
identifies the target column. All remaining columns become `X` in exactly the
order in which they appear in the CSV. Feature columns must be numeric, while
labels may be categorical.

Column names help define and validate the registered input, but MELITE
does not persist feature names into exported model artifacts. Training and
later inference must therefore use the same feature representation, number of
features, and feature order.

A column whose name starts with `Unnamed:` is rejected because it commonly
represents an accidentally exported pandas index. When creating a CSV with
pandas, omit that index explicitly:

```python
dataframe.to_csv("data/features.csv", index=False)
```

MELITE does not:

- impute missing values;
- encode categorical feature columns;
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

MELITE currently supports four classifier keys:

| Key | Classifier | Tunable | Active by default |
|---|---|---|---|
| `svc` | Support Vector Classifier | Yes | Yes |
| `rf` | Random Forest | Yes | Yes |
| `xgb` | XGBoost | Yes | Yes |
| `stack` | Stacking classifier | No | No |

The default active classifiers are:

```toml
[classifiers]
active = ["svc", "rf", "xgb"]
```

Add `"stack"` to evaluate Stacking alongside the default classifiers.

Standalone SVC uses a `StandardScaler` → `SVC` pipeline. Probability fitting is
disabled during standalone SVC evaluation and retained where required for
exported inference artifacts.

Random Forest and XGBoost operate on the supplied numeric features without
scaling.

Stacking is stable but opt-in. It combines SVC, Random Forest, and XGBoost as
base estimators with logistic regression as the final estimator.

Hyperparameter grids for tunable classifiers are defined by MELITE. User
configuration determines which supported classifiers are active rather than
defining individual grid values.

## Configuration

MELITE reads its default settings from `melite/config_default.toml`. A user
TOML file overrides only the settings that need to change.

Relative dataset, label, and output paths are resolved from the working
directory in which `melite` is executed, not from the location of the TOML
file.

For modern registered datasets, the actual dataset file is determined by
`[datasets.*].path`. `[paths].input` remains part of MELITE's historical path
contract and is not used to locate a registered CSV dataset. The bundled CSV
example deliberately points both `input` and `dataset` to
`melite_example/data/` so it does not create an unused `raw/` directory.

### Minimal Configuration

A minimal dataset configuration can be written as:

```toml
[paths]
output = "output/"

[datasets.tabular_a]
path = "data/features.csv"
label_column = "label"
description = "Prepared numeric feature matrix."

[classifiers]
active = ["svc", "rf", "xgb"]
```

Run MELITE from the project directory expected by those paths:

```bash
melite run --config my_config.toml
```

### Dataset Registry

Each numeric feature matrix is registered under a user-defined
`[datasets.<dataset_id>]` entry.

CSV example:

```toml
[datasets.my_dataset]
path = "data/my_dataset.csv"
label_column = "label"
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
- `family`;
- `method`;
- `variant`;
- `level`;
- `description`.

Unknown fields are rejected rather than silently ignored.

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
is unrelated to the classifier selected or evaluated by MELITE.

Each `dataset_id` identifies one concrete numeric feature matrix evaluated as an
independent dataset.

For CSV, MELITE does not persist feature names into exported model
artifacts. Training and later inference must use the same feature
representation and feature order.

### Evaluation Settings

The normal cross-validation defaults are:

```toml
[cv]
n_splits = 5
n_repeats = 3
inner_n_splits = 3
```

For each dataset and active classifier, the outer evaluation contains
`n_splits × n_repeats` held-out splits. With the default settings, this produces
15 outer evaluations per classifier. For tunable classifiers, each outer split
contains its own inner hyperparameter search.

- `n_splits` controls the number of folds in the outer repeated stratified
  cross-validation.
- `n_repeats` controls how many times the outer stratified cross-validation is
  repeated.
- `inner_n_splits` controls the stratified inner cross-validation used for
  hyperparameter search. Stacking also uses this split count for its internal
  out-of-fold predictions.

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

F1-macro is the fixed optimization and selection metric in MELITE.
Hyperparameter search optimizes F1-macro, and the classifier with the highest
mean outer-CV F1-macro is selected. Accuracy and AUC-ROC are reported as
additional evaluation metrics.

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

Evaluation time depends on the number of registered datasets, active
classifiers, cross-validation settings, and the hyperparameter searches
required by the active classifiers. A normal evaluation can therefore take
substantially longer than a smoke run.

### `melite export`

After a normal evaluation, inspect `results.csv` and export the selected result
for the desired dataset.

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
on all available data. For tunable classifiers, the final fit includes a
full-data hyperparameter search.

Smoke results are blocked from export by default. The `--force` option exists
for deliberate testing or diagnostic use.

## Outputs

A standard MELITE workflow produces evaluation artifacts and, when requested,
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

- `results.txt` — human-readable summary of selected results;
- `results.csv` — selected classifier result for each dataset;
- `evaluations.csv` — aggregate evaluation evidence for every active
  classifier;
- `evaluation_folds.csv` — outer-CV evidence for every dataset, classifier, and
  outer split;
- `figures/evaluation_f1_macro_<dataset>.png` — visualization of the outer-CV
  F1-macro evidence used for classifier selection;
- `Model_<classifier>_<dataset>.pkl` — final full-data fitted model created by
  `melite export`.

`results.csv`, `evaluations.csv`, and `evaluation_folds.csv` include a `smoke`
column identifying whether their evidence was produced in smoke mode.

### Output Data Contract

The three CSV files are persistent data artifacts whose schemas are part of
the MELITE package-version contract. MELITE does not embed an
independent machine-readable `schema_version`. A separate schema version
should be reconsidered only if output schemas need to evolve independently of
the MELITE package version.

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
| `f1_macro` | Mean outer-CV F1-macro for the selected classifier. |
| `f1_std` | Population standard deviation of outer-CV F1-macro. |
| `accuracy` | Mean outer-CV accuracy for the selected classifier. |
| `acc_std` | Population standard deviation of outer-CV accuracy. |
| `auc_roc` | Mean outer-CV AUC-ROC when available. |
| `auc_std` | Population standard deviation of outer-CV AUC-ROC when available. |
| `smoke` | Whether the row was produced in smoke mode. |

The metric values in this selected-result summary are written rounded to four
decimal places by the current `melite run` workflow. The recorded `parameters`
must not be assumed to be the final parameters stored in a subsequently
exported model. For tunable classifiers, `melite export` performs a final
full-data hyperparameter search, so the exported fitted artifact may use
parameters determined during that final fitting stage.

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

## Python API

The public Python API can load an exported model artifact and run inference on
new numeric feature matrices.

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

`X_new` must use the same feature representation expected by the exported
model.

The returned `probabilities` value is `None` when the loaded estimator does not
provide probability estimates.

For the complete public Python interface, see the [API Reference](api.md).

## Evaluation Contract

For each registered dataset, MELITE follows a reproducible evaluation contract:

1. `X` is validated as a two-dimensional numeric feature matrix, and `y`
   provides labels for the same samples.
2. Each active classifier is evaluated under the configured repeated
   stratified outer cross-validation design.
3. For tunable classifiers, hyperparameter search occurs only within the
   training portion of each outer split.
4. Evaluation evidence comes from the held-out outer folds across all repeats.
5. Mean outer-CV F1-macro determines the selected classifier for each dataset.
6. Aggregate and per-fold evidence are preserved for every evaluated
   classifier.
7. The F1-macro evidence figure is generated from preserved outer-CV evidence
   without additional fitting, tuning, cross-validation, or selection.
8. Selection and final fitting are separate stages.
9. After selection, the chosen classifier is fitted using all available data.
   If it is tunable, MELITE performs a final full-data hyperparameter search to
   determine the exported configuration.
10. `melite export` does not run a second post-selection evaluation.
