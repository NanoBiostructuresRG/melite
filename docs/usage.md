# Usage

## Installation

### Package Users

Install MELITE in a supported Python environment:

```bash
python -m pip install melite
```

Verify the installation:

```bash
melite --version
```

### Contributors and Developers

Clone the repository and install MELITE in editable mode with its development
dependencies:

```bash
git clone https://github.com/NanoBiostructuresRG/melite.git
cd melite
python -m pip install -e ".[dev]"
```

To work on the documentation:

```bash
python -m pip install -e ".[dev,docs]"
mkdocs serve
```

## Quick Start

The bundled synthetic example provides a fast way to verify the installation
and execution flow.

### Run a Smoke Evaluation

```bash
melite run --smoke --config examples/example_config.toml
```

Smoke mode uses reduced evaluation settings for a fast execution check. It is
not intended for final model selection.

### Run a Configured Evaluation

```bash
melite run --config my_config.toml
```

### Export a Selected Model

After reviewing `results.csv`, export a selected result:

```bash
melite export --config my_config.toml --row 0
```

Smoke results are blocked from export by default. Override the guard only when
that behavior is intentional:

```bash
melite export --config my_config.toml --row 0 --force
```

### Predict from Python

Use an exported model artifact for inference:

```python
import numpy as np
from melite import predict

X_new = np.load("examples/sample_PCA70.npz")["X"]
result = predict("examples/output/Model_SVC_sample_pca70.pkl", X_new)

print(result["predictions"])
print(result["probabilities"])
```

## Command-Line Interface

### Help and Version

```bash
melite --help
melite run --help
melite export --help
melite --version
```

### `melite run`

Run with the default configuration:

```bash
melite run
```

Use a project-specific TOML configuration:

```bash
melite run --config my_config.toml
```

Enable verbose logs:

```bash
melite run --verbose
```

Run a smoke evaluation:

```bash
melite run --smoke --config my_config.toml
```

### `melite export`

Launch interactive row selection:

```bash
melite export
```

Export a specific row:

```bash
melite export --row 0
```

Use explicit result and output paths:

```bash
melite export --row 0 --csv output/results.csv --outdir output/
```

Use a project-specific configuration:

```bash
melite export --config my_config.toml --row 0
```

`melite export` reconstructs the selected model, fits it on all available data,
and serializes the final `.pkl` artifact. It does not run additional
post-selection cross-validation.

## Configuration

MELITE reads defaults from `melite/config_default.toml`. A user TOML file can
override only the settings that need to change.

### Minimal Configuration

```toml
[paths]
output = "output/"

[datasets.morgan_r2_2048]
path = "data/morgan_r2_2048.npz"
label_path = "raw/labels.npy"
family = "fingerprints"
method = "Morgan"
variant = "r2_2048"
description = "Morgan fingerprints, radius 2, 2048 bits"

[models]
active = ["svc", "rf", "xgb"]
```

Use the configuration from the CLI:

```bash
melite run --config my_config.toml
melite export --config my_config.toml --row 0
```

### Dataset Registry

Each numeric matrix is registered under a user-defined
`[datasets.<dataset_id>]` entry.

Required fields:

- `path` — path to the `.npz` file containing `X`;
- `label_path` — path to the target-label array.

Optional metadata fields:

- `family`;
- `method`;
- `variant`;
- `level`;
- `description`.

These fields are preserved for reporting and traceability. They do not trigger
dataset-specific execution logic.

The `family` field is dataset metadata describing the feature representation.
It is unrelated to the classifier selected or evaluated by MELITE.

### Input Format

A registered `.npz` dataset must contain an `X` array. MELITE validates that:

- `X` is two-dimensional;
- `X` is numeric;
- the number of rows in `X` matches the number of labels in `y`;
- an embedded `y` array, when present, matches the configured label vector.

A typical input layout is:

```text
raw/
`-- labels.npy

data/
|-- morgan_r2_2048.npz
|-- rdkit_descriptors.npz
|-- PCA85.npz
`-- UMAP90.npz
```

The filenames and feature-generation methods are examples only. MELITE does not
require PCA, UMAP, fingerprints, descriptors, or any other specific feature
representation.

### Supported Classifiers

MELITE currently supports four classifier keys:

| Key | Classifier | Active by default |
|---|---|---|
| `svc` | Support Vector Classifier | Yes |
| `rf` | Random Forest | Yes |
| `xgb` | XGBoost | Yes |
| `stack` | Stacking classifier | No |

The default configuration is:

```toml
[models]
active = ["svc", "rf", "xgb"]
```

Add `"stack"` to evaluate Stacking alongside the default classifiers.

Standalone SVC uses a `StandardScaler` -> `SVC` pipeline. Probability fitting
is disabled during standalone SVC evaluation and retained where required for
exported inference artifacts. Random Forest and XGBoost remain unscaled.
Stacking is stable but opt-in and combines SVC, Random Forest, and XGBoost with
logistic regression as the final estimator.

Hyperparameter grids are defined in `melite/config.py`. User configuration
controls which supported classifiers are active rather than individual grid
values.

### Evaluation Settings

MELITE separates hyperparameter tuning from model evaluation. For tunable
classifiers, search occurs within the training portion of each outer split.
Evaluation evidence comes from repeated stratified outer cross-validation, and
the selected classifier is determined by mean outer-CV F1-macro.

The configured inner split count is also used by Stacking for its internal
out-of-fold predictions.

### Legacy Configuration

The legacy `[benchmark]` configuration section remains supported for backward
compatibility. When `[datasets]` is absent, legacy reduction settings are
normalized into equivalent dataset entries.

New configurations should use the dataset registry.

## Main Outputs

A standard MELITE workflow produces evaluation artifacts and, when requested,
a final exported model:

```text
output/
|-- results.txt
|-- results.csv
|-- evaluations.csv
|-- evaluation_folds.csv
|-- figures/
|   `-- evaluation_f1_macro_<dataset>.png
`-- Model_<model>_<dataset>.pkl
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
- `Model_<model>_<dataset>.pkl` — final full-data fitted model created by
  `melite export`.

The evaluation figure is generated from already-computed outer-CV evidence. It
does not trigger additional fitting, tuning, cross-validation, or selection.
