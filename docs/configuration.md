# Configuration

MELITE reads defaults from `melite/config_default.toml`. A user TOML file can
override only the settings that need to change.

## Minimal Override

```toml
[paths]
output = "my_output/"

[datasets.morgan_r2_2048]
path = "data/morgan_r2_2048.npz"
label_path = "raw/labels.npy"
family = "fingerprints"
method = "Morgan"
variant = "r2_2048"

[models]
active = ["svc", "rf"]
```

Use the file from the CLI:

```bash
melite run --config my_config.toml
melite export --config my_config.toml --row 0
```

## Input Layout

MELITE consumes pre-computed feature matrices and labels:

```text
raw/labels.npy          <- target vector y, shape (n_samples,)
data/morgan_r2_2048.npz <- required key: X, optional key: y
data/maccs.npz
data/rdkit_descriptors.npz
data/PCA85.npz
data/UMAP90.npz
```

Each `.npz` file must contain an `X` array. If an embedded `y` array is present,
MELITE validates it against the configured `label_path` to avoid silent feature-label
mismatches.

MELITE is tabular at the modeling level. The learning algorithms only consume
numeric `X` and `y` arrays, so the feature matrix may come from PCA, UMAP,
fingerprints, descriptors, clinical variables, experimental measurements,
industrial features, or manually selected numeric features.

Each concrete matrix candidate is registered under `[datasets.<dataset_id>]`.
Required fields are `path` and `label_path`. Optional metadata fields are
`family`, `method`, `variant`, `level`, and `description`; they are preserved
in reports for traceability and do not control special-case execution logic.

```toml
[datasets.morgan_r2_2048]
path = "data/morgan_r2_2048.npz"
label_path = "raw/labels.npy"
family = "fingerprints"
method = "Morgan"
variant = "r2_2048"

[datasets.maccs]
path = "data/maccs.npz"
label_path = "raw/labels.npy"
family = "fingerprints"
method = "MACCS"

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

[datasets.umap90]
path = "data/UMAP90.npz"
label_path = "raw/labels.npy"
family = "dimensionality"
method = "UMAP"
level = 90
```

Registered datasets are loaded strictly. A missing dataset file, missing
`label_path`, missing `X`, non-2D `X`, non-numeric `X`, length mismatch, or
embedded `y` mismatch raises an error instead of silently skipping the entry.

Legacy `[benchmark].reduction_types` and `levels` remain supported for
compatibility. When `[datasets]` is absent, MELITE synthesizes entries such as
`PCA70` and `UMAP90` with dimensionality metadata.

## Outputs

By default, MELITE writes results under `output/`:

```text
output/
|-- results.txt
|-- results.csv
|-- Model_<model>_<dataset>.pkl
`-- figures/
    `-- <model>_<dataset>.png
```

| Output | Purpose |
|--------|---------|
| `results.txt` | Human-readable benchmark report. |
| `results.csv` | Structured rows with model performance, parameters, and smoke marker. |
| `.pkl` artifact | Final selected model retrained on all available data. |
| PNG figure | F1, Accuracy, and AUC-ROC cross-validation distributions. |

## Model Families and Hyperparameter Grids

`config_default.toml` controls which model families are active:

```toml
[models]
active = ["svc", "rf", "xgb"]
```

Remove a key from `active` to skip that model family in a run. Add `"stack"`
to opt in to the experimental stacking workflow. The detailed hyperparameter
grids are defined in `melite/config.py`; they are developer-facing defaults
rather than user TOML settings.

| Config key | Model family | Benchmark coverage |
|------------|--------------|--------------------|
| `svc` | Support Vector Classifier | Full runs evaluate polynomial and RBF kernels. Smoke mode uses a linear SVC only. |
| `rf` | Random Forest Classifier | Evaluates tree count, tree depth, feature sampling, and split/leaf controls. |
| `xgb` | XGBoost Classifier | Evaluates tree count, learning rate, depth, sampling, gamma, and regularization. |
| `stack` | Experimental StackingClassifier | Opt-in stack of scaled SVC, unscaled RF, and unscaled XGBoost with logistic regression as the final estimator. |

Standalone SVC is trained and exported as a `StandardScaler` -> `SVC` sklearn
pipeline because SVM/kernel-based models are sensitive to feature scale. Random
Forest and XGBoost are tree-based models and do not require feature scaling by
default, so MELITE keeps them as unscaled estimators.

### SVC Kernels

| Mode | Kernels | Parameters |
|------|---------|------------|
| Full benchmark | `poly`, `rbf` | `C` and `gamma`; plus `coef0` and `degree` for `poly`. |
| Smoke mode | `linear` | `C = 1`. |

Current full-run SVC grids:

| Kernel | Values |
|--------|--------|
| `poly` | `C = [0.01, 0.1, 1, 10]`; `degree = [3, 4, 5]`; `coef0 = [0.0, 0.1, 0.02, 0.6, 0.8, 1]`; `gamma = [0.001, 0.002, 0.004, 0.008, 0.01, 0.02, 0.04, 0.08, 0.1, 0.2]` |
| `rbf` | `C = [0.01, 0.02, 0.1, 0.2, 1, 2, 10, 20]`; `gamma = [0.001, 0.002, 0.004, 0.008, 0.01, 0.02, 0.04, 0.08, 0.1, 0.2]` |

### Tree-Based Models

| Model | Main parameters explored |
|-------|--------------------------|
| Random Forest | `n_estimators`, `max_depth`, `max_features`, `min_samples_split`, and `min_samples_leaf`. |
| XGBoost | `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `gamma`, `reg_alpha`, and `reg_lambda`. |

These grids can be restricted at the family level through `[models].active`.
Changing the individual hyperparameter values currently requires editing
`melite/config.py`.

### Experimental Stacking

Stacking is disabled by default. Enable it explicitly:

```toml
[models]
active = ["svc", "rf", "xgb", "stack"]
```

!!! note
    The stacking workflow uses sklearn `StackingClassifier` with
    `stack_method="predict_proba"`, `passthrough=False`, and `LogisticRegression`
    as the initial final estimator. Its SVC base estimator is a
    `StandardScaler` -> `SVC(probability=True)` pipeline so that the stack combines
    probability outputs from SVC, Random Forest, and XGBoost. RF and XGBoost remain
    unscaled inside the stack.

    The stacking-internal CV uses the configured split count and random state
    without repeated splits because sklearn stacking builds out-of-fold
    meta-features with `cross_val_predict`. This ensures each training sample
    contributes exactly one out-of-fold prediction for training the final
    estimator, while the outer **MELITE** grid search and reporting workflow
    still uses the existing repeated CV/F1 evaluation.

    Export remains a `.pkl` artifact serialized with `joblib`; Optuna and
    MLflow are not part of this workflow.