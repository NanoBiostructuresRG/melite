# Configuration

MELITE reads defaults from `melite/config_default.toml`. A user TOML file can
override only the settings that need to change.

## Minimal Override

```toml
[paths]
output = "my_output/"

[benchmark]
levels = [70, 85, 95]

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
data/PCA70.npz          <- required key: X, optional key: y
data/PCA85.npz
data/UMAP70.npz
data/UMAP85.npz
```

Each `.npz` file must contain an `X` array. If an embedded `y` array is present,
MELITE validates it against `raw/labels.npy` to avoid silent feature-label
mismatches.

MELITE is tabular at the modeling level. The learning algorithms only consume
numeric `X` and `y` arrays, so the feature matrix may come from PCA, UMAP,
fingerprints, descriptors, clinical variables, experimental measurements,
industrial features, or manually selected numeric features.

The current dataset orchestration still reflects MELITE's PCA/UMAP origin and
uses concepts such as reduction type and level. Future versions will generalize
dataset definitions so arbitrary prepared tabular matrices can be registered
directly. Future configuration may look conceptually like this; it is not
current behavior:

```toml
[datasets.morgan]
path = "data/morgan.npz"
label_path = "raw/labels.npy"

[datasets.descriptors]
path = "data/descriptors.npz"
label_path = "raw/labels.npy"

[datasets.pca85]
path = "data/PCA85.npz"
label_path = "raw/labels.npy"
```

## Outputs

By default, MELITE writes results under `output/`:

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

Remove a key from `active` to skip that model family in a run. The detailed
hyperparameter grids are defined in `melite/config.py`; they are
developer-facing defaults rather than user TOML settings.

| Config key | Model family | Benchmark coverage |
|------------|--------------|--------------------|
| `svc` | Support Vector Classifier | Full runs evaluate polynomial and RBF kernels. Smoke mode uses a linear SVC only. |
| `rf` | Random Forest Classifier | Evaluates tree count, tree depth, feature sampling, and split/leaf controls. |
| `xgb` | XGBoost Classifier | Evaluates tree count, learning rate, depth, sampling, gamma, and regularization. |

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
