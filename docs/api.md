# API Reference

MELITE exposes a focused public Python API, defined by `melite.__all__`.
Complete classification and regression evaluation is performed through the
command-line interface (CLI) using `melite run`.

The Python API complements these task-specific workflows with programmatic
access to configuration, dataset loading, classification evaluation-evidence
visualization, classification and regression inference, and version
information.


```python
from melite import predict
from melite import Config
from melite import load_datasets
from melite import plot_f1_macro_evidence
from melite import __version__
```

MELITE is currently pre-stable. The documented symbols are supported for the
current release but may evolve before version 1.0.


## `predict`

Loads a fitted artifact produced by `melite export`. Classification artifacts
return labels and optional probabilities; regression artifacts return
continuous predictions with `probabilities` set to `None`.

Binary and multiclass classifier artifacts preserve their original class
identities. For multiclass inference, probability columns follow the loaded
model's `classes_` order. Callers must still provide the same numeric feature
representation, feature order, and feature count used for fitting.

::: melite.predict.predict

## `Config`

Loads and normalizes dataset tasks, active classifiers, active regressors,
cross-validation settings, optimization settings, and paths.

::: melite.config.Config

## `load_datasets`

Loads registered NPZ and CSV datasets while preserving each normalized
classification or regression task.

::: melite.load_dataset.load_datasets

## `plot_f1_macro_evidence`

Creates the classification F1-macro evidence visualization from existing
outer-cross-validation results. Regression does not currently expose a
corresponding plotting function.

::: melite.plot_metrics.plot_f1_macro_evidence

## `__version__`

::: melite.version.__version__


For workflow-oriented examples, configuration, and the evaluation contract, see
[Usage](usage.md).
