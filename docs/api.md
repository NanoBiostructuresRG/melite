# API Reference

MELITE exposes a focused public Python API, defined by `melite.__all__`.
Complete classifier evaluation is performed through the command-line interface
(CLI) using `melite run`.

The Python API complements this workflow with
programmatic access to configuration, dataset loading, evaluation-evidence
visualization, inference, and version information.


```python
from melite import predict
from melite import Config
from melite import load_datasets
from melite import plot_f1_macro_evidence
from melite import __version__
```

MELITE is currently pre-stable. During the 0.2.x series, the documented symbols
are supported for the current release but may evolve before version 1.0.


## `predict`

::: melite.predict.predict

## `Config`

::: melite.config.Config

## `load_datasets`

::: melite.load_dataset.load_datasets

## `plot_f1_macro_evidence`

::: melite.plot_metrics.plot_f1_macro_evidence

## `__version__`

::: melite.version.__version__


For workflow-oriented examples, configuration, and the evaluation contract, see
[Usage](usage.md).
