# API Reference

MELITE exposes an intended public API through six symbols. The project is
pre-stable, so this API may change before 0.2.0. Internal modules are importable
directly but are not part of the public contract.

```python
from melite import Config
from melite import load_datasets
from melite import load_dataset
from melite import ResultManager
from melite import plot_cv_distributions
from melite import predict
from melite import __version__
```

---

## Config

::: melite.config.Config

---

## load_dataset

::: melite.load_dataset.load_datasets

---

## load_dataset legacy wrapper

::: melite.load_dataset.load_dataset

---

## ResultManager

::: melite.result_manager.ResultManager

---

## plot_cv_distributions

::: melite.plot_metrics.plot_cv_distributions

---

## predict

::: melite.predict.predict

---

## Version

::: melite.version
