# SPDX-License-Identifier: LGPL-3.0-or-later
"""MOSAIC — multi-model selection, cross-validation and export toolkit.

MOSAIC evaluates machine learning workflows based on PCA- or UMAP-reduced
feature matrices, performs grid search over SVC, Random Forest and XGBoost
classifiers, evaluates model configurations with repeated stratified
cross-validation, exports TXT/CSV performance summaries, supports final
model retraining on all available data, saves deployable ``.pkl`` model
artifacts, and generates three-panel metric plots for F1, Accuracy and
AUC-ROC.

Public API
----------
The following symbols form the stable public API of MOSAIC:

.. code-block:: python

    from mosaic import Config
    from mosaic import load_dataset
    from mosaic import ResultManager
    from mosaic import plot_cv_distributions
    from mosaic import __version__

Modules not listed above are importable directly but are not part of the
stable API and may change between versions.

Examples
--------
Basic programmatic usage:

.. code-block:: python

    from mosaic import Config, load_dataset

    cfg = Config()
    cfg.setup()
    dataset = load_dataset(cfg, "PCA", [70, 85])
    X, y = dataset["PCA70"]
"""

import logging

from mosaic.config import Config
from mosaic.load_dataset import load_dataset
from mosaic.result_manager import ResultManager
from mosaic.plot_metrics import plot_cv_distributions
from mosaic.version import __version__

__all__ = [
    "Config",
    "load_dataset",
    "ResultManager",
    "plot_cv_distributions",
    "__version__",
]

logging.getLogger(__name__).addHandler(logging.NullHandler())
