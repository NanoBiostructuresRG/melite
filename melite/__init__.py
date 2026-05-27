# SPDX-License-Identifier: LGPL-3.0-or-later
"""MELITE — multi-model selection, cross-validation and export toolkit.

Public API
----------
The following symbols are part of the stable public API:

    from melite import Config
    from melite import load_datasets
    from melite import load_dataset
    from melite import ResultManager
    from melite import plot_cv_distributions
    from melite import predict
    from melite import __version__
"""

import logging

from .config import Config
from .load_dataset import load_datasets, load_dataset
from .result_manager import ResultManager
from .plot_metrics import plot_cv_distributions
from .predict import predict
from .version import __version__

__all__ = [
    "Config",
    "load_datasets",
    "load_dataset",
    "ResultManager",
    "plot_cv_distributions",
    "predict",
    "__version__",
]

logging.getLogger(__name__).addHandler(logging.NullHandler())
