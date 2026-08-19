# SPDX-License-Identifier: LGPL-3.0-or-later
"""MELITE — Multi-Model Classifier Evaluator.

Public API
----------
The following symbols are exposed through the public API:

    from melite import Config
    from melite import load_datasets
    from melite import plot_f1_macro_evidence
    from melite import predict
    from melite import __version__
"""

import logging

from .config import Config
from .load_dataset import load_datasets
from .plot_metrics import plot_f1_macro_evidence
from .predict import predict
from .version import __version__

__all__ = [
    "Config",
    "load_datasets",
    "plot_f1_macro_evidence",
    "predict",
    "__version__",
]

logging.getLogger(__name__).addHandler(logging.NullHandler())
