# SPDX-License-Identifier: LGPL-3.0-or-later
"""MOSAIC — multi-model selection, cross-validation and export toolkit.

Public API
----------
The following symbols are part of the stable public API:

    from mosaic import Config
    from mosaic import load_dataset
    from mosaic import ResultManager
    from mosaic import plot_cv_distributions
    from mosaic import __version__
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
