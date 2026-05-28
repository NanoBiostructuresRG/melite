# SPDX-License-Identifier: LGPL-3.0-or-later
"""Package version metadata for MELITE.

This module is the single source of truth for the project version.
It is read by ``hatchling`` at build time via ``[tool.hatch.version]``
and imported by ``result_manager`` to stamp generated reports.
"""

__version__ = "0.2.2"
PROJECT_NAME = "MELITE"
PROJECT_VERSION = __version__
PROJECT_STATUS = "alpha"
PROJECT_LICENSE = "LGPL-3.0-or-later"

__all__ = [
    "__version__",
    "PROJECT_NAME",
    "PROJECT_VERSION",
    "PROJECT_STATUS",
    "PROJECT_LICENSE",
]
