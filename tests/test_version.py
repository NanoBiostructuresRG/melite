# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for melite.version."""

import re
from melite.version import (
    __version__,
    PROJECT_NAME,
    PROJECT_VERSION,
    PROJECT_STATUS,
    PROJECT_LICENSE,
)


def test_version_is_string():
    assert isinstance(__version__, str)


def test_version_matches_semver():
    assert re.match(r"^\d+\.\d+\.\d+$", __version__), (
        f"__version__ '{__version__}' does not match X.Y.Z pattern"
    )


def test_project_name():
    assert PROJECT_NAME == "MELITE"


def test_project_status():
    assert PROJECT_STATUS == "alpha"


def test_project_license():
    assert PROJECT_LICENSE == "LGPL-3.0-or-later"


def test_project_version_matches_version():
    assert PROJECT_VERSION == __version__
