# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for melite.version."""

from melite import __version__ as public_version
from melite.version import __version__ as source_version


def test_version_is_string():
    assert isinstance(source_version, str)


def test_public_version_matches_source_of_truth():
    assert public_version == source_version
