# SPDX-License-Identifier: LGPL-3.0-or-later
"""Contract tests for the public README surface."""

import argparse
import re
import shlex
from pathlib import Path
from urllib.parse import unquote

from melite import __version__
from melite.cli import _build_parser


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
README_PATH = REPOSITORY_ROOT / "README.md"
README_TEXT = README_PATH.read_text(encoding="utf-8")


def _section(title: str) -> str:
    match = re.search(
        rf"^## {re.escape(title)}\s*$\n(.*?)(?=^## |\Z)",
        README_TEXT,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"README section not found: {title}"
    return match.group(1)


def _fenced_blocks(language: str) -> list[str]:
    return re.findall(
        rf"```{re.escape(language)}\s*\n(.*?)^```",
        README_TEXT,
        flags=re.MULTILINE | re.DOTALL,
    )


def _cli_subcommands() -> set[str]:
    parser = _build_parser()
    subparser_actions = [
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    ]
    assert len(subparser_actions) == 1
    return set(subparser_actions[0].choices)


def test_quick_start_uses_canonical_packaged_example():
    quick_start = _section("Quick Start")

    assert "melite example" in quick_start
    assert "melite run --smoke --config melite_example/config.toml" in quick_start
    for obsolete_reference in (
        "examples/",
        "sample_PCA70",
        "Model_SVC_sample_pca70.pkl",
        "environment.yml",
    ):
        assert obsolete_reference not in quick_start


def test_version_badge_matches_package_version():
    match = re.search(
        r"\[!\[Version\]\(https://img\.shields\.io/badge/version-v(.*?)-blue\.svg\)\]\(\)",
        README_TEXT,
    )

    assert match is not None, "README Version badge not found"
    assert match.group(1) == __version__


def test_readme_melite_commands_use_real_cli_subcommands():
    commands = [
        line.strip()
        for block in _fenced_blocks("bash")
        for line in block.splitlines()
        if line.strip().startswith("melite ")
    ]
    subcommands = _cli_subcommands()

    assert commands
    for command in commands:
        arguments = shlex.split(command)[1:]
        if not arguments or arguments[0].startswith("-"):
            continue
        assert arguments[0] in subcommands, (
            f"Unknown MELITE command in README: {command}"
        )


def test_readme_toml_examples_use_current_classifier_section():
    toml_blocks = _fenced_blocks("toml")

    assert toml_blocks
    assert all("[models]" not in block for block in toml_blocks)


def test_readme_local_links_reference_existing_files():
    without_code_blocks = re.sub(
        r"```.*?```",
        "",
        README_TEXT,
        flags=re.DOTALL,
    )
    linked_image_targets = re.findall(
        r"\[!\[[^\]]*\]\([^)]*\)\]\(([^)]+)\)",
        without_code_blocks,
    )
    regular_targets = re.findall(
        r"(?<!!)\[[^\]]+\]\(([^)]+)\)",
        without_code_blocks,
    )

    for target in {*linked_image_targets, *regular_targets}:
        target = target.strip()
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        local_path = unquote(target.split("#", 1)[0].split("?", 1)[0])
        assert local_path
        assert (REPOSITORY_ROOT / local_path).exists(), (
            f"README local link does not exist: {target}"
        )
