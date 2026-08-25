# SPDX-License-Identifier: LGPL-3.0-or-later
"""This contract test verifies documented output column names and order.

It does not validate the semantic accuracy of the prose descriptions.
"""

import csv
import re
from pathlib import Path

import pytest

from melite.result_manager import ResultManager


USAGE_PATH = Path(__file__).resolve().parents[1] / "docs" / "usage.md"


def _parse_table_row(line: str) -> list[str] | None:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return None
    return [cell.strip() for cell in stripped[1:-1].split("|")]


def _documented_schema(markdown: str, filename: str) -> list[str]:
    marker = f"<!-- melite-schema:{filename} -->"
    marker_count = markdown.count(marker)
    assert marker_count == 1, (
        f"Schema marker {marker!r} must appear exactly once; found {marker_count}."
    )

    following_lines = markdown.split(marker, 1)[1].splitlines()
    table_start = next(
        (index for index, line in enumerate(following_lines) if line.strip()),
        None,
    )
    assert table_start is not None, f"No valid Markdown table follows {marker!r}."

    header = _parse_table_row(following_lines[table_start])
    assert header is not None, f"No valid Markdown table follows {marker!r}."
    assert header == ["Column", "Meaning"], (
        f"Schema table after {marker!r} must have the expected 'Column' header."
    )

    separator_index = table_start + 1
    assert separator_index < len(following_lines), (
        f"Schema table after {marker!r} has no valid separator row."
    )
    separator = _parse_table_row(following_lines[separator_index])
    assert separator is not None and len(separator) == len(header), (
        f"Schema table after {marker!r} has no valid separator row."
    )
    assert all(
        re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in separator
    ), f"Schema table after {marker!r} has no valid separator row."

    columns = []
    for line in following_lines[separator_index + 1 :]:
        row = _parse_table_row(line)
        if row is None:
            break
        assert len(row) == len(header), (
            f"Schema table after {marker!r} contains an invalid row: {line!r}."
        )
        column = row[0].strip("`").strip()
        if column:
            columns.append(column)

    assert columns, f"Schema table after {marker!r} contains no schema rows."
    return columns


def _writer_schema(tmp_path: Path, writer_name: str, filename: str) -> list[str]:
    manager = ResultManager(tmp_path / "results.txt")
    csv_path = tmp_path / filename
    getattr(manager, writer_name)([{}], csv_path)

    with open(csv_path, newline="", encoding="utf-8") as file:
        fieldnames = csv.DictReader(file).fieldnames

    assert fieldnames is not None, f"ResultManager wrote no header for {filename}."
    return fieldnames


@pytest.mark.parametrize(
    ("filename", "writer_name"),
    [
        ("results.csv", "write_csv"),
        ("evaluations.csv", "write_evaluations_csv"),
        ("evaluation_folds.csv", "write_evaluation_folds_csv"),
    ],
)
def test_documented_schema_matches_real_result_manager_writer(
    tmp_path, filename, writer_name
):
    markdown = USAGE_PATH.read_text(encoding="utf-8")

    documented = _documented_schema(markdown, filename)
    actual = _writer_schema(tmp_path, writer_name, filename)

    assert documented == actual


def test_schema_parser_fails_when_marker_is_missing():
    with pytest.raises(AssertionError, match=r"exactly once; found 0"):
        _documented_schema("# Usage\n", "results.csv")


def test_schema_parser_fails_when_marker_is_duplicated():
    marker = "<!-- melite-schema:results.csv -->"

    with pytest.raises(AssertionError, match=r"exactly once; found 2"):
        _documented_schema(f"{marker}\n{marker}\n", "results.csv")


def test_schema_parser_fails_when_no_table_follows_marker():
    markdown = "<!-- melite-schema:results.csv -->\n\nNot a table.\n"

    with pytest.raises(AssertionError, match="No valid Markdown table"):
        _documented_schema(markdown, "results.csv")


def test_schema_parser_fails_when_column_header_is_missing():
    markdown = (
        "<!-- melite-schema:results.csv -->\n\n"
        "| Field | Meaning |\n"
        "|---|---|\n"
        "| dataset | Dataset id. |\n"
    )

    with pytest.raises(AssertionError, match="expected 'Column' header"):
        _documented_schema(markdown, "results.csv")


def test_schema_parser_fails_when_table_has_no_schema_rows():
    markdown = "<!-- melite-schema:results.csv -->\n\n| Column | Meaning |\n|---|---|\n"

    with pytest.raises(AssertionError, match="no schema rows"):
        _documented_schema(markdown, "results.csv")
