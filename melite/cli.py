# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unified CLI entry point for MELITE.

This module provides the ``melite`` command registered in ``pyproject.toml``
under ``[project.scripts]``. It exposes three subcommands:

- ``melite run`` — execute the full evaluation pipeline.
- ``melite export`` — retrain a selected model and export a ``.pkl`` artifact.
- ``melite example`` — copy the bundled example project.

Global flags (``--verbose``, ``--config``, ``--version``) are available to
all subcommands via argparse parent parsers.
"""

import argparse
import logging
import shutil
import sys
from importlib import resources
from pathlib import Path

from .version import __version__

__all__ = ["main"]

logger = logging.getLogger(__name__)

_EXAMPLE_DIRECTORY = "melite_example"
_EXAMPLE_RESOURCE_DIRECTORY = "_example_assets"


def _global_parser() -> argparse.ArgumentParser:
    """Return a parent parser with shared flags for all subcommands."""
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--verbose",
        action="store_true",
        help="Enable progress logging (INFO level).",
    )
    parent.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to a user TOML configuration file. Overrides defaults.",
    )
    return parent


def _build_parser() -> argparse.ArgumentParser:
    global_parent = _global_parser()

    parser = argparse.ArgumentParser(
        prog="melite",
        description="MELITE — Multi-Model Classifier Evaluator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[global_parent],
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"MELITE {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    # ------------------------------------------------------------------ #
    # melite run
    # ------------------------------------------------------------------ #
    run_parser = subparsers.add_parser(
        "run",
        help="Run the full evaluation pipeline.",
        description="Evaluate configured classifiers across all registered datasets.",
        parents=[global_parent],
    )
    run_parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Lightweight mode: reduced search and cross-validation settings. "
            "Results are not suitable for final classifier selection."
        ),
    )

    # ------------------------------------------------------------------ #
    # melite export
    # ------------------------------------------------------------------ #
    export_parser = subparsers.add_parser(
        "export",
        help="Export a selected model from results.csv.",
        description="Retrain a selected model on all available data and save a .pkl artifact.",
        parents=[global_parent],
    )
    export_parser.add_argument(
        "--row",
        type=int,
        default=None,
        metavar="INDEX",
        help="Row index from results.csv to export without interactive prompt.",
    )
    export_parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to results CSV file. Defaults to output/results.csv.",
    )
    export_parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        metavar="PATH",
        help="Destination directory for the .pkl artifact. Defaults to output/.",
    )
    export_parser.add_argument(
        "--force",
        action="store_true",
        help="Override smoke-mode export guard. Use with caution.",
    )

    # ------------------------------------------------------------------ #
    # melite example
    # ------------------------------------------------------------------ #
    subparsers.add_parser(
        "example",
        help="Copy the bundled synthetic example project.",
        description=(
            "Copy a ready-to-run synthetic numeric-tabular example into "
            "./melite_example/."
        ),
        parents=[global_parent],
    )

    return parser


def _configure_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(levelname)s:%(name)s:%(message)s",
    )


def _run(args: argparse.Namespace) -> None:
    from .main import Main

    Main(smoke=args.smoke, user_config=args.config).run()


def _export(args: argparse.Namespace) -> None:
    from .config import Config
    from .export_best_model import Finalizer

    config = Config(user_config=args.config)
    csv_path = args.csv or Path(config.PATHS["OUTPUT"]) / "results.csv"
    outdir = args.outdir or Path(config.PATHS["OUTPUT"])
    Finalizer(
        csv_path,
        outdir,
        config,
        row_index=args.row,
        force=getattr(args, "force", False),
    ).run()


def _copy_resource_tree(source, destination: Path) -> None:
    """Copy an importlib resource directory without merging destinations."""
    destination.mkdir()
    for entry in source.iterdir():
        target = destination / entry.name
        if entry.is_dir():
            _copy_resource_tree(entry, target)
        else:
            with entry.open("rb") as source_file, target.open("wb") as target_file:
                shutil.copyfileobj(source_file, target_file)


def _example(_args: argparse.Namespace) -> None:
    destination = Path.cwd() / _EXAMPLE_DIRECTORY
    source = resources.files("melite").joinpath(_EXAMPLE_RESOURCE_DIRECTORY)

    if not source.is_dir():
        raise RuntimeError("Bundled MELITE example resources are missing.")

    try:
        _copy_resource_tree(source, destination)
    except FileExistsError:
        print(
            f"[ERROR] {destination} already exists; no files were overwritten.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(f"Example project created at: {destination}")
    print(f"Run: melite run --smoke --config {_EXAMPLE_DIRECTORY}/config.toml")


def main() -> None:
    """Entry point for the ``melite`` CLI command.

    Registered in ``pyproject.toml`` as::

        [project.scripts]
        melite = "melite.cli:main"

    Parses arguments, configures logging, and dispatches to the appropriate
    subcommand handler (``_run``, ``_export``, or ``_example``).

    Notes
    -----
    The ``--verbose`` flag sets the root logger to ``INFO`` level, which
    exposes progress messages from all ``melite.*`` modules. Without it,
    only ``WARNING`` and above are shown.
    """
    parser = _build_parser()
    args = parser.parse_args()
    _configure_logging(args.verbose)

    if args.command == "run":
        _run(args)
    elif args.command == "export":
        _export(args)
    elif args.command == "example":
        _example(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
