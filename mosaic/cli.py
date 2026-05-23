# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unified CLI entry point for MOSAIC."""

import argparse
import logging
import sys
from pathlib import Path

from mosaic.version import __version__

logger = logging.getLogger(__name__)


def _global_parser() -> argparse.ArgumentParser:
    """Shared flags inherited by all subcommands."""
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
        prog="mosaic",
        description="MOSAIC — multi-model selection, cross-validation and export toolkit.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[global_parent],
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"MOSAIC {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    # ------------------------------------------------------------------ #
    # mosaic run
    # ------------------------------------------------------------------ #
    run_parser = subparsers.add_parser(
        "run",
        help="Run the full benchmarking pipeline.",
        description="Run grid search and cross-validation over all configured datasets and models.",
        parents=[global_parent],
    )
    run_parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Lightweight mode: single-value grids and 3-fold CV. "
            "Results are not benchmark-quality."
        ),
    )

    # ------------------------------------------------------------------ #
    # mosaic export
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

    return parser


def _configure_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(levelname)s:%(name)s:%(message)s",
    )


def _run(args: argparse.Namespace) -> None:
    from mosaic.main import Main

    Main(smoke=args.smoke).run()


def _export(args: argparse.Namespace) -> None:
    from mosaic.config import Config
    from mosaic.export_best_model import Finalizer

    config = Config()
    csv_path = args.csv or Path(config.PATHS["OUTPUT"]) / "results.csv"
    outdir = args.outdir or Path(config.PATHS["OUTPUT"])
    Finalizer(csv_path, outdir, config, row_index=args.row, force=getattr(args, "force", False)).run()


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _configure_logging(args.verbose)

    if args.command == "run":
        _run(args)
    elif args.command == "export":
        _export(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
