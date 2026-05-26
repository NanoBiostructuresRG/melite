# Installation

MOSAIC is developed for Python 3.11 and 3.12.

## Local Install

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/NanoBiostructuresRG/mosaic.git
cd mosaic
python -m pip install -e .
```

This installs the `mosaic` command and the Python package.

## Optional Dependencies

Install development tools when running tests or building distributions:

```bash
python -m pip install -e ".[dev]"
```

Install documentation tools when building the MkDocs site:

```bash
python -m pip install -e ".[docs]"
```

## PyPI Status

MOSAIC is being prepared for publication on PyPI as `mosaic-tabular`.
Until that package is published, install from the repository in editable mode.

## Verify Installation

```bash
mosaic --help
mosaic --version
```

Expected version for this development branch:

```text
MOSAIC 0.1.9
```
