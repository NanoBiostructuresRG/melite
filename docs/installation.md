# Installation

MELITE is developed for Python 3.11 and 3.12.

## Local Install

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/NanoBiostructuresRG/melite.git
cd melite
python -m pip install -e .
```

This installs the `melite` command and the Python package.

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

MELITE is being prepared for publication on PyPI as `melite`.
After publication, install with:

```bash
python -m pip install melite
```

Until that package is published, install from the repository in editable mode.

## Verify Installation

```bash
melite --help
melite --version
```

Expected version for this release:

```text
MELITE 0.2.2
```
