# Release Notes

MOSAIC `0.1.9` is a documentation expansion and PyPI-preparation release.

## 0.1.9 Highlights

- Added dedicated MkDocs pages for installation, quick start, CLI reference,
  configuration, and release notes.
- Reduced the home page to a clearer project overview with workflow diagram,
  scope table, and direct navigation.
- Aligned README and package metadata with the `dev/v0.1.9` branch.
- Kept the public Python API unchanged.
- Prepared documentation language for planned PyPI publication as
  `mosaic-tabular`.

## Validation Targets

Before release, validate:

```bash
mkdocs build --strict
pytest tests/ -v
python -m build
python -m twine check dist/*
mosaic --help
mosaic run --help
mosaic export --help
mosaic --version
```

## Full Changelog

The complete version history is maintained in the repository changelog:

--8<-- "CHANGELOG.md"
