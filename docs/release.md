# Release Notes

MELITE `0.1.11` is an identity and packaging rename release.

## 0.1.10 Highlights

- Renamed the user-facing project identity to MELITE.
- Renamed the Python import package to `melite`.
- Renamed the CLI command to `melite`.
- Renamed the PyPI distribution target to `melite`.
- Kept the public API symbols unchanged under the new import package.
- Preserved the v0.1.9 documentation architecture.

## Validation Targets

Before release, validate:

```bash
mkdocs build --strict
python -m pytest tests/ -v --basetemp=.review_pytest_tmp -o cache_dir=.review_pytest_cache
python -m build
python -m twine check dist/*
melite --help
melite run --help
melite export --help
melite --version
```

## Full Changelog

The complete version history is maintained in the repository changelog:

--8<-- "CHANGELOG.md"
