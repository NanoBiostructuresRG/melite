# Release Notes

MELITE `0.1.11` prepares the project documentation and package metadata for
the first PyPI publication as `melite`.

## 0.1.11 Highlights

- Uses final release metadata version `0.1.11`.
- Clarifies that MELITE is tabular at the modeling level and consumes numeric
  `X` and `y` arrays.
- Documents that current dataset orchestration remains PCA/UMAP-oriented for
  historical reasons.
- Records generalized `[datasets.*]` definitions as a future direction, not
  current behavior.
- Does not change functional training, selection, export, prediction, or CLI
  behavior.

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
