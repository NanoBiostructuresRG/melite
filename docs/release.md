# Release Notes

MELITE `0.2.0` introduces the generalized tabular dataset registry and keeps
legacy PCA/UMAP configuration compatibility.

## 0.2.0 Highlights

- Registers concrete tabular matrices under `[datasets.<dataset_id>]`.
- Requires `path` and `label_path`; preserves optional metadata fields
  `family`, `method`, `variant`, `level`, and `description`.
- Runs benchmarks through strict `cfg.DATASETS` loading.
- Exports dataset-based artifacts such as `Model_SVC_morgan_r2_2048.pkl`.
- Falls back to legacy `reduction_type` + `level` export rows for older CSVs.

## 0.1.11 Highlights

MELITE `0.1.11` prepared the project documentation and package metadata for
the first PyPI publication as `melite`.

- Uses final release metadata version `0.1.11`.
- Clarifies that MELITE is tabular at the modeling level and consumes numeric
  `X` and `y` arrays.
- Documented generalized `[datasets.*]` definitions as a future direction at
  that time.
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
