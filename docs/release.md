# Release Notes

MELITE `0.2.1` hardens the generalized tabular dataset workflow while
preserving the top-level public API.

## 0.2.1 Highlights

- `[models].active` controls which model families are trained.
- Export uses strict dataset loading and requires explicit `X` in individual
  `.npz` files.
- Installed-wheel smoke validation runs and exports a toy `[datasets.toy]`
  workflow outside the repository checkout.
- The public API remains `Config`, `load_datasets`, `plot_cv_distributions`,
  `predict`, and `__version__`.
- Legacy `reduction_type` + `level` export rows remain supported, but
  individual legacy `.npz` files must contain an explicit `X` array.

## Validation Targets

Before release, validate:

```bash
mkdocs build --strict
python -m pytest tests/ -v --basetemp=.review_pytest_tmp -o cache_dir=.review_pytest_cache
python -m build --no-isolation
python -m twine check dist/*
python scripts/smoke_install_wheel.py
melite --help
melite run --help
melite export --help
melite --version
```

## Full Changelog

The complete release history is maintained in the repository changelog:

--8<-- "CHANGELOG.md"
