# Release Notes

MELITE `0.2.2` adds SVC feature scaling while preserving the top-level public
API and existing CLI behavior.

## 0.2.2 Highlights

- SVC is trained as a `StandardScaler` -> `SVC` sklearn `Pipeline`.
- Scaling is applied only to SVC; Random Forest and XGBoost remain unscaled.
- Exported SVC models preserve the same `StandardScaler` -> `SVC` pipeline.
- Legacy export compatibility is preserved for older unprefixed SVC
  parameter dictionaries such as `{"C": 1, "kernel": "linear"}`.

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
