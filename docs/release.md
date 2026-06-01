# Release Notes

MELITE `0.2.3` adds an opt-in experimental stacking workflow while preserving
the top-level public API, existing CLI behavior, and `.pkl`/`joblib` export
format.

## 0.2.3 Highlights

- Experimental stacking can be enabled with `"stack"` in `[models].active`.
- Stacking uses sklearn `StackingClassifier` with
  `stack_method="predict_proba"`, `passthrough=False`, and
  `LogisticRegression` as the final estimator.
- The SVC base estimator inside stacking is a `StandardScaler` ->
  `SVC(probability=True)` pipeline.
- Random Forest and XGBoost remain unscaled direct estimators because they are
  tree-based models and do not require feature scaling by default.
- Stacking-internal CV uses the configured split count and random state with
  one repeat to satisfy sklearn's out-of-fold prediction requirements.
- Optuna and MLflow are not part of v0.2.3.

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
