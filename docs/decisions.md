# Deferred Product Decisions

This page records deliberate deferred product decisions and their revisit
criteria. It is not a release roadmap or a substitute for issue tracking. An
entry is removed when it is implemented or no longer applies.

Every entry must include an explicit revisit criterion. Ideas without one
belong in issue tracking, not here.

The [Changelog](changelog.md) records what happened. This page records what was
deliberately deferred and the observable condition that would justify
reconsidering it.

## Parquet Input

**Status:** Deferred.

**Reason:** CSV addresses the current adoption barrier without adding `pyarrow`
or optional-format complexity.

**Revisit criterion:** Demonstrated user demand for Parquet, or observed
dataset scale or type-preservation limitations that CSV cannot adequately
address.

**Direction if reopened:** Prefer optional dependency support rather than
adding `pyarrow` to the base runtime installation.

## In-Memory DataFrame Datasets

**Status:** Deferred.

**Reason:** Pathless or in-memory datasets belong to a programmatic workflow
rather than being added as an exception to the current file-oriented registry.

**Revisit criterion:** Design of a stable high-level programmatic evaluation
workflow.

**Direction if reopened:** Design DataFrame support together with that workflow
and its artifact and feature semantics.

## `paths.input` Legacy Compatibility

**Status:** Retained deliberately.

**Reason:** It remains part of the historical path contract even though modern
registered CSV datasets are located through `[datasets.*].path`.

**Revisit criterion:** Explicit removal or redesign of the legacy
reduction-based configuration path.

## `reduction_type` Output Field

**Status:** Retained deliberately.

**Reason:** It preserves historical compatibility and is normally empty for
modern registered datasets.

**Revisit criterion:** An explicit breaking cleanup of legacy reduction
compatibility. It must not be removed piecemeal.

## Independent Machine-Readable `schema_version`

**Status:** Deferred.

**Reason:** Current output schemas can be tied to the MELITE package version.

**Revisit criterion:** A demonstrated need for output schemas to evolve
independently of package releases or for external consumers to negotiate
schema versions.

## Optimization Engine and Policy

**Status:** Fixed runtime architecture. The classification policy was
established for v0.3.0; v0.3.2 applies the corresponding task-specific contract
to regression. Optuna is the single optimization engine for tunable models;
coexistence with `GridSearchCV` is not part of the current design. The only
optimization-specific user setting is `n_trials`, with a normal default of 100.
The sampler seed derives from the canonical `RANDOM_STATE`; MELITE does not
expose a separate optimization seed. The smoke budget is an internal,
non-configurable 5 trials.

**Classification policy:** Tunable classifiers maximize mean inner-CV
F1-macro. Inner folds use shuffled `StratifiedKFold`, and outer evidence uses
`RepeatedStratifiedKFold`. The classifier with the highest mean outer-CV
F1-macro is selected; accuracy and ROC-AUC are additional evidence metrics.
Stacking has no separate Optuna search.

**Regression policy:** Ridge and Lasso minimize mean inner-CV RMSE over their
`alpha` search spaces. Inner folds use shuffled `KFold`, and outer evidence uses
`RepeatedKFold`. The regressor with the lowest mean outer-CV RMSE is selected;
MAE and R² are additional evidence metrics. LinearRegression and
PolynomialRegression are fitted without Optuna.

**Fixed method:** Seeded, sequential, independent TPE with
`n_startup_trials=20`, `multivariate=False`, `group=False`,
`constant_liar=False`, no pruning, in-memory studies, and study `n_jobs=1`.
Each sampler receives `RANDOM_STATE` explicitly. Conditional branches receive
adaptive allocation without quotas or an exhaustive coverage guarantee.
Normal runs whose effective `n_trials` does not exceed `n_startup_trials`
remain valid but warn that they remain within startup sampling and do not reach
model-based TPE. Smoke mode is intentionally exempt.

**Failure and evidence boundary:** Failed candidate evaluations may fail their
trial without aborting the study; contract, programming, and final refit
failures propagate. For tunable models, completed outer and final optimization
search summaries are persisted. Direct-fit-only runs may
produce header-only search artifacts. Full per-trial traces
are not persisted.

**Execution boundary:** Sklearn `cross_validate` remains the outer orchestrator,
with task-specific tunable searches adapted through sklearn-compatible Optuna
estimators. Classification tunable outer fits use `error_score="raise"`; direct
classification evaluation retains its historical sklearn error behavior.
Regression outer evaluation uses `error_score="raise"` for tunable and direct
models. Optuna startup accounting includes COMPLETE and PRUNED trials. MELITE's
no-op pruner makes PRUNED a contract violation, while FAIL consumes budget
without advancing startup accounting.

For either task, optimization within an outer split uses only its training
portion. Exact aggregate-metric ties preserve configured model order. After
selection, a tunable winner receives a fresh full-data optimization and fit;
non-tunable winners fit directly. Export later reconstructs the selected model
from the recorded final parameters and does not repeat optimization.

**Compatibility:** MELITE v0.3.0 validates Optuna 4.x. Optuna 5 may be adopted
only after it is stable and MELITE explicitly verifies compatible sampler
behavior, trial/error semantics, and optimization policy.

## Optimization Evidence and Provenance

**Status:** Fixed task-specific artifact contracts. The classification contract
was established for v0.3.0; the regression artifact family was added for
v0.3.2.

For classification, MELITE persists one row per completed optimization search,
not per trial, in `optimization_searches.csv`. Outer and final searches are
distinguished by `search_scope`; `selected` applies only to outer rows and is
not applicable to final rows. A Stack-only run legitimately produces a
header-only search artifact. `evaluations.csv` and `evaluation_folds.csv`
preserve aggregate and per-outer-split F1-macro, accuracy, and ROC-AUC evidence.
`results.csv` contains the selected classifier and its final parameters. The
F1-macro figure is derived from existing outer evidence without further model
fitting.

For regression, the corresponding files are
`regression_optimization_searches.csv`, `regression_evaluations.csv`,
`regression_evaluation_folds.csv`, and `regression_results.csv`. Search rows
exist only for Ridge and Lasso; a run containing only LinearRegression and/or
PolynomialRegression produces a header-only regression search artifact. RMSE,
MAE, and R² are preserved as aggregate and per-outer-split evidence, and the
selected result is determined by the lowest mean outer RMSE. Regression does
not currently produce an evaluation figure.

For both tasks, exact selection ties retain the first model in configured
order, and full trial traces are deliberately not persisted. Outer-CV evidence
describes the configured evaluation design; it is not evidence of external
scientific validation or generalizability.

`optimization_provenance.json` records the classification cross-validation
design, canonical `RANDOM_STATE`, effective budget, fixed policy, backend
identity and runtime version, and active classifier search spaces; Stack is
represented as `null`. `regression_optimization_provenance.json` records the
task-specific outer and inner strategies, RMSE minimization contract, active
regressors, and their search spaces; LinearRegression and PolynomialRegression
are represented as `null`. Filesystem, data, and environment provenance remain
outside these artifacts.

The MELITE package version governs these artifact schemas; no independent
`schema_version` is added. Optimization evidence is not an operational model
input: `results.csv` is the persisted classification parameter source used by
`melite export`, while regression export uses `regression_results.csv`.

## Public Classifier Extensibility

**Status:** Public classifier registration remains deferred.

**Resolved internal decision:** v0.3.0 work establishes a durable internal
search-space contract that can represent discrete, integer, continuous, and
conditional search policy without depending on one optimization backend.

**Reason:** The internal contract does not itself define a stable public API for
registering user classifiers and their estimator or artifact semantics.

**Revisit criterion:** Actual work begins on public classifier registration.

## v0.3.0 Optimization Characterization

**Status:** Characterization completed and passed for v0.3.0.

The calibration stage executes only the historical v0.2.5 GridSearchCV engine;
no v0.3.0 candidate metrics are inspected. The later comparison is defined as
the v0.2.5 engine versus the v0.3.0 engine on one shared current pinned
scientific stack, not as reproduction of the historical v0.2.5 environment.

The fixed synthetic generator uses 240 samples, 20 features, 12 informative
features, 4 redundant features, balanced binary classes, `flip_y=0.05`, and
`random_state=42`. Stage 1 evaluates SVC only at ordered `class_sep` candidates
0.60, 0.70, 0.80, 0.90, and 1.00. Eligible SVC means lie in `[0.65, 0.95]`;
the selected profile is closest to 0.80, with lower `class_sep` breaking an
exact tie.

Stage 2 evaluates SVC, RandomForest, and XGBoost exactly once on the selected
profile. Every classifier mean must lie in `[0.65, 0.95]`, and the selected
classifier mean must lie in `[0.70, 0.90]`. Failure at either stage stops the
protocol without candidate execution or automatic fallback. If B5-0 is
explicitly reopened after review, the predefined next axis is `n_informative`
with ordered candidates `(10, 14, 8, 16)`; additional outer repeats are not the
reopen axis.

Calibration is not smoke mode: outer CV is 5 folds × 1 repeat with 3 inner
folds. Later characterization uses only SVC, RandomForest, and XGBoost.
`fit_count` estimates `estimator.fit()` invocations and includes the
best-model refit performed by GridSearchCV. `end_to_end_wall_seconds` measures
the complete MELITE subprocess and is descriptive, not optimization-only
timing. Candidate comparison may proceed only after verifying the candidate Python
interpreter and common pinned dependencies against the baseline environment
recorded during the v0.3.0 calibration procedure, and after verifying the frozen selected
dataset bytes against the exact recorded SHA-256.

Candidate characterization reuses the committed baseline; v0.2.5 is not rerun.
The orchestrator Python version and all seven common dependency versions must
match the baseline evidence exactly. During v0.3.0 characterization, the experimental input dataset was frozen
byte-for-byte and protected by its recorded SHA-256 before MELITE executes. The
candidate uses a fresh
output directory, SVC, RandomForest, and XGBoost, 100 trials per search,
5 folds × 1 repeat outside, and 3 folds inside.

Every optimization row must account for its complete 100-trial budget
(`n_trials_complete + n_trials_failed == n_trials_requested`) and must have no
failed trials. Scientific acceptance requires each candidate classifier's mean
outer F1-macro delta from baseline to be at least `-0.05`; a winner change is
informational only, and the worst classifier delta is reported. Estimated fit
count and end-to-end wall-clock remain separate evidence: wall-clock is
descriptive and is not a runtime gate. The full verbose candidate console log
is retained locally for diagnostics.

The completed characterization produced the following mean outer-CV F1-macro
comparison:

| Classifier | v0.2.5 baseline | v0.3.0 candidate | Delta | Result |
|---|---:|---:|---:|---|
| SVC | `0.8153189729660317` | `0.8285693852008889` | `+0.013250412234857123` | Pass |
| RandomForest | `0.7660740105672047` | `0.7701989893283668` | `+0.004124978761162135` | Pass |
| XGBoost | `0.7534739915895242` | `0.7366799005213484` | `-0.016794091068175865` | Pass |

All three classifiers passed the predefined `-0.05` acceptance margin. SVC was
the baseline and candidate winner, so the winner did not change. Under the
static fit-count definition, the estimated workload fell from `42,208` fits
for v0.2.5 to `4,816` for v0.3.0: approximately 88.6% fewer estimator fits, or
approximately 8.8× fewer fits.

End-to-end wall-clock was `2981.05904740002 s` for the baseline and
`3005.0019774999237 s` for the candidate, a ratio of
`1.0080316859609963`. Wall-clock was descriptive and not an acceptance gate.
The fit-count reduction demonstrates a smaller declared optimization workload,
not a general wall-clock speedup claim: the historical GridSearchCV baseline
allowed parallel execution with `n_jobs=-1`, while the v0.3.0 Optuna policy is
sequential with `n_jobs=1`.

Under this frozen B5 characterization condition, v0.3.0 preserved classifier
selection and predictive performance within the predefined scientific margin.
These results must not be generalized to every dataset or search budget.
Parallel Optuna trials or searches may be reconsidered in a future version only
with separate characterization; v0.3.0 retains `n_jobs=1` and
`constant_liar=False`.
