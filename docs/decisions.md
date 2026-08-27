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

**Status:** Fixed design and runtime architecture for v0.3.0. Optuna is the
single optimization engine for tunable classifiers; coexistence with
`GridSearchCV` is not part of the design. The only optimization-specific user
setting is `n_trials`,
with a normal default of 100. The sampler seed derives from the existing
canonical `RANDOM_STATE`; MELITE does not expose a separate optimization seed.
The smoke budget is an internal, non-configurable 5 trials.

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
failures propagate. Level-2 outer-search evidence is required, while full
per-trial traces are not.

**Execution boundary:** Sklearn `cross_validate` remains the outer orchestrator,
with tunable searches adapted through a sklearn-compatible Optuna estimator.
Tunable outer fits use `error_score="raise"` so fatal optimization failures are
not converted to NaN; direct non-tunable evaluation retains historical sklearn
error behavior. Optuna startup accounting includes COMPLETE and PRUNED trials.
MELITE's no-op pruner makes PRUNED a contract violation, while FAIL consumes
budget without advancing startup accounting.

**Compatibility:** MELITE v0.3.0 validates Optuna 4.x. Optuna 5 may be adopted
only after it is stable and MELITE explicitly verifies compatible sampler
behavior, trial/error semantics, and optimization policy.

## Optimization Evidence and Provenance

**Status:** Fixed for v0.3.0.

MELITE persists one row per complete optimization search, not per trial.
Outer and final searches share `optimization_searches.csv` through
`search_scope`; `selected` applies only to outer rows and is not applicable to
final rows. A Stack-only run legitimately produces a header-only optimization
search artifact. Full trial traces are deliberately not persisted.

Optimization provenance records the effective cross-validation design,
canonical `RANDOM_STATE`, effective optimization budget, fixed optimization
policy, backend identity and runtime version, and only the active search-space
contracts. Stack is represented as `null`. Filesystem, data, and environment
provenance are outside B4.

The MELITE package version governs these artifact schemas; no independent
`schema_version` is added. Optimization evidence is not an operational model
input: `results.csv` remains the sole persisted parameter source used by
`melite export`.

## Public Classifier Extensibility

**Status:** Public classifier registration remains deferred.

**Resolved internal decision:** v0.3.0 work establishes a durable internal
search-space contract that can represent discrete, integer, continuous, and
conditional search policy without depending on one optimization backend.

**Reason:** The internal contract does not itself define a stable public API for
registering user classifiers and their estimator or artifact semantics.

**Revisit criterion:** Actual work begins on public classifier registration.

## v0.3.0 Optimization Characterization

**Status:** Baseline calibration fixed; candidate characterization protocol fixed.

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
recorded in `B5_calibration.json`, and after verifying the frozen selected
dataset bytes against the exact recorded SHA-256.

Candidate characterization reuses the committed baseline; v0.2.5 is not rerun.
The orchestrator Python version and all seven common dependency versions must
match the baseline evidence exactly. The characterization dataset is frozen
byte-for-byte in `scripts/b5_characterization_dataset.csv` as an experimental
input and protected by its recorded SHA-256 before MELITE executes. The
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
