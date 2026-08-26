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

**Status:** Fixed target design for v0.3.0. Optuna is the single optimization
engine for tunable classifiers; coexistence with `GridSearchCV` is not part of
the target design. The only optimization-specific user setting is `n_trials`,
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

**Compatibility:** MELITE v0.3.0 validates Optuna 4.x. Optuna 5 may be adopted
only after it is stable and MELITE explicitly verifies compatible sampler
behavior, trial/error semantics, and optimization policy.

## Public Classifier Extensibility

**Status:** Public classifier registration remains deferred.

**Resolved internal decision:** v0.3.0 work establishes a durable internal
search-space contract that can represent discrete, integer, continuous, and
conditional search policy without depending on one optimization backend.

**Reason:** The internal contract does not itself define a stable public API for
registering user classifiers and their estimator or artifact semantics.

**Revisit criterion:** Actual work begins on public classifier registration.
