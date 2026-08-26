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

## Public Classifier Extensibility

**Status:** Public classifier registration remains deferred.

**Resolved internal decision:** v0.3.0 work establishes a durable internal
search-space contract that can represent discrete, integer, continuous, and
conditional search policy without depending on one optimization backend.

**Reason:** The internal contract does not itself define a stable public API for
registering user classifiers and their estimator or artifact semantics.

**Revisit criterion:** Actual work begins on public classifier registration.
