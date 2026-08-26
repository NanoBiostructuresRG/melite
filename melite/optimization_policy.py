# SPDX-License-Identifier: LGPL-3.0-or-later
"""Declare MELITE's backend-independent v0.3.0 optimization policy.

The policy specifies one fresh in-memory study per tunable search using seeded,
sequential, independent TPE. Every sampler receives ``Config.RANDOM_STATE``
explicitly; conditional branches receive adaptive allocation without quotas or
an exhaustive coverage guarantee. Studies use no pruning and optimize with
``n_jobs=1``.

Candidate-evaluation failures are failed trials that do not stop the study, and
failed trials consume the requested budget. Contract, translation, and
programming errors propagate. Zero completed trials are an explicit search
failure, as is a best-configuration refit failure.

A normal run whose effective ``N_TRIALS`` is less than or equal to
``OPTIMIZATION_POLICY.n_startup_trials`` is valid, but the contract requires a
warning that its budget remains within startup sampling and does not reach
model-based TPE. Smoke mode is exempt because its five-trial budget exists only
to exercise the execution workflow.

Normal Optuna verbosity is WARNING and MELITE verbose mode uses INFO, with
Optuna global verbosity restored after the complete ``Main.run()`` scope.
Level-2 outer-search evidence is required, but full per-trial traces are not
persisted. Optimization execution is implemented by the optimization engine;
this policy module does not execute studies, logging scopes, or persistence.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class _OptimizationPolicy:
    sampler: str
    n_startup_trials: int
    smoke_n_trials: int
    multivariate: bool
    group: bool
    constant_liar: bool
    pruning: bool
    storage: str
    n_jobs: int
    direction: str
    objective: str


OPTIMIZATION_POLICY = _OptimizationPolicy(
    sampler="tpe",
    n_startup_trials=20,
    smoke_n_trials=5,
    multivariate=False,
    group=False,
    constant_liar=False,
    pruning=False,
    storage="in_memory",
    n_jobs=1,
    direction="maximize",
    objective="f1_macro",
)
