# Contributing to MELITE

Thank you for your interest in contributing to MELITE.

MELITE is maintained by the
[NanoBiostructures Research Group](https://nanobiostructuresrg.github.io)
at Tecnológico de Monterrey. Contributions that improve correctness,
reproducibility, usability, documentation, testing, or supported workflows are
welcome.

## How to Contribute

### Reporting Bugs

Open an issue on
[GitHub Issues](https://github.com/NanoBiostructuresRG/melite/issues)
and include, when relevant:

- a clear description of the problem;
- the expected and observed behavior;
- a minimal reproducible example;
- the MELITE and Python versions;
- the operating system and installation method;
- the command, API call, or configuration involved;
- the relevant traceback or error output.

Do not include confidential, proprietary, or sensitive datasets in public
issues.

### Suggesting Features

Open an issue describing:

- the problem or use case;
- the proposed behavior;
- why it would be useful beyond a single workflow;
- any expected effect on the public API, CLI, configuration, outputs, or
  evaluation contract.

### Submitting a Pull Request

1. Fork the repository.
2. Create a descriptive branch from `main`.
3. Make a focused set of changes.
4. Add or update tests when behavior changes.
5. Run the relevant validation locally.
6. Push your branch and open a pull request against `main`.

Pull requests should pass CI before merge. User-facing changes should include
appropriate documentation and changelog updates.

## Development Setup

Create an environment with a supported Python version using the environment
manager you prefer, then install MELITE in editable mode with its development
dependencies:

```bash
git clone https://github.com/NanoBiostructuresRG/melite.git
cd melite
python -m pip install -e ".[dev]"
```

To work on the documentation:

```bash
python -m pip install -e ".[dev,docs]"
mkdocs serve
```

## Testing

Run the complete test suite before opening a pull request:

```bash
python -m pytest tests -q
```

Changes should include tests when they modify behavior, fix a bug, add or
modify a classifier, change configuration handling, alter generated outputs,
or affect a public interface.

For documentation changes, also run:

```bash
mkdocs build --strict
```

## Code Style

Keep changes focused and consistent with the existing codebase. Use clear
names, type hints where appropriate, and NumPy-style docstrings consistent with
the surrounding modules.

Avoid unrelated refactoring or formatting changes in the same pull request.

## Scientific and Evaluation Changes

Changes that affect classifier tuning, cross-validation, scoring, classifier
construction, classifier selection, evaluation evidence, or final model fitting
require particular care.

Such contributions should:

- explain the scientific or methodological rationale;
- preserve the separation between hyperparameter tuning and outer evaluation;
- avoid information leakage between tuning and evaluation;
- identify any change to the classifier-selection criterion or evaluation contract;
- add or update tests that exercise the changed behavior;
- preserve reproducibility through explicit configuration and random-state
  handling where applicable;
- update documentation and the changelog when user-visible behavior changes.

## Documentation and Changelog

Update the relevant documentation when a change affects:

- public behavior;
- the CLI or Python API;
- configuration;
- supported classifiers;
- input requirements;
- generated outputs;
- examples or documented workflows.

Add an entry to `CHANGELOG.md` for user-visible changes. Purely internal
refactoring normally does not require a changelog entry unless it changes
observable behavior.

## Compatibility

MELITE is pre-stable, so public interfaces may evolve before version 1.0.
Compatibility changes should nevertheless be intentional, documented, tested,
and explained in the pull request.

Avoid changing public API symbols, CLI behavior, configuration keys, output
schemas, or serialized-artifact behavior without explicit justification.

## Pull Request Scope

Keep pull requests focused on one coherent change. Avoid combining feature
work, broad refactoring, formatting changes, and unrelated cleanup in the same
pull request.

Small, reviewable changes are easier to validate scientifically and
technically.

## Code of Conduct

Participation in MELITE is governed by the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Questions

For questions about contributing, open a GitHub issue. Sensitive
conduct-related matters should follow the private reporting process described
in the [Code of Conduct](CODE_OF_CONDUCT.md).
