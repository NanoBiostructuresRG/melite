# SPDX-License-Identifier: LGPL-3.0-or-later
"""Immutable internal classifier search-space contracts for MELITE.

The structures in this module describe search policy independently of any
optimization backend. v0.3.0 supports at most one conditional selector per
classifier; nested or multiple independent selectors are deliberately outside
this contract.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Any

__all__ = [
    "FloatDomain",
    "IntDomain",
    "CategoricalDomain",
    "ParameterSpec",
    "FixedParameterSpec",
    "BranchSelector",
    "SearchBranch",
    "ClassifierSearchSpace",
    "SEARCH_SPACE_POLICY",
    "get_search_space",
    "search_space_to_json",
    "search_space_from_json",
]


def _non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    return value


def _finite_real(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a finite real number, excluding bool.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field} must be finite.")
    return normalized


def _canonical_scalar(value: Any, field: str) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, Real):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError(f"{field} must be finite for canonical serialization.")
        return normalized
    raise TypeError(
        f"{field} must be a JSON-compatible scalar value for canonical serialization."
    )


def _ordered_unique_choices(value: Any, field: str) -> tuple[Any, ...]:
    if isinstance(value, (set, frozenset)):
        raise TypeError(f"{field} must be ordered; set and frozenset are invalid.")
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field} must be an ordered sequence.")

    choices = tuple(_canonical_scalar(choice, f"{field} choice") for choice in value)
    if len(choices) < 2:
        raise ValueError(f"{field} must contain at least two choices.")
    for index, choice in enumerate(choices):
        if any(choice == previous for previous in choices[:index]):
            raise ValueError(f"{field} must not contain duplicate choices.")
    return choices


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping.")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], context: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{context} must contain exactly {sorted(expected)}; got {sorted(actual)}."
        )


def _require_list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a list in canonical data.")
    return value


@dataclass(frozen=True)
class FloatDomain:
    """Continuous floating-point domain, optionally sampled logarithmically."""

    low: float
    high: float
    log: bool = False

    def __post_init__(self) -> None:
        low = _finite_real(self.low, "FloatDomain.low")
        high = _finite_real(self.high, "FloatDomain.high")
        if not isinstance(self.log, bool):
            raise TypeError("FloatDomain.log must be bool.")
        if low >= high:
            raise ValueError("FloatDomain.low must be less than high.")
        if self.log and low <= 0:
            raise ValueError("FloatDomain with log=True requires low > 0.")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "float", "low": self.low, "high": self.high, "log": self.log}


@dataclass(frozen=True)
class IntDomain:
    """Inclusive integer domain with an exactly compatible step."""

    low: int
    high: int
    step: int = 1

    def __post_init__(self) -> None:
        for field, value in (
            ("IntDomain.low", self.low),
            ("IntDomain.high", self.high),
            ("IntDomain.step", self.step),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field} must be an int, excluding bool.")
        if self.low >= self.high:
            raise ValueError("IntDomain.low must be less than high.")
        if self.step <= 0:
            raise ValueError("IntDomain.step must be positive.")
        if (self.high - self.low) % self.step != 0:
            raise ValueError("IntDomain interval must be exactly compatible with step.")

    def to_dict(self) -> dict[str, Any]:
        return {"type": "int", "low": self.low, "high": self.high, "step": self.step}


@dataclass(frozen=True)
class CategoricalDomain:
    """Ordered categorical domain with unique serializable choices."""

    choices: tuple[Any, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "choices",
            _ordered_unique_choices(self.choices, "CategoricalDomain.choices"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"type": "categorical", "choices": list(self.choices)}


Domain = FloatDomain | IntDomain | CategoricalDomain


def _domain_from_dict(value: Any) -> Domain:
    data = _require_mapping(value, "domain")
    domain_type = data.get("type")
    if domain_type == "float":
        _require_exact_keys(data, {"type", "low", "high", "log"}, "FloatDomain")
        return FloatDomain(low=data["low"], high=data["high"], log=data["log"])
    if domain_type == "int":
        _require_exact_keys(data, {"type", "low", "high", "step"}, "IntDomain")
        return IntDomain(low=data["low"], high=data["high"], step=data["step"])
    if domain_type == "categorical":
        _require_exact_keys(data, {"type", "choices"}, "CategoricalDomain")
        return CategoricalDomain(
            choices=tuple(_require_list(data["choices"], "CategoricalDomain.choices"))
        )
    raise ValueError(f"Unknown domain type: {domain_type!r}.")


@dataclass(frozen=True)
class ParameterSpec:
    """Searchable logical parameter mapped to an estimator parameter target."""

    name: str
    target: str
    domain: Domain

    def __post_init__(self) -> None:
        _non_empty_string(self.name, "ParameterSpec.name")
        _non_empty_string(self.target, "ParameterSpec.target")
        if not isinstance(self.domain, (FloatDomain, IntDomain, CategoricalDomain)):
            raise TypeError("ParameterSpec.domain must be a supported domain.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target": self.target,
            "domain": self.domain.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Any) -> ParameterSpec:
        data = _require_mapping(value, "ParameterSpec")
        _require_exact_keys(data, {"name", "target", "domain"}, "ParameterSpec")
        return cls(
            name=data["name"],
            target=data["target"],
            domain=_domain_from_dict(data["domain"]),
        )


@dataclass(frozen=True)
class FixedParameterSpec:
    """Fixed branch parameter mapped to an estimator parameter target."""

    name: str
    target: str
    value: Any

    def __post_init__(self) -> None:
        _non_empty_string(self.name, "FixedParameterSpec.name")
        _non_empty_string(self.target, "FixedParameterSpec.target")
        object.__setattr__(
            self,
            "value",
            _canonical_scalar(self.value, "FixedParameterSpec.value"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "target": self.target, "value": self.value}

    @classmethod
    def from_dict(cls, value: Any) -> FixedParameterSpec:
        data = _require_mapping(value, "FixedParameterSpec")
        _require_exact_keys(data, {"name", "target", "value"}, "FixedParameterSpec")
        return cls(name=data["name"], target=data["target"], value=data["value"])


@dataclass(frozen=True)
class BranchSelector:
    """Single ordered selector controlling conditional search branches."""

    name: str
    target: str | None
    choices: tuple[Any, ...]

    def __post_init__(self) -> None:
        _non_empty_string(self.name, "BranchSelector.name")
        if self.target is not None:
            _non_empty_string(self.target, "BranchSelector.target")
        elif not self.name.endswith("_mode"):
            raise ValueError(
                "BranchSelector with target=None requires a name ending in '_mode'."
            )
        object.__setattr__(
            self,
            "choices",
            _ordered_unique_choices(self.choices, "BranchSelector.choices"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "target": self.target, "choices": list(self.choices)}

    @classmethod
    def from_dict(cls, value: Any) -> BranchSelector:
        data = _require_mapping(value, "BranchSelector")
        _require_exact_keys(data, {"name", "target", "choices"}, "BranchSelector")
        return cls(
            name=data["name"],
            target=data["target"],
            choices=tuple(_require_list(data["choices"], "BranchSelector.choices")),
        )


@dataclass(frozen=True)
class SearchBranch:
    """Parameters effective for one selector value."""

    selector_value: Any
    fixed_parameters: tuple[FixedParameterSpec, ...] = ()
    parameters: tuple[ParameterSpec, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selector_value",
            _canonical_scalar(self.selector_value, "SearchBranch.selector_value"),
        )
        object.__setattr__(self, "fixed_parameters", tuple(self.fixed_parameters))
        object.__setattr__(self, "parameters", tuple(self.parameters))
        if not all(
            isinstance(parameter, FixedParameterSpec)
            for parameter in self.fixed_parameters
        ):
            raise TypeError(
                "SearchBranch.fixed_parameters must contain FixedParameterSpec."
            )
        if not all(
            isinstance(parameter, ParameterSpec) for parameter in self.parameters
        ):
            raise TypeError("SearchBranch.parameters must contain ParameterSpec.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "selector_value": self.selector_value,
            "fixed_parameters": [
                parameter.to_dict() for parameter in self.fixed_parameters
            ],
            "parameters": [parameter.to_dict() for parameter in self.parameters],
        }

    @classmethod
    def from_dict(cls, value: Any) -> SearchBranch:
        data = _require_mapping(value, "SearchBranch")
        _require_exact_keys(
            data,
            {"selector_value", "fixed_parameters", "parameters"},
            "SearchBranch",
        )
        return cls(
            selector_value=data["selector_value"],
            fixed_parameters=tuple(
                FixedParameterSpec.from_dict(parameter)
                for parameter in _require_list(
                    data["fixed_parameters"], "SearchBranch.fixed_parameters"
                )
            ),
            parameters=tuple(
                ParameterSpec.from_dict(parameter)
                for parameter in _require_list(
                    data["parameters"], "SearchBranch.parameters"
                )
            ),
        )


def _duplicate(values: Sequence[Any]) -> Any | None:
    for index, value in enumerate(values):
        if any(value == previous for previous in values[:index]):
            return value
    return None


@dataclass(frozen=True)
class ClassifierSearchSpace:
    """One classifier's backend-independent search-space contract.

    At most one conditional selector is supported. Each effective branch path
    must have unique logical names and unique estimator targets. Across
    mutually exclusive branches, one logical name maps to one estimator target,
    and searchable occurrences use the same domain; fixed-in-one and
    searchable-in-another remains valid with the same target. This is a
    deliberate v0.3.0 boundary that requires an explicit contract revision to
    support branch-specific domains for one logical parameter.
    """

    classifier: str
    common_parameters: tuple[ParameterSpec, ...] = ()
    selector: BranchSelector | None = None
    branches: tuple[SearchBranch, ...] = ()

    def __post_init__(self) -> None:
        _non_empty_string(self.classifier, "ClassifierSearchSpace.classifier")
        object.__setattr__(self, "common_parameters", tuple(self.common_parameters))
        object.__setattr__(self, "branches", tuple(self.branches))
        if not all(
            isinstance(parameter, ParameterSpec) for parameter in self.common_parameters
        ):
            raise TypeError(
                "ClassifierSearchSpace.common_parameters must contain ParameterSpec."
            )
        if self.selector is not None and not isinstance(self.selector, BranchSelector):
            raise TypeError(
                "ClassifierSearchSpace.selector must be BranchSelector or None."
            )
        if not all(isinstance(branch, SearchBranch) for branch in self.branches):
            raise TypeError("ClassifierSearchSpace.branches must contain SearchBranch.")

        if self.selector is None:
            if self.branches:
                raise ValueError("selector=None requires no branches.")
            if not self.common_parameters:
                raise ValueError(
                    "A tunable classifier without a selector requires at least one "
                    "common parameter."
                )
            self._validate_effective_path(())
            return
        if not self.branches:
            raise ValueError("A selector requires branches.")

        branch_values = tuple(branch.selector_value for branch in self.branches)
        unknown = [
            value for value in branch_values if value not in self.selector.choices
        ]
        if unknown:
            raise ValueError(f"Unknown branch selector_value(s): {unknown}.")
        duplicate = _duplicate(branch_values)
        if duplicate is not None:
            raise ValueError(f"Duplicate branch for selector_value {duplicate!r}.")
        missing = [
            choice for choice in self.selector.choices if choice not in branch_values
        ]
        if missing:
            raise ValueError(f"Missing branch(es) for selector choice(s): {missing}.")
        if self.selector.target is None and not any(
            branch.fixed_parameters or branch.parameters for branch in self.branches
        ):
            raise ValueError(
                "A MELITE-only selector requires at least one branch with a fixed "
                "or searchable parameter."
            )

        for branch in self.branches:
            self._validate_effective_path((branch,))
        self._validate_branch_identity()

    def _validate_branch_identity(self) -> None:
        targets: dict[str, str] = {}
        searchable_domains: dict[str, Domain] = {}

        for branch in self.branches:
            branch_parameters: tuple[FixedParameterSpec | ParameterSpec, ...] = (
                *branch.fixed_parameters,
                *branch.parameters,
            )
            for parameter in branch_parameters:
                existing_target = targets.get(parameter.name)
                if existing_target is not None and existing_target != parameter.target:
                    raise ValueError(
                        f"Logical parameter {parameter.name!r} maps to inconsistent "
                        "estimator targets across branches."
                    )
                targets.setdefault(parameter.name, parameter.target)

                if isinstance(parameter, ParameterSpec):
                    existing_domain = searchable_domains.get(parameter.name)
                    if (
                        existing_domain is not None
                        and existing_domain != parameter.domain
                    ):
                        raise ValueError(
                            f"Searchable logical parameter {parameter.name!r} uses "
                            "inconsistent domains across branches."
                        )
                    searchable_domains.setdefault(parameter.name, parameter.domain)

    def branch_for(self, selector_value: Any) -> SearchBranch:
        """Return the branch selected by ``selector_value``."""
        if self.selector is None:
            raise ValueError(
                f"Classifier {self.classifier!r} does not define a branch selector."
            )

        for branch in self.branches:
            if branch.selector_value == selector_value:
                return branch

        raise KeyError(
            f"Unknown selector value {selector_value!r} for classifier "
            f"{self.classifier!r}."
        )

    def _validate_effective_path(self, branches: tuple[SearchBranch, ...]) -> None:
        logical_names = [parameter.name for parameter in self.common_parameters]
        estimator_targets = [parameter.target for parameter in self.common_parameters]
        if self.selector is not None:
            logical_names.append(self.selector.name)
            if self.selector.target is not None:
                estimator_targets.append(self.selector.target)
        for branch in branches:
            logical_names.extend(
                parameter.name for parameter in branch.fixed_parameters
            )
            logical_names.extend(parameter.name for parameter in branch.parameters)
            estimator_targets.extend(
                parameter.target for parameter in branch.fixed_parameters
            )
            estimator_targets.extend(
                parameter.target for parameter in branch.parameters
            )

        duplicate_name = _duplicate(logical_names)
        if duplicate_name is not None:
            raise ValueError(
                f"Logical parameter name collision on effective branch path: "
                f"{duplicate_name!r}."
            )
        duplicate_target = _duplicate(estimator_targets)
        if duplicate_target is not None:
            raise ValueError(
                f"Estimator target collision on effective branch path: "
                f"{duplicate_target!r}."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "classifier": self.classifier,
            "common_parameters": [
                parameter.to_dict() for parameter in self.common_parameters
            ],
            "selector": self.selector.to_dict() if self.selector is not None else None,
            "branches": [branch.to_dict() for branch in self.branches],
        }

    @classmethod
    def from_dict(cls, value: Any) -> ClassifierSearchSpace:
        data = _require_mapping(value, "ClassifierSearchSpace")
        _require_exact_keys(
            data,
            {"classifier", "common_parameters", "selector", "branches"},
            "ClassifierSearchSpace",
        )
        selector_data = data["selector"]
        return cls(
            classifier=data["classifier"],
            common_parameters=tuple(
                ParameterSpec.from_dict(parameter)
                for parameter in _require_list(
                    data["common_parameters"],
                    "ClassifierSearchSpace.common_parameters",
                )
            ),
            selector=(
                None
                if selector_data is None
                else BranchSelector.from_dict(selector_data)
            ),
            branches=tuple(
                SearchBranch.from_dict(branch)
                for branch in _require_list(
                    data["branches"], "ClassifierSearchSpace.branches"
                )
            ),
        )


def search_space_to_json(search_space: ClassifierSearchSpace) -> str:
    """Serialize one search-space contract to deterministic canonical JSON."""
    if not isinstance(search_space, ClassifierSearchSpace):
        raise TypeError("search_space must be a ClassifierSearchSpace.")
    return json.dumps(
        search_space.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def search_space_from_json(payload: str) -> ClassifierSearchSpace:
    """Deserialize canonical JSON into a validated search-space contract."""
    if not isinstance(payload, str):
        raise TypeError("payload must be a JSON string.")
    return ClassifierSearchSpace.from_dict(json.loads(payload))


_SVC_SEARCH_SPACE = ClassifierSearchSpace(
    classifier="svc",
    common_parameters=(
        ParameterSpec("C", "svc__C", FloatDomain(0.01, 20.0, log=True)),
    ),
    selector=BranchSelector(
        name="kernel",
        target="svc__kernel",
        choices=("linear", "rbf", "poly"),
    ),
    branches=(
        SearchBranch(selector_value="linear"),
        SearchBranch(
            selector_value="rbf",
            parameters=(
                ParameterSpec(
                    "gamma",
                    "svc__gamma",
                    FloatDomain(0.001, 0.2, log=True),
                ),
            ),
        ),
        SearchBranch(
            selector_value="poly",
            parameters=(
                ParameterSpec(
                    "gamma",
                    "svc__gamma",
                    FloatDomain(0.001, 0.2, log=True),
                ),
                ParameterSpec(
                    "coef0",
                    "svc__coef0",
                    FloatDomain(0.0, 1.0, log=False),
                ),
                ParameterSpec(
                    "degree",
                    "svc__degree",
                    IntDomain(3, 5, step=1),
                ),
            ),
        ),
    ),
)

_RF_SEARCH_SPACE = ClassifierSearchSpace(
    classifier="rf",
    common_parameters=(
        ParameterSpec(
            "n_estimators",
            "n_estimators",
            CategoricalDomain((200, 400, 800)),
        ),
        ParameterSpec(
            "max_features",
            "max_features",
            CategoricalDomain(("sqrt", "log2")),
        ),
        ParameterSpec(
            "min_samples_split",
            "min_samples_split",
            IntDomain(2, 5, step=1),
        ),
        ParameterSpec(
            "min_samples_leaf",
            "min_samples_leaf",
            IntDomain(1, 2, step=1),
        ),
    ),
    selector=BranchSelector(
        name="depth_mode",
        target=None,
        choices=("unbounded", "bounded"),
    ),
    branches=(
        SearchBranch(
            selector_value="unbounded",
            fixed_parameters=(FixedParameterSpec("max_depth", "max_depth", None),),
        ),
        SearchBranch(
            selector_value="bounded",
            parameters=(
                ParameterSpec(
                    "max_depth",
                    "max_depth",
                    IntDomain(10, 40, step=1),
                ),
            ),
        ),
    ),
)

_XGB_SEARCH_SPACE = ClassifierSearchSpace(
    classifier="xgb",
    common_parameters=(
        ParameterSpec(
            "n_estimators",
            "n_estimators",
            IntDomain(300, 600, step=1),
        ),
        ParameterSpec(
            "learning_rate",
            "learning_rate",
            FloatDomain(0.01, 0.1, log=True),
        ),
        ParameterSpec("max_depth", "max_depth", IntDomain(4, 8, step=1)),
        ParameterSpec(
            "subsample",
            "subsample",
            FloatDomain(0.7, 0.85, log=False),
        ),
        ParameterSpec(
            "colsample_bytree",
            "colsample_bytree",
            FloatDomain(0.7, 1.0, log=False),
        ),
        ParameterSpec(
            "reg_alpha",
            "reg_alpha",
            FloatDomain(0.0, 0.5, log=False),
        ),
        ParameterSpec(
            "reg_lambda",
            "reg_lambda",
            FloatDomain(1.0, 2.0, log=False),
        ),
    ),
    selector=BranchSelector(
        name="gamma_mode",
        target=None,
        choices=("zero", "positive"),
    ),
    branches=(
        SearchBranch(
            selector_value="zero",
            fixed_parameters=(FixedParameterSpec("gamma", "gamma", 0.0),),
        ),
        SearchBranch(
            selector_value="positive",
            parameters=(
                ParameterSpec(
                    "gamma",
                    "gamma",
                    FloatDomain(0.01, 5.0, log=True),
                ),
            ),
        ),
    ),
)


SEARCH_SPACE_POLICY: Mapping[str, ClassifierSearchSpace | None] = MappingProxyType(
    {
        "svc": _SVC_SEARCH_SPACE,
        "rf": _RF_SEARCH_SPACE,
        "xgb": _XGB_SEARCH_SPACE,
        "stack": None,
    }
)


def get_search_space(classifier_key: str) -> ClassifierSearchSpace | None:
    """Return a built-in classifier policy or fail for an unknown key."""
    try:
        return SEARCH_SPACE_POLICY[classifier_key]
    except KeyError:
        raise KeyError(f"Unknown MELITE classifier key: {classifier_key!r}.") from None
