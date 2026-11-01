# SPDX-License-Identifier: LGPL-3.0-or-later
"""Immutable search-space contracts for MELITE regressors."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from melite.search_spaces import FloatDomain, ParameterSpec

__all__ = [
    "RegressionSearchSpace",
    "REGRESSION_SEARCH_SPACE_POLICY",
    "get_regression_search_space",
]


@dataclass(frozen=True, slots=True)
class RegressionSearchSpace:
    """Searchable parameters for one regression model."""

    regressor: str
    parameters: tuple[ParameterSpec, ...]


_ALPHA_DOMAIN = FloatDomain(1e-4, 1e2, log=True)

REGRESSION_SEARCH_SPACE_POLICY: Mapping[str, RegressionSearchSpace | None] = (
    MappingProxyType(
        {
            "linear": None,
            "poly": None,
            "ridge": RegressionSearchSpace(
                regressor="ridge",
                parameters=(ParameterSpec("alpha", "ridge__alpha", _ALPHA_DOMAIN),),
            ),
            "lasso": RegressionSearchSpace(
                regressor="lasso",
                parameters=(ParameterSpec("alpha", "lasso__alpha", _ALPHA_DOMAIN),),
            ),
        }
    )
)


def get_regression_search_space(model_key: str) -> RegressionSearchSpace | None:
    """Return a built-in regression search space or fail for an unknown key."""
    try:
        return REGRESSION_SEARCH_SPACE_POLICY[model_key]
    except KeyError:
        raise KeyError(f"Unknown MELITE regressor key: {model_key!r}.") from None
