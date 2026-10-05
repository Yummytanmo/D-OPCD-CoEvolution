"""Shared warnings for advisory LLM-output limits that never discard content."""

from __future__ import annotations

import warnings


class SoftConstraintWarning(UserWarning):
    """An LLM output exceeded a preferred shape but was retained in full."""


def warn_soft_limit(
    name: str,
    *,
    actual: int,
    preferred_max: int,
    unit: str,
) -> None:
    if int(actual) <= int(preferred_max):
        return
    warnings.warn(
        f"Soft constraint exceeded for {name}: preferred at most "
        f"{int(preferred_max)} {unit}, received {int(actual)}; retained all content.",
        SoftConstraintWarning,
        stacklevel=2,
    )


def warn_soft_range(
    name: str,
    *,
    actual: int,
    preferred_min: int,
    preferred_max: int,
    unit: str,
) -> None:
    value = int(actual)
    if int(preferred_min) <= value <= int(preferred_max):
        return
    warnings.warn(
        f"Soft constraint missed for {name}: preferred "
        f"{int(preferred_min)}-{int(preferred_max)} {unit}, received {value}; "
        "retained all content.",
        SoftConstraintWarning,
        stacklevel=2,
    )


__all__ = ["SoftConstraintWarning", "warn_soft_limit", "warn_soft_range"]
