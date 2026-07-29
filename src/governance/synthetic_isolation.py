"""Fail-closed boundary for synthetic/demo models (BL-43)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class SyntheticIsolationError(ValueError):
    """Raised when synthetic content attempts to cross into a real surface."""


_DEMO_RELATION = "demo.synthetic_model"
_REQUIRED_SYNTHETIC_VALUES = {
    "algorithm": "SYNTHETIC",
    "environment": "demo",
    "surface": "synthetic",
    "execution_eligible": False,
}


def validate_model_boundary(
    model: Mapping[str, Any],
    *,
    relation: str,
) -> None:
    """Validate that synthetic models exist only in the isolated demo domain.

    Real models may never be stored in ``demo.synthetic_model`` and synthetic
    models may never be stored anywhere else.  The comparison is deliberately
    case-sensitive so malformed values cannot be normalized into eligibility.
    """

    if not isinstance(model, Mapping):
        raise SyntheticIsolationError("model metadata must be a mapping")

    algorithm = model.get("algorithm")
    if relation == _DEMO_RELATION:
        missing = [
            key
            for key, expected in _REQUIRED_SYNTHETIC_VALUES.items()
            if model.get(key) != expected
        ]
        if missing:
            raise SyntheticIsolationError(
                "demo.synthetic_model requires exact synthetic metadata: "
                + ", ".join(missing)
            )
        if not isinstance(model.get("model_id"), str) or not model["model_id"].strip():
            raise SyntheticIsolationError("synthetic model_id must be non-empty")
        return

    if algorithm == "SYNTHETIC":
        raise SyntheticIsolationError(
            f"synthetic model cannot be stored in real relation {relation!r}"
        )

    for key in ("environment", "surface", "execution_eligible"):
        if model.get(key) == _REQUIRED_SYNTHETIC_VALUES[key]:
            raise SyntheticIsolationError(
                f"demo marker {key!r} cannot appear in real relation {relation!r}"
            )
