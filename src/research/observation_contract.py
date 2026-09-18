"""Versioned research observations. Never infer a version from vector length.

The operator approved five slots, not a five-state HMM: K remains chosen on
development by the existing BIC rule. The legacy order/hash are not rewritten.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from src.research.features import FEATURE_ORDER, GROUPS

LEGACY_VERSION = "research37_v2"
REGIME5_VERSION = "research_regime5_v1"


@dataclass(frozen=True)
class ObservationContract:
    version: str
    order: tuple[str, ...]
    regime_slots: int
    minimum_k: int

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "order": list(self.order),
            "n_features": len(self.order),
            "regime_slots": self.regime_slots,
            "minimum_k": self.minimum_k,
            "state_order": "ascending_frozen_vol_order",
            "padding": "zeros_after_k",
            "posterior_repair": "forbidden",
        }

    def posterior(self, values, k: int) -> np.ndarray:
        if (
            isinstance(k, bool | np.bool_)
            or not isinstance(k, int | np.integer)
            or not self.minimum_k <= k <= self.regime_slots
        ):
            raise ValueError("K is outside the observation contract; truncation forbidden")
        if np.ma.isMaskedArray(values) and np.ma.getmaskarray(values).any():
            raise ValueError("masked posterior")
        if isinstance(values, list | tuple) and any(isinstance(v, bool | np.bool_) for v in values):
            raise ValueError("boolean posterior")
        arr = np.asarray(values)
        if (
            arr.shape != (k,)
            or arr.dtype.kind not in "fiu"
            or not np.isfinite(arr).all()
            or (arr < 0).any()
            or (arr > 1).any()
            or not np.isclose(arr.sum(), 1.0, atol=1e-9, rtol=0)
        ):
            raise ValueError("posterior must be finite, nonnegative and sum to one")
        return np.pad(arr.astype(float), (0, self.regime_slots - k))

    def validate_arrays(self, market, context, *, bars: int) -> None:
        n_market = len(self.order) - len(GROUPS["posicion"]) - 3 - self.regime_slots
        for value, shape in ((market, (bars, n_market)), (context, (3 + self.regime_slots,))):
            arr = np.asarray(value)
            if (
                arr.shape != shape
                or arr.dtype.kind not in "fiu"
                or not np.isfinite(arr).all()
                or (np.abs(arr) > 5).any()
            ):
                raise ValueError(f"observation arrays violate {self.version}")


def observation_contract(version: str = LEGACY_VERSION) -> ObservationContract:
    if version == LEGACY_VERSION:
        return ObservationContract(version, tuple(FEATURE_ORDER), 4, 1)
    if version == REGIME5_VERSION:
        if len(FEATURE_ORDER) != 37 or tuple(GROUPS["regimen"]) != tuple(
            f"p_regime_{i}" for i in range(4)
        ):
            raise ValueError("legacy schema changed; new contract requires explicit review")
        return ObservationContract(version, (*FEATURE_ORDER, "p_regime_4"), 5, 2)
    raise ValueError(f"unknown observation version: {version!r}")


def require_model_contract(model, version: str) -> None:
    """New models must carry a verified binding, not just a convenient input width."""
    contract = observation_contract(version)
    shape = getattr(getattr(model, "observation_space", None), "shape", None)
    if shape is not None and shape != (len(contract.order),):
        raise ValueError("checkpoint observation shape mismatch")
    if version != LEGACY_VERSION and (
        getattr(model, "research_observation_sha256", None) != contract.sha256
        or shape != (len(contract.order),)
    ):
        raise ValueError("checkpoint observation identity/shape mismatch")
