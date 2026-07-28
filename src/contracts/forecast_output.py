"""
Forecast Output Contract (DIAGNOSTIC surface)
=============================================
Typed prediction record for the forecasting zoo (FABRIC §15.2 / BL-15).

A ``ForecastOutput`` is a DIAGNOSTIC artifact: it estimates a price/return for a
horizon and NOTHING else. It produces no decision, no PnL, no orders. The
allocator/book accepts exclusively ``strategy_output`` records validated by
contract — a ``ForecastOutput`` is rejected by TYPE before any logic runs
(physical rejection, not convention):

- ``diagnostic_only`` is forced to ``True`` at construction; it cannot be unset.
- ``ForecastOutput`` shares no fields with ``StrategyTrade`` — passing its dict
  where a ``StrategyTrade`` is expected raises ``TypeError``
  (guarded by tests/unit/test_forecast_output_contract.py).

Spec: .claude/specs/planes/04-CTR-QLAB-FABRIC-004.md §15.2 +
      .claude/specs/planes/backlog/BL-15-contrato-forecast-output.md
TS mirror: usdcop-trading-dashboard/lib/contracts/forecast-output.contract.ts
Contract: CTR-FORECAST-OUTPUT-001
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime

# ---------------------------------------------------------------------------
# Constants (mirrored in forecast-output.contract.ts — change BOTH sides)
# ---------------------------------------------------------------------------

#: Allowed prediction types. A "direction" belief is NOT a prediction type —
#: it goes in ``direction_probability`` (and a raw score is not a probability;
#: see quant-constitution).
PREDICTION_TYPES = ("return", "log_return", "price")


class ForecastOutputError(ValueError):
    """Raised by ``ForecastOutput.validate()`` when the record breaks contract."""


# ---------------------------------------------------------------------------
# Nested records
# ---------------------------------------------------------------------------

@dataclass
class ForecastPrediction:
    """Point prediction with optional interval bounds."""
    type: str                    # one of PREDICTION_TYPES
    point: float
    lower: float | None = None   # interval lower bound (same unit as point)
    upper: float | None = None   # interval upper bound


@dataclass
class DirectionProbability:
    """Calibrated probability that the target moves up over the horizon."""
    up: float                    # in [0, 1]


# ---------------------------------------------------------------------------
# Core record
# ---------------------------------------------------------------------------

@dataclass
class ForecastOutput:
    """One model x horizon prediction, point-in-time honest.

    ``as_of`` is the information cutoff, ``available_at`` is when the forecast
    physically existed (>= as_of, anti-look-ahead), ``target_time`` is the
    timestamp the prediction refers to (> as_of).
    """

    # --- Identity ---
    forecast_id: str             # unique id, e.g. uuid
    forecast_spec_id: str        # e.g. "usdcop_forecast_zoo_v3"
    asset: str                   # e.g. "usdcop"
    model_id: str                # e.g. "ridge_v2"
    horizon: str                 # e.g. "5d"

    # --- Point-in-time ---
    as_of: str                   # ISO8601 — information cutoff
    available_at: str            # ISO8601 — when the forecast existed (>= as_of)
    target_time: str             # ISO8601 — what the prediction refers to (> as_of)

    # --- Prediction ---
    prediction: ForecastPrediction

    # --- Lineage ---
    model_fingerprint: str       # e.g. "sha256:..."
    data_snapshot_id: str

    # --- Optional belief ---
    direction_probability: DirectionProbability | None = None

    # --- Wall (forced True; not an input in practice) ---
    diagnostic_only: bool = field(default=True)

    def __post_init__(self) -> None:
        # Physical guarantee: a forecast can NEVER claim to be actionable.
        self.diagnostic_only = True
        # Accept plain dicts for the nested records (JSON round-trips).
        if isinstance(self.prediction, dict):
            allowed = {f.name for f in fields(ForecastPrediction)}
            self.prediction = ForecastPrediction(
                **{k: v for k, v in self.prediction.items() if k in allowed}
            )
        if isinstance(self.direction_probability, dict):
            self.direction_probability = DirectionProbability(
                up=self.direction_probability.get("up")
            )

    # -- validation --------------------------------------------------------

    def validate(self) -> "ForecastOutput":
        """Raise ``ForecastOutputError`` unless the record is contract-clean.

        No NaN/Inf anywhere, required ids non-empty, timestamps parseable and
        point-in-time ordered, probability in [0, 1], interval brackets point.
        Returns ``self`` so call sites can chain ``ForecastOutput(...).validate()``.
        """
        errors: list[str] = []

        for name in ("forecast_id", "forecast_spec_id", "asset", "model_id",
                     "horizon", "model_fingerprint", "data_snapshot_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{name} must be a non-empty string")

        ts = {name: _parse_ts(name, getattr(self, name), errors)
              for name in ("as_of", "available_at", "target_time")}
        if ts["as_of"] and ts["available_at"] and ts["available_at"] < ts["as_of"]:
            errors.append("available_at must be >= as_of (anti-look-ahead)")
        if ts["as_of"] and ts["target_time"] and ts["target_time"] <= ts["as_of"]:
            errors.append("target_time must be > as_of")

        p = self.prediction
        if not isinstance(p, ForecastPrediction):
            errors.append("prediction must be a ForecastPrediction")
        else:
            if p.type not in PREDICTION_TYPES:
                errors.append(
                    f"prediction.type {p.type!r} not in {PREDICTION_TYPES}"
                )
            _check_finite("prediction.point", p.point, errors)
            _check_finite("prediction.lower", p.lower, errors, allow_none=True)
            _check_finite("prediction.upper", p.upper, errors, allow_none=True)
            if (_is_finite(p.lower) and _is_finite(p.upper)
                    and _is_finite(p.point)):
                if p.lower > p.upper:
                    errors.append("prediction.lower must be <= prediction.upper")
                elif not (p.lower <= p.point <= p.upper):
                    errors.append(
                        "prediction.point must lie within [lower, upper]"
                    )

        dp = self.direction_probability
        if dp is not None:
            if not isinstance(dp, DirectionProbability):
                errors.append("direction_probability must be a DirectionProbability")
            else:
                _check_finite("direction_probability.up", dp.up, errors)
                if _is_finite(dp.up) and not (0.0 <= dp.up <= 1.0):
                    errors.append("direction_probability.up must be in [0, 1]")

        if self.diagnostic_only is not True:  # unreachable via constructor; belt+braces
            errors.append("diagnostic_only must be True — forecasts are DIAGNOSTIC")

        if errors:
            raise ForecastOutputError("; ".join(errors))
        return self

    # -- (de)serialization -------------------------------------------------

    def to_dict(self) -> dict:
        d = asdict(self)
        if d.get("direction_probability") is None:
            del d["direction_probability"]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ForecastOutput":
        """Create from dict, ignoring unknown keys (nested dicts accepted)."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(value)


def _check_finite(name: str, value, errors: list[str], allow_none: bool = False) -> None:
    if value is None:
        if not allow_none:
            errors.append(f"{name} is required")
        return
    if not _is_finite(value):
        errors.append(f"{name} must be a finite number (no NaN/Inf), got {value!r}")


def _parse_ts(name: str, value, errors: list[str]) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{name} must be a non-empty ISO8601 string")
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        errors.append(f"{name} is not valid ISO8601: {value!r}")
        return None
