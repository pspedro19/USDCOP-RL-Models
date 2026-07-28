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

INGEST WALL (BL-15 remedio)
---------------------------
``ingest_forecast_output()`` / ``ForecastOutput.from_dict()`` are the ONLY way a
record enters the system, and they are fail-closed:

- unknown keys are REJECTED (never silently dropped — dropping them laundered
  actionable payloads into innocent-looking forecasts);
- ``diagnostic_only`` must arrive as the literal ``True`` (a payload claiming to
  be actionable is rejected instead of being coerced);
- every record is validated; there is no "construct now, validate later" path.

TIMESTAMPS (strict, bilateral)
------------------------------
Grammar accepted by BOTH runtimes (Python here, TypeScript mirror):

    YYYY-MM-DDTHH:MM:SS[.ffffff][Z|±HH:MM]

with a REAL calendar check (leap years included), ``HH<=23``, ``MM<=59``,
``SS<=59`` and offsets in ``±00:00..±23:59``. Everything else is rejected on both
sides: date-only, basic/compact form, week dates, space separator, lowercase
``z``, ``+HHMM`` without colon, surrounding whitespace. This is deliberately
narrower than ``datetime.fromisoformat`` (which accepts compact/week forms) and
than ``Date.parse`` (which silently ROLLS impossible dates: ``2026-02-30`` became
``2026-03-02``). Both engines are bypassed by the shared parser below.

The three timestamps must share tz-awareness: all aware or all naive. Mixed
naive/aware is a CONTRACT rejection (``ForecastOutputError``), never a raw
``TypeError`` from comparing datetimes. Rationale (`.claude/rules/data-governance.md`):
USD/COP timestamps live in ``America/Bogota`` and assets not bounded by the COP
session (XAU/USD, BTC/USDT) are instant-based TIMESTAMPTZ — a record that mixes
both conventions has no defined ordering, so it cannot be validated, only rejected.

Spec: .claude/specs/planes/04-CTR-QLAB-FABRIC-004.md §15.2 +
      .claude/specs/planes/backlog/BL-15-contrato-forecast-output.md
TS mirror: usdcop-trading-dashboard/lib/contracts/forecast-output.contract.ts
Shared case table: tests/fixtures/forecast_output_cases.v1.json (both runners)
Contract: CTR-FORECAST-OUTPUT-001
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping

# ---------------------------------------------------------------------------
# Constants (mirrored in forecast-output.contract.ts — change BOTH sides)
# ---------------------------------------------------------------------------

CONTRACT_ID = "CTR-FORECAST-OUTPUT-001"

#: Allowed prediction types. A "direction" belief is NOT a prediction type —
#: it goes in ``direction_probability`` (and a raw score is not a probability;
#: see quant-constitution).
PREDICTION_TYPES = ("return", "log_return", "price")

#: Closed field sets — anything else is an ingest-wall rejection.
FORECAST_OUTPUT_FIELDS = (
    "forecast_id", "forecast_spec_id", "asset", "model_id", "horizon",
    "as_of", "available_at", "target_time",
    "prediction", "model_fingerprint", "data_snapshot_id",
    "direction_probability", "diagnostic_only",
)
PREDICTION_FIELDS = ("type", "point", "lower", "upper")
DIRECTION_PROBABILITY_FIELDS = ("up",)

#: Required non-empty string identifiers.
REQUIRED_STRING_FIELDS = (
    "forecast_id", "forecast_spec_id", "asset", "model_id", "horizon",
    "model_fingerprint", "data_snapshot_id",
)

TIMESTAMP_FIELDS = ("as_of", "available_at", "target_time")

#: Strict ISO8601 grammar (see module docstring). Mirrored character by
#: character in the TS contract.
_ISO8601_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(\.\d{1,6})?"
    r"(Z|[+-]\d{2}:\d{2})?$"
)


class ForecastOutputError(ValueError):
    """Raised when a record/payload breaks CTR-FORECAST-OUTPUT-001."""


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
        # Accept plain dicts for the nested records (JSON round-trips). Unknown
        # nested keys are NOT dropped here — validate() rejects them.
        if isinstance(self.prediction, Mapping):
            allowed = {f.name for f in fields(ForecastPrediction)}
            unknown = {k: v for k, v in self.prediction.items() if k not in allowed}
            if unknown:
                raise ForecastOutputError(
                    f"prediction has unknown field(s) not in {CONTRACT_ID}: "
                    f"{sorted(unknown)}"
                )
            self.prediction = ForecastPrediction(**dict(self.prediction))
        if isinstance(self.direction_probability, Mapping):
            allowed = set(DIRECTION_PROBABILITY_FIELDS)
            unknown = sorted(k for k in self.direction_probability if k not in allowed)
            if unknown:
                raise ForecastOutputError(
                    f"direction_probability has unknown field(s) not in "
                    f"{CONTRACT_ID}: {unknown}"
                )
            self.direction_probability = DirectionProbability(
                **dict(self.direction_probability)
            )

    # -- validation --------------------------------------------------------

    def validate(self) -> "ForecastOutput":
        """Raise ``ForecastOutputError`` unless the record is contract-clean.

        Delegates to :func:`validate_forecast_payload` on this record's dict, so
        the object path and the raw-payload/ingest path can never diverge.
        Returns ``self`` so call sites can chain ``ForecastOutput(...).validate()``.
        """
        errors = validate_forecast_payload(self.to_dict())
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
    def from_dict(cls, d: Mapping[str, Any]) -> "ForecastOutput":
        """Ingest wall: strict. Unknown keys, missing keys and any contract
        violation raise ``ForecastOutputError`` (nothing is silently dropped or
        coerced). This is the ONLY supported entry point from JSON/CSV."""
        return ingest_forecast_output(d)


# ---------------------------------------------------------------------------
# Ingest wall (raw payload -> validated record)
# ---------------------------------------------------------------------------

def validate_forecast_payload(raw: Any) -> list[str]:
    """Structural + semantic validation of an UNTRUSTED payload.

    Returns the list of contract violations (empty list = valid). Exact mirror
    of ``validateForecastOutput()`` in forecast-output.contract.ts — both sides
    are exercised against tests/fixtures/forecast_output_cases.v1.json and must
    return the same verdict for every case.
    """
    if not isinstance(raw, Mapping):
        return ["forecast_output must be an object"]

    errors: list[str] = []

    unknown = sorted(k for k in raw if k not in FORECAST_OUTPUT_FIELDS)
    if unknown:
        errors.append(f"unknown field(s) not in {CONTRACT_ID}: {unknown}")

    for name in REQUIRED_STRING_FIELDS:
        value = raw.get(name)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{name} must be a non-empty string")

    # --- timestamps: strict grammar, then awareness homogeneity, then order --
    parsed: dict[str, datetime | None] = {}
    aware: dict[str, bool] = {}
    for name in TIMESTAMP_FIELDS:
        dt, is_aware, error = _parse_contract_timestamp(name, raw.get(name))
        parsed[name] = dt
        if dt is not None:
            aware[name] = bool(is_aware)
        if error:
            errors.append(error)

    if len(aware) == len(TIMESTAMP_FIELDS) and len(set(aware.values())) > 1:
        detail = ", ".join(
            f"{n}={'aware' if aware[n] else 'naive'}" for n in TIMESTAMP_FIELDS
        )
        errors.append(
            "timestamps must be all timezone-aware or all naive — mixed "
            f"naive/aware has no defined ordering ({detail})"
        )
    elif len(aware) == len(TIMESTAMP_FIELDS):
        as_of, available_at, target_time = (parsed[n] for n in TIMESTAMP_FIELDS)
        if available_at < as_of:                                # type: ignore[operator]
            errors.append("available_at must be >= as_of (anti-look-ahead)")
        if target_time <= as_of:                                # type: ignore[operator]
            errors.append("target_time must be > as_of")

    # --- prediction ---------------------------------------------------------
    p = raw.get("prediction")
    if not isinstance(p, Mapping):
        errors.append("prediction must be an object")
    else:
        unknown_p = sorted(k for k in p if k not in PREDICTION_FIELDS)
        if unknown_p:
            errors.append(
                f"prediction has unknown field(s) not in {CONTRACT_ID}: {unknown_p}"
            )
        if p.get("type") not in PREDICTION_TYPES:
            errors.append(
                f"prediction.type {p.get('type')!r} not in {PREDICTION_TYPES}"
            )
        point, lower, upper = p.get("point"), p.get("lower"), p.get("upper")
        _check_finite("prediction.point", point, errors)
        _check_finite("prediction.lower", lower, errors, allow_none=True)
        _check_finite("prediction.upper", upper, errors, allow_none=True)
        if _is_finite(lower) and _is_finite(upper) and _is_finite(point):
            if lower > upper:
                errors.append("prediction.lower must be <= prediction.upper")
            elif not (lower <= point <= upper):
                errors.append("prediction.point must lie within [lower, upper]")

    # --- direction_probability (optional) -----------------------------------
    dp = raw.get("direction_probability")
    if dp is not None:
        if not isinstance(dp, Mapping):
            errors.append("direction_probability must be an object")
        else:
            unknown_dp = sorted(k for k in dp if k not in DIRECTION_PROBABILITY_FIELDS)
            if unknown_dp:
                errors.append(
                    f"direction_probability has unknown field(s) not in "
                    f"{CONTRACT_ID}: {unknown_dp}"
                )
            up = dp.get("up")
            if not _is_finite(up) or not (0.0 <= up <= 1.0):
                errors.append(
                    "direction_probability.up must be a finite number in [0, 1]"
                )

    # --- the wall -----------------------------------------------------------
    if raw.get("diagnostic_only") is not True:
        errors.append("diagnostic_only must be true — forecasts are DIAGNOSTIC")

    return errors


def is_forecast_payload(raw: Any) -> bool:
    """True when ``raw`` satisfies the contract (mirror of ``isForecastOutput``)."""
    return not validate_forecast_payload(raw)


def ingest_forecast_output(raw: Any) -> ForecastOutput:
    """THE ingest wall: an untrusted payload becomes a record, or nothing.

    Raises ``ForecastOutputError`` (never ``TypeError``) listing every violation.
    """
    errors = validate_forecast_payload(raw)
    if errors:
        raise ForecastOutputError("; ".join(errors))
    data = {k: v for k, v in dict(raw).items() if k in FORECAST_OUTPUT_FIELDS}
    data.pop("diagnostic_only", None)   # forced True by the constructor
    return ForecastOutput(**data)


def ingest_forecast_outputs(rows: Iterable[Any]) -> list[ForecastOutput]:
    """All-or-nothing ingest of a batch (one bad row = nothing enters)."""
    out: list[ForecastOutput] = []
    problems: list[str] = []
    for i, row in enumerate(rows):
        errors = validate_forecast_payload(row)
        if errors:
            problems.append(f"[{i}] {'; '.join(errors)}")
        else:
            out.append(ingest_forecast_output(row))
    if problems:
        raise ForecastOutputError(
            f"{len(problems)} row(s) rejected by {CONTRACT_ID}: " + " | ".join(problems)
        )
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(value)


def _check_finite(name: str, value: Any, errors: list[str], allow_none: bool = False) -> None:
    if value is None:
        if not allow_none:
            errors.append(f"{name} is required")
        return
    if not _is_finite(value):
        errors.append(f"{name} must be a finite number (no NaN/Inf), got {value!r}")


def _days_in_month(year: int, month: int) -> int:
    if month == 2:
        leap = (year % 4 == 0 and year % 100 != 0) or year % 400 == 0
        return 29 if leap else 28
    return 30 if month in (4, 6, 9, 11) else 31


def _parse_contract_timestamp(
    name: str, value: Any
) -> tuple[datetime | None, bool | None, str | None]:
    """Strict ISO8601 parse. Returns ``(datetime, is_aware, error)``.

    Neither ``datetime.fromisoformat`` nor ``Date.parse`` is used: the first
    accepts compact/week forms, the second silently rolls impossible calendar
    dates (``2026-02-30`` -> ``2026-03-02``). The grammar below is the SAME on
    both sides of the mirror.
    """
    if not isinstance(value, str) or not value:
        return None, None, f"{name} must be a non-empty ISO8601 string"

    m = _ISO8601_RE.fullmatch(value)
    if not m:
        return None, None, (
            f"{name} is not strict ISO8601 "
            f"(YYYY-MM-DDTHH:MM:SS[.ffffff][Z|±HH:MM]): {value!r}"
        )

    year, month, day, hour, minute, second = (int(m.group(i)) for i in range(1, 7))
    frac, offset_raw = m.group(7), m.group(8)

    if not (1 <= month <= 12):
        return None, None, f"{name} has an impossible month: {value!r}"
    if not (1 <= day <= _days_in_month(year, month)):
        return None, None, f"{name} has an impossible calendar date: {value!r}"
    if hour > 23 or minute > 59 or second > 59:
        return None, None, f"{name} has an impossible time: {value!r}"

    microsecond = int(round(float(frac) * 1_000_000)) if frac else 0

    tzinfo = None
    if offset_raw is not None:
        if offset_raw == "Z":
            tzinfo = timezone.utc
        else:
            sign = 1 if offset_raw[0] == "+" else -1
            oh, om = int(offset_raw[1:3]), int(offset_raw[4:6])
            if oh > 23 or om > 59:
                return None, None, f"{name} has an impossible UTC offset: {value!r}"
            tzinfo = timezone(sign * timedelta(hours=oh, minutes=om))

    return (
        datetime(year, month, day, hour, minute, second, microsecond, tzinfo=tzinfo),
        tzinfo is not None,
        None,
    )
