"""Strict multi-frequency join for an observed forward publication ledger.

Input values are native levels, not future-final/revised series. This utility
does not manufacture past publication or first-seen timestamps from daily dates.
The historical T-1 dataset remains an explicitly weaker, separate representation.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

POLICY_FIELDS = {"unit", "max_period_age_calendar_days", "minimum_observations"}
OUTPUT_COLUMNS = [
    "decision_id",
    "series",
    "cutoff_utc",
    "unit",
    "level",
    "log_return_previous",
    "status",
    "source_sha256",
    "previous_source_sha256",
    "period_end",
    "available_at",
    "period_age_calendar_days",
    "previous_period_end",
    "previous_available_at",
    "previous_period_age_calendar_days",
]


def _identifier(value):
    return isinstance(value, str) and bool(value.strip()) and value == value.strip()


def _period_labels(values):
    for value in values:
        if isinstance(value, str):
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                raise ValueError("period_end requires an ISO YYYY-MM-DD date label")
        elif not isinstance(value, date | datetime | pd.Timestamp):
            raise ValueError("period_end requires a date, not a numeric epoch")
    periods = pd.to_datetime(values, errors="raise")
    if (
        periods.dt.tz is not None
        or periods.isna().any()
        or (periods != periods.dt.normalize()).any()
    ):
        raise ValueError("period_end must be a timezone-naive calendar label")
    return periods


def _aware_utc(values, label):
    timestamps = [pd.Timestamp(value) for value in values]
    if any(pd.isna(t) or t.tzinfo is None for t in timestamps):
        raise ValueError(f"{label} requires explicit timezone-aware timestamps")
    return pd.DatetimeIndex([t.tz_convert("UTC") for t in timestamps], tz="UTC")


def join_observed_releases(
    decisions: pd.DataFrame, releases: pd.DataFrame, policies: dict[str, dict], *,
    strict_prior_period: bool = False,
) -> pd.DataFrame:
    """Return one auditable long-form row per decision/series, no silent filling.

    Each policy declares native unit, observation-period freshness in calendar
    days, and minimum distinct observations (2 for a previous log return).
    Freshness applies to EVERY required operand, not only the latest level.
    Period ages use the COP decision date. No frequency/cadence is inferred:
    two previous known periods need not be consecutive daily observations.
    Releases require period_end, publication_at, first_seen_at, value, source_sha256.
    available_at=max(publication_at,first_seen_at) must be STRICTLY before cutoff.
    Choose latest vintage known then, for latest observation period known then.
    Revising an older period must not replace a newer-period level.
    Input validation is ledger-wide: malformed future rows also cause rejection.
    A source hash is provenance supplied by the caller, not source authentication.
    """
    if type(strict_prior_period) is not bool:
        raise ValueError("strict_prior_period must be a boolean")
    required_decisions = {"decision_id", "cutoff_utc"}
    required_releases = {
        "series",
        "period_end",
        "publication_at",
        "first_seen_at",
        "value",
        "unit",
        "source_sha256",
    }
    if (
        not decisions.columns.is_unique
        or not releases.columns.is_unique
        or not required_decisions <= set(decisions)
        or not required_releases <= set(releases)
    ):
        raise ValueError("explicit decision/publication/provenance columns required")
    if (
        decisions["decision_id"].duplicated().any()
        or not decisions["decision_id"].map(_identifier).all()
    ):
        raise ValueError("unique decisions and explicit per-series policies required")
    if not isinstance(policies, dict) or not policies:
        raise ValueError("explicit per-series policy required")
    for series, policy in policies.items():
        if (
            not _identifier(series)
            or not isinstance(policy, dict)
            or set(policy) != POLICY_FIELDS
            or not _identifier(policy["unit"])
            or type(policy["max_period_age_calendar_days"]) is not int
            or policy["max_period_age_calendar_days"] < 0
            or type(policy["minimum_observations"]) is not int
            or policy["minimum_observations"] not in (1, 2)
        ):
            raise ValueError("invalid explicit freshness/transform policy")
    if not releases["series"].map(_identifier).all():
        raise ValueError("explicit source series identity required")
    if set(releases["series"]) - set(policies):
        raise ValueError("undeclared source series")
    d, r = decisions.copy(), releases.copy()
    d["cutoff_utc"] = _aware_utc(d["cutoff_utc"], "cutoff")
    r["publication_at"] = _aware_utc(r["publication_at"], "publication")
    r["first_seen_at"] = _aware_utc(r["first_seen_at"], "first-seen")
    r["period_end"] = _period_labels(r["period_end"])
    r["available_at"] = r[["publication_at", "first_seen_at"]].max(axis=1)
    if r["value"].isna().any():
        raise ValueError("missing source level")
    if r["unit"].isna().any() or not r["unit"].map(_identifier).all():
        raise ValueError("explicit source unit required")
    if (
        r["value"]
        .map(
            lambda value: isinstance(
                value,
                bool | np.bool_ | date | datetime | timedelta | np.datetime64 | np.timedelta64,
            )
        )
        .any()
    ):
        raise ValueError("source level cannot be a boolean, datetime or duration")
    r["value"] = pd.to_numeric(r["value"], errors="raise")
    if pd.api.types.is_complex_dtype(r["value"]) or not np.isfinite(r["value"]).all():
        raise ValueError("source level must be finite and real")
    if not all(isinstance(h, str) and re.fullmatch("[a-f0-9]{64}", h) for h in r["source_sha256"]):
        raise ValueError("missing raw source hash")
    if r.duplicated(["series", "period_end", "publication_at"]).any():
        raise ValueError("ambiguous duplicate release/vintage")
    if (r["period_end"].dt.date > r["publication_at"].dt.date).any():
        raise ValueError("observation period ends after publication")
    rows = []
    for series, policy in policies.items():
        source = r[r["series"] == series]
        if not (source["unit"] == policy["unit"]).all():
            raise ValueError("source unit differs from declared variable")
        for decision in d.itertuples(index=False):
            cutoff = decision.cutoff_utc
            available = source[source["available_at"] < cutoff]
            if strict_prior_period:
                available = available[available["period_end"].dt.date < cutoff.tz_convert("America/Bogota").date()]
            # Important: late revision of an old period is not the latest level.
            known = (
                available.sort_values(["period_end", "publication_at", "first_seen_at"])
                .drop_duplicates("period_end", keep="last")
                .sort_values("period_end")
            )
            record = dict.fromkeys(OUTPUT_COLUMNS)
            record.update(
                {
                    "decision_id": decision.decision_id,
                    "series": series,
                    "cutoff_utc": cutoff,
                    "unit": policy["unit"],
                    "level": np.nan,
                    "log_return_previous": np.nan,
                    "status": "MISSING_BEFORE_CUTOFF",
                    "source_sha256": None,
                    "previous_source_sha256": None,
                    "period_end": None,
                    "available_at": None,
                }
            )
            if len(known) >= policy["minimum_observations"]:
                last = known.iloc[-1]
                cutoff_date = cutoff.tz_convert("America/Bogota").date()
                age = (cutoff_date - last["period_end"].date()).days
                record.update(
                    period_end=last["period_end"],
                    available_at=last["available_at"],
                    source_sha256=last["source_sha256"],
                    period_age_calendar_days=age,
                )
                previous = None
                if policy["minimum_observations"] == 2:
                    previous = known.iloc[-2]
                    previous_age = (cutoff_date - previous["period_end"].date()).days
                    record.update(
                        previous_source_sha256=previous["source_sha256"],
                        previous_period_end=previous["period_end"],
                        previous_available_at=previous["available_at"],
                        previous_period_age_calendar_days=previous_age,
                    )
                if age < 0:
                    record["status"] = "FUTURE_OBSERVATION_PERIOD"
                elif age > policy["max_period_age_calendar_days"]:
                    record["status"] = "STALE_OBSERVATION_PERIOD"
                elif previous is not None and previous_age > policy["max_period_age_calendar_days"]:
                    record["status"] = "STALE_PREVIOUS_OBSERVATION_PERIOD"
                else:
                    record.update(level=float(last["value"]), status="AVAILABLE")
                    if previous is not None:
                        if last["value"] <= 0 or previous["value"] <= 0:
                            raise ValueError("log return requires two positive levels")
                        log_return = float(np.log(last["value"]) - np.log(previous["value"]))
                        if not np.isfinite(log_return):
                            raise ValueError("log return must remain finite")
                        record["log_return_previous"] = log_return
            rows.append(record)
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
