"""Asset-parameterized OHLCV seed validators (CTR-DQ-OHLCV-001).

WHY THIS EXISTS
---------------
The Gold daily seed shipped with ~20% of its bars mis-dated onto **Sunday** (a day-shift bug in
``scripts/ingest_asset_ohlcv.py::_daily_to_nyclose`` — it converted a 00:00-UTC stamp to ET, landing
on the *previous* ET day, then ``normalize()`` snapped the calendar date back one day). Nothing in
the pipeline validated **weekday coverage / bars-per-period / calendar gaps / timezone-of-close** on
an OHLCV seed, so the corruption silently biased every downstream Gold metric (ann.vol, Sharpe,
vol-target sizing). The macro-focused validators in ``airflow/dags/validators/data_validators.py``
are keyed on the ``fecha`` column and never look at OHLCV weekday structure.

This module is the missing gate. It is **asset-generic**: every threshold is derived from the
``AssetProfile.session`` SSOT (``config/assets/<id>.yaml``) — session days, timezone, mode,
bars_per_day/year, daily_close_tz — so a new asset/granularity is covered with zero new code. It is a
leaf module (pandas + AssetProfile only) so it imports cheaply inside ingestion scripts and DAGs.

It consolidates the OHLC-integrity / dup / NaN / bars-per-day / gap logic that was previously
duplicated across ``ingest_asset_ohlcv.py::_audit`` and ``build_unified_fx_seed.py::validate_pair``.

USAGE
-----
    from src.data_quality.ohlcv_validators import validate_ohlcv_seed
    report = validate_ohlcv_seed(df, profile, granularity="daily")
    report.raise_if_failed()          # hard gate — raises on any ERROR
    log.info(report.summary_line())   # or inspect report.issues / report.to_dict()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

# Severity levels
ERROR = "ERROR"
WARN = "WARN"


@dataclass(frozen=True)
class ValidationIssue:
    check: str
    severity: str  # ERROR | WARN
    message: str
    count: int = 0

    def __str__(self) -> str:
        n = f" ({self.count})" if self.count else ""
        return f"[{self.severity}] {self.check}: {self.message}{n}"


@dataclass
class OHLCVValidationReport:
    asset_id: str
    symbol: str
    granularity: str
    rows: int
    issues: list[ValidationIssue] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    # ---- outcome ----------------------------------------------------------
    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == WARN]

    @property
    def ok(self) -> bool:
        """True when there are no ERROR-level issues (warnings are allowed)."""
        return not self.errors

    def raise_if_failed(self) -> "OHLCVValidationReport":
        if not self.ok:
            bullets = "\n  - ".join(str(i) for i in self.errors)
            raise OHLCVValidationError(
                f"OHLCV validation FAILED for {self.symbol} [{self.granularity}] "
                f"({len(self.errors)} error(s)):\n  - {bullets}"
            )
        return self

    def summary_line(self) -> str:
        e, w = len(self.errors), len(self.warnings)
        verdict = "PASS" if self.ok else "FAIL"
        return (f"OHLCV[{self.granularity}] {self.symbol}: {verdict} "
                f"rows={self.rows} errors={e} warnings={w}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "symbol": self.symbol,
            "granularity": self.granularity,
            "rows": self.rows,
            "ok": self.ok,
            "errors": [str(i) for i in self.errors],
            "warnings": [str(i) for i in self.warnings],
            "stats": self.stats,
        }


class OHLCVValidationError(ValueError):
    """Raised by ``OHLCVValidationReport.raise_if_failed`` on any ERROR-level issue."""


# --------------------------------------------------------------------------- checks
def _session_local(df: pd.DataFrame, tz: str) -> pd.Series:
    t = pd.to_datetime(df["time"], utc=True)
    return t.dt.tz_convert(ZoneInfo(tz))


def validate_ohlcv_seed(
    df: pd.DataFrame,
    profile,
    granularity: str = "daily",
    *,
    price_range: tuple[float, float] | None = None,
    bars_per_period_tol: float = 0.20,
    gap_business_days: int = 6,
    weekday_is_error: bool = True,
) -> OHLCVValidationReport:
    """Validate an OHLCV frame against its ``AssetProfile``.

    Parameters
    ----------
    df : DataFrame with a tz-aware ``time`` column + open/high/low/close(/volume).
    profile : AssetProfile (drives every threshold from ``profile.session``).
    granularity : "daily" | "5min" (selects bars-per-period + tz-of-close checks).
    bars_per_period_tol : fractional tolerance for bars/day (intraday) or bars/year (daily).
    gap_business_days : a WARN is raised when consecutive session days are missing beyond this many.
    weekday_is_error : off-session weekday bars are an ERROR (the Gold-bug gate). Set False to soften.
    """
    sess = profile.session
    rep = OHLCVValidationReport(
        asset_id=getattr(profile, "asset_id", "?"),
        symbol=getattr(profile, "symbol", "?"),
        granularity=granularity,
        rows=len(df),
    )
    if df is None or df.empty:
        rep.issues.append(ValidationIssue("non_empty", ERROR, "frame is empty"))
        return rep

    local = _session_local(df, sess.timezone)
    dow = local.dt.dayofweek

    # 1) WEEKDAY COVERAGE — the gate that would have caught the Gold Sunday pile-up.
    allowed = set(sess.days)
    off = df[~dow.isin(list(allowed))]
    dow_hist = {int(k): int(v) for k, v in dow.value_counts().sort_index().items()}
    rep.stats["dow_histogram"] = dow_hist
    if len(off):
        sev = ERROR if weekday_is_error else WARN
        bad_days = sorted({int(d) for d in dow[~dow.isin(list(allowed))].unique()})
        rep.issues.append(ValidationIssue(
            "weekday_coverage", sev,
            f"{len(off)} bars fall outside session days {sorted(allowed)} "
            f"(offending dow={bad_days}, 0=Mon..6=Sun)", count=len(off)))

    # 2) BARS PER PERIOD — density sanity vs the AssetProfile expectation.
    if granularity == "daily":
        per_year = df.groupby(local.dt.year).size()
        # ignore partial first/last years for the tolerance check
        full = per_year.iloc[1:-1] if len(per_year) > 2 else per_year
        expected = sess.trading_days_per_year
        rep.stats["bars_per_year_median"] = int(per_year.median())
        if len(full):
            worst = full[(full < expected * (1 - bars_per_period_tol)) |
                         (full > expected * (1 + bars_per_period_tol))]
            for yr, n in worst.items():
                rep.issues.append(ValidationIssue(
                    "bars_per_year", WARN,
                    f"{yr}: {int(n)} bars vs expected ~{expected} "
                    f"(±{int(bars_per_period_tol*100)}%)"))
    else:  # intraday
        per_day = df.groupby(local.dt.date).size()
        med = int(per_day.median())
        rep.stats["bars_per_day_median"] = med
        exp = sess.bars_per_day
        if exp and (med < exp * (1 - bars_per_period_tol) or med > exp * (1 + bars_per_period_tol)):
            rep.issues.append(ValidationIssue(
                "bars_per_day", WARN,
                f"median {med} bars/day vs expected ~{exp} (±{int(bars_per_period_tol*100)}%)"))

    # 3) CALENDAR GAPS — long stretches of missing session days.
    day_index = pd.Index(sorted(set(local.dt.normalize())))
    if len(day_index) > 2:
        deltas = day_index.to_series().diff().dropna().dt.days
        big = deltas[deltas > gap_business_days]
        rep.stats["max_gap_days"] = int(deltas.max()) if len(deltas) else 0
        if len(big):
            rep.issues.append(ValidationIssue(
                "calendar_gap", WARN,
                f"{len(big)} gap(s) > {gap_business_days} days (max {int(deltas.max())}d)",
                count=len(big)))

    # 4) TIMEZONE-OF-CLOSE SANITY.
    if granularity == "daily":
        close_tz = (profile.raw.get("session", {}).get("daily_close_tz")
                    if getattr(profile, "raw", None) else None)
        if close_tz:
            close_local = _session_local(df, close_tz)
            hours = close_local.dt.hour.value_counts()
            rep.stats["daily_close_hours"] = {int(k): int(v) for k, v in hours.items()}
            if len(hours) > 1:
                rep.issues.append(ValidationIssue(
                    "tz_close_consistency", WARN,
                    f"daily bars close at multiple {close_tz} hours {sorted(rep.stats['daily_close_hours'])} "
                    f"(expected a single close hour)"))
    elif sess.mode == "exchange_hours" and sess.open and sess.close:
        oh, om = map(int, sess.open.split(":"))
        ch, cm = map(int, sess.close.split(":"))
        mins = local.dt.hour * 60 + local.dt.minute
        oob = int(((mins < oh * 60 + om) | (mins > ch * 60 + cm)).sum())
        if oob:
            rep.issues.append(ValidationIssue(
                "session_window", ERROR,
                f"{oob} intraday bars outside [{sess.open},{sess.close}] {sess.timezone}", count=oob))

    # 5) OHLC INTEGRITY / DUP / NAN (consolidated from _audit + validate_pair).
    dup = int(df["time"].duplicated().sum())
    if dup:
        rep.issues.append(ValidationIssue("duplicate_time", ERROR, f"{dup} duplicate timestamps", count=dup))
    ohlc = ["open", "high", "low", "close"]
    nan = int(df[ohlc].isna().any(axis=1).sum())
    if nan:
        rep.issues.append(ValidationIssue("nan_ohlc", ERROR, f"{nan} rows with NaN OHLC", count=nan))
    bad_hl = int((df["high"] < df["low"]).sum())
    if bad_hl:
        rep.issues.append(ValidationIssue("high_lt_low", ERROR, f"{bad_hl} rows high<low", count=bad_hl))
    bad_int = int(((df["high"] < df[["open", "close"]].max(axis=1)) |
                   (df["low"] > df[["open", "close"]].min(axis=1))).sum())
    if bad_int:
        rep.issues.append(ValidationIssue(
            "ohlc_integrity", ERROR, f"{bad_int} OHLC-integrity violations", count=bad_int))

    # 6) PRICE RANGE (soft — provider glitches, not structural).
    pr = price_range or getattr(profile, "price_range", None)
    if pr:
        lo, hi = pr
        oor = int((~df["close"].between(lo, hi)).sum())
        if oor > len(df) * 0.02:
            rep.issues.append(ValidationIssue(
                "price_range", WARN,
                f"{oor} closes outside [{lo},{hi}] (>2%)", count=oor))

    return rep
