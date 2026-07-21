"""The OHLCV freshness gate counts trading days, not calendar days.

Contract: CTR-DQ-001

Found live on 2026-07-21: Monday July 20 is Colombia's Independence Day. The market was
closed, zero bars were missing, the gap detector agreed — and H5-L3 was blocked all Tuesday
morning because Friday→Tuesday is 4 CALENDAR days against a 3-day threshold. A staleness gate
that counts days the market could not have produced data measures the calendar, not the data.

The fix keeps the threshold at 3 and changes the unit for session-bound series: Fri→Sun is now
0 trading days (stricter than the old 2 calendar days on normal weeks), and Fri→Tuesday-after-
a-Monday-holiday is 1-2 trading days instead of 4. The gate still blocks a genuinely dead
provider — five missing TRADING days is five missing trading days in any calendar.

Loaded by file path: importing `utils.data_quality` as a package pulls utils/__init__, which
demands POSTGRES_PASSWORD. The gate logic itself has no such dependency and must stay testable
without a live environment.
"""
from __future__ import annotations

import importlib.util
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location(
        "data_quality", ROOT / "airflow" / "dags" / "utils" / "data_quality.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Cur:
    def __init__(self, latest):
        self._latest = latest

    def execute(self, *_):
        pass

    def fetchone(self):
        return (self._latest,)

    def close(self):
        pass


class _Conn:
    def __init__(self, latest):
        self._latest = latest

    def cursor(self):
        return _Cur(self._latest)


def test_weekend_counts_zero_trading_days():
    dq = _load()
    # Friday -> the following Sunday: no trading day has elapsed.
    assert dq._trading_days_since(date(2026, 7, 17), date(2026, 7, 19)) == 0


def test_friday_to_tuesday_after_monday_holiday_passes():
    """The exact live case: Fri 07-17 -> Tue 07-21 with Mon 07-20 a Colombian holiday.

    Calendar age = 4 (would block). Trading age = 1-2 depending on whether the
    colombian_holidays package resolves Monday (weekend-only fallback counts Mon+Tue = 2).
    Either way it is <= 3, so the gate passes — as it should, with zero missing bars.
    """
    dq = _load()
    age = dq._trading_days_since(date(2026, 7, 17), date(2026, 7, 21))
    assert age <= 2, f"expected <=2 trading days, got {age}"

    latest = datetime(2026, 7, 17, 12, 55, tzinfo=timezone.utc)
    if (datetime.now(timezone.utc) - latest).days <= 4:
        # Only meaningful while "now" is near the incident date; the synthetic check above
        # is the timeless one.
        dq.check_table_freshness(_Conn(latest), "t", "time", 3, "OHLCV", count="trading")


def test_dead_provider_still_blocks():
    """Five missing TRADING days must still raise — the gate lost no protection."""
    dq = _load()
    today = datetime.now(timezone.utc)
    # Walk back until `latest` is >3 trading days behind today under ANY calendar
    # (9 calendar days always contains >=5 weekdays).
    latest = today - timedelta(days=9)
    with pytest.raises(ValueError, match="trading days"):
        dq.check_table_freshness(_Conn(latest), "t", "time", 3, "OHLCV", count="trading")


def test_calendar_mode_is_unchanged_default():
    """Other callers keep calendar semantics: no silent behavior change outside OHLCV."""
    dq = _load()
    latest = datetime.now(timezone.utc) - timedelta(days=4)
    with pytest.raises(ValueError):
        dq.check_table_freshness(_Conn(latest), "t", "time", 3, "X")  # default count
