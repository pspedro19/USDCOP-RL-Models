"""Regression tests for publication-calendar frequency and cache semantics."""

from datetime import datetime

import pandas as pd
import pytest

from src.data.economic_calendar import EconomicCalendar


def calendar_for(variable: str = "daily_series") -> EconomicCalendar:
    cal = EconomicCalendar.__new__(EconomicCalendar)
    cal.variables = {
        variable: {
            "frequency": "daily",
            "publication": {
                "delay_days": 1,
                "timezone": "UTC",
                "time": "16:00:00",
            },
        }
    }
    cal.config = {"global_rules": {"ffill_limits": {}}}
    cal._pub_date_cache = {}
    return cal


def test_daily_publication_uses_delay_days_not_month_typical_day():
    cal = calendar_for()
    result = cal.get_publication_date("daily_series", "2024-01-10")
    assert result == pd.Timestamp("2024-01-11").date()


def test_publication_cache_keeps_date_and_datetime_requests_separate():
    cal = calendar_for()
    as_date = cal.get_publication_date("daily_series", "2024-01-10", return_datetime=False)
    as_datetime = cal.get_publication_date("daily_series", "2024-01-10", return_datetime=True)
    assert as_date == datetime(2024, 1, 11).date()
    assert as_datetime == pd.Timestamp("2024-01-11 16:00:00", tz="UTC")


def test_unknown_variable_is_not_silently_available():
    cal = calendar_for()
    assert cal.get_publication_date("unknown", "2024-01-10") is None


def test_daily_ffill_respects_publication_timestamp():
    cal = calendar_for()
    index = pd.date_range("2024-01-10 08:00", "2024-01-11 17:00", freq="8h")
    frame = pd.DataFrame({"daily_series": [10.0, None, None, None, 11.0]}, index=index)
    safe = cal.apply_publication_aware_ffill(frame, "daily_series")
    assert safe.loc[pd.Timestamp("2024-01-10 08:00")] != 10.0
    assert safe.loc[pd.Timestamp("2024-01-11 16:00")] == pytest.approx(10.0)


def test_ffill_expires_at_ssot_row_limit():
    cal = calendar_for()
    cal.config = {"global_rules": {"ffill_limits": {
        "daily_bars": {"daily_data": 1}
    }}}
    index = pd.date_range("2024-01-10 00:00", periods=43, freq="h")
    frame = pd.DataFrame({"daily_series": [10.0] + [None] * 42}, index=index)
    safe = cal.apply_publication_aware_ffill(frame, "daily_series")
    assert safe.loc[pd.Timestamp("2024-01-11 16:00")] == pytest.approx(10.0)
    assert safe.loc[pd.Timestamp("2024-01-11 17:00")] == pytest.approx(10.0)
    assert pd.isna(safe.loc[pd.Timestamp("2024-01-11 18:00")])


def test_publication_timezone_is_not_collapsed_to_midnight():
    cal = calendar_for()
    index = pd.date_range("2024-01-10 00:00", periods=42, freq="h")
    frame = pd.DataFrame({"daily_series": [10.0] + [None] * 41}, index=index)
    safe = cal.apply_publication_aware_ffill(frame, "daily_series")
    assert pd.isna(safe.loc[pd.Timestamp("2024-01-11 15:00")])
    assert safe.loc[pd.Timestamp("2024-01-11 16:00")] == pytest.approx(10.0)
