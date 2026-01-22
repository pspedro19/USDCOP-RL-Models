"""
Unit Tests for DateParser
=========================

Tests for the unified date parser used in L0 macro data extraction.

Contract: CTR-L0-DATEPARSE-001-TEST
"""

import pytest
from datetime import datetime, date
import pandas as pd

import sys
sys.path.insert(0, '/opt/airflow')

from airflow.dags.utils.date_parser import DateParser, DateValidator


class TestDateParser:
    """Test cases for DateParser class."""

    # =========================================================================
    # ISO Format Tests
    # =========================================================================

    def test_parse_iso_format(self):
        """Test parsing ISO format dates (YYYY-MM-DD)."""
        assert DateParser.parse("2024-01-15") == "2024-01-15"
        assert DateParser.parse("2025-12-31") == "2025-12-31"
        assert DateParser.parse("2020-02-29") == "2020-02-29"  # Leap year

    def test_parse_slash_format(self):
        """Test parsing slash format dates (YYYY/MM/DD)."""
        assert DateParser.parse("2024/01/15") == "2024-01-15"
        assert DateParser.parse("2025/12/31") == "2025-12-31"

    def test_parse_european_format(self):
        """Test parsing European format dates (DD/MM/YYYY)."""
        assert DateParser.parse("15/01/2024") == "2024-01-15"
        assert DateParser.parse("31/12/2025") == "2025-12-31"

    def test_parse_us_format(self):
        """Test parsing US format dates (MM/DD/YYYY)."""
        assert DateParser.parse("01/15/2024") == "2024-01-15"
        assert DateParser.parse("12/31/2025") == "2025-12-31"

    def test_parse_compact_format(self):
        """Test parsing compact format dates (YYYYMMDD)."""
        assert DateParser.parse("20240115") == "2024-01-15"
        assert DateParser.parse("20251231") == "2025-12-31"

    # =========================================================================
    # EMBI Format Tests
    # =========================================================================

    def test_parse_embi_format(self):
        """Test parsing EMBI date format (DDMmmYY)."""
        assert DateParser.parse("06Ene26") == "2026-01-06"
        assert DateParser.parse("17Nov25") == "2025-11-17"
        assert DateParser.parse("31Dic24") == "2024-12-31"
        assert DateParser.parse("01Feb20") == "2020-02-01"

    def test_parse_embi_format_case_insensitive(self):
        """Test EMBI format is case insensitive."""
        assert DateParser.parse("06ENE26") == "2026-01-06"
        assert DateParser.parse("17NOV25") == "2025-11-17"
        assert DateParser.parse("06ene26") == "2026-01-06"

    def test_parse_embi_september_variants(self):
        """Test EMBI format handles both Sep and Set for September."""
        assert DateParser.parse("15Sep24") == "2024-09-15"
        assert DateParser.parse("15Set24") == "2024-09-15"

    def test_parse_embi_year_boundary(self):
        """Test EMBI 2-digit year boundary (<=49 = 2000s, >49 = 1900s)."""
        assert DateParser.parse("01Ene49") == "2049-01-01"
        assert DateParser.parse("01Ene50") == "1950-01-01"
        assert DateParser.parse("01Ene00") == "2000-01-01"
        assert DateParser.parse("01Ene99") == "1999-01-01"

    # =========================================================================
    # Object Type Tests
    # =========================================================================

    def test_parse_datetime_object(self):
        """Test parsing datetime objects."""
        dt = datetime(2024, 1, 15, 12, 30, 45)
        assert DateParser.parse(dt) == "2024-01-15"

    def test_parse_date_object(self):
        """Test parsing date objects."""
        d = date(2024, 1, 15)
        assert DateParser.parse(d) == "2024-01-15"

    def test_parse_pandas_timestamp(self):
        """Test parsing pandas Timestamp objects."""
        ts = pd.Timestamp("2024-01-15 12:30:45")
        assert DateParser.parse(ts) == "2024-01-15"

    # =========================================================================
    # Edge Cases and Error Handling
    # =========================================================================

    def test_parse_none(self):
        """Test parsing None returns None."""
        assert DateParser.parse(None) is None

    def test_parse_empty_string(self):
        """Test parsing empty string returns None."""
        assert DateParser.parse("") is None
        assert DateParser.parse("   ") is None

    def test_parse_invalid_date(self):
        """Test parsing invalid date returns None."""
        assert DateParser.parse("not-a-date") is None
        assert DateParser.parse("abc123") is None
        assert DateParser.parse("2024-13-45") is None  # Invalid month/day

    def test_parse_strips_whitespace(self):
        """Test parsing strips leading/trailing whitespace."""
        assert DateParser.parse("  2024-01-15  ") == "2024-01-15"
        assert DateParser.parse("\t2024-01-15\n") == "2024-01-15"

    # =========================================================================
    # Conversion Methods Tests
    # =========================================================================

    def test_parse_to_date(self):
        """Test parse_to_date returns date object."""
        result = DateParser.parse_to_date("2024-01-15")
        assert isinstance(result, date)
        assert result == date(2024, 1, 15)

    def test_parse_to_datetime(self):
        """Test parse_to_datetime returns datetime object."""
        result = DateParser.parse_to_datetime("2024-01-15")
        assert isinstance(result, datetime)
        assert result == datetime(2024, 1, 15)

    def test_is_valid_date(self):
        """Test is_valid_date returns correct boolean."""
        assert DateParser.is_valid_date("2024-01-15") is True
        assert DateParser.is_valid_date("invalid") is False
        assert DateParser.is_valid_date(None) is False

    # =========================================================================
    # BanRep Format Tests
    # =========================================================================

    def test_parse_banrep_date(self):
        """Test parsing BanRep SUAMECA date format (YYYY/MM/DD)."""
        assert DateParser.parse_banrep_date("2024/01/15") == "2024-01-15"
        assert DateParser.parse_banrep_date("2025/12/31") == "2025-12-31"

    def test_parse_banrep_date_empty(self):
        """Test BanRep date parsing handles empty input."""
        assert DateParser.parse_banrep_date("") is None
        assert DateParser.parse_banrep_date(None) is None

    # =========================================================================
    # API Format Tests
    # =========================================================================

    def test_format_for_api_iso(self):
        """Test formatting for ISO API format."""
        assert DateParser.format_for_api("2024-01-15", "iso") == "2024-01-15"
        assert DateParser.format_for_api(date(2024, 1, 15), "iso") == "2024-01-15"

    def test_format_for_api_compact(self):
        """Test formatting for compact API format."""
        assert DateParser.format_for_api("2024-01-15", "compact") == "20240115"

    def test_format_for_api_invalid(self):
        """Test formatting invalid date returns None."""
        assert DateParser.format_for_api("invalid", "iso") is None


class TestDateValidator:
    """Test cases for DateValidator class."""

    def test_is_weekend_saturday(self):
        """Test Saturday is detected as weekend."""
        assert DateValidator.is_weekend(date(2024, 1, 13)) is True  # Saturday

    def test_is_weekend_sunday(self):
        """Test Sunday is detected as weekend."""
        assert DateValidator.is_weekend(date(2024, 1, 14)) is True  # Sunday

    def test_is_weekend_weekday(self):
        """Test weekdays are not weekends."""
        assert DateValidator.is_weekend(date(2024, 1, 15)) is False  # Monday
        assert DateValidator.is_weekend(date(2024, 1, 16)) is False  # Tuesday
        assert DateValidator.is_weekend(date(2024, 1, 17)) is False  # Wednesday
        assert DateValidator.is_weekend(date(2024, 1, 18)) is False  # Thursday
        assert DateValidator.is_weekend(date(2024, 1, 19)) is False  # Friday

    def test_is_weekend_string_date(self):
        """Test weekend detection with string dates."""
        assert DateValidator.is_weekend("2024-01-13") is True  # Saturday
        assert DateValidator.is_weekend("2024-01-15") is False  # Monday

    def test_is_future(self):
        """Test future date detection."""
        future = date.today() + pd.Timedelta(days=30)
        past = date.today() - pd.Timedelta(days=30)
        assert DateValidator.is_future(future) is True
        assert DateValidator.is_future(past) is False

    def test_days_since(self):
        """Test days_since calculation."""
        yesterday = date.today() - pd.Timedelta(days=1)
        result = DateValidator.days_since(yesterday)
        assert result == 1

    def test_days_since_invalid(self):
        """Test days_since with invalid date returns -1."""
        assert DateValidator.days_since("invalid") == -1


class TestDateParserIntegration:
    """Integration tests for DateParser with real-world data formats."""

    def test_fred_date_format(self):
        """Test parsing dates as returned by FRED API."""
        # FRED returns ISO format
        assert DateParser.parse("2024-01-15") == "2024-01-15"

    def test_twelvedata_date_format(self):
        """Test parsing dates as returned by TwelveData API."""
        # TwelveData returns ISO format with optional time
        assert DateParser.parse("2024-01-15") == "2024-01-15"

    def test_investing_date_format(self):
        """Test parsing dates from Investing.com scraping."""
        # Investing.com often returns various formats
        # pandas will handle most of them
        ts = pd.Timestamp("Jan 15, 2024")
        assert DateParser.parse(ts) == "2024-01-15"

    def test_bcrp_embi_date_format(self):
        """Test parsing EMBI dates from BCRP Peru."""
        assert DateParser.parse("06Ene26") == "2026-01-06"
        assert DateParser.parse("17Nov25") == "2025-11-17"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
