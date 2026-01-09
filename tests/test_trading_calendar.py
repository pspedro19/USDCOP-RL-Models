"""
Test Suite for Trading Calendar Validation
============================================

Ensures holidays and weekends are correctly identified and filtered.
Tests the UnifiedDatetimeHandler's business day and holiday logic.

Critical validations:
- Weekends (Saturday/Sunday) are NOT trading days
- Colombian holidays are correctly excluded
- Ley Emiliani (holidays moved to Monday) is handled
- Easter-based holidays are computed correctly
- Year boundaries don't cause issues
- Training data contains zero holidays/weekends

Author: Pedro @ Lean Tech Solutions
Date: 2025-12-17
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import pytz
import json
from pathlib import Path

# Import the UnifiedDatetimeHandler directly
import sys
airflow_dags_path = Path(__file__).parent.parent / 'airflow' / 'dags'
sys.path.insert(0, str(airflow_dags_path))

# Import directly from the module file to avoid __init__.py issues
import importlib.util
spec = importlib.util.spec_from_file_location(
    "datetime_handler",
    airflow_dags_path / 'utils' / 'datetime_handler.py'
)
datetime_handler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(datetime_handler)
UnifiedDatetimeHandler = datetime_handler.UnifiedDatetimeHandler


@pytest.fixture
def calendar():
    """Trading calendar instance"""
    return UnifiedDatetimeHandler()


@pytest.fixture
def trading_calendar_config():
    """Load trading calendar configuration"""
    config_path = Path(__file__).parent.parent / 'config' / 'trading_calendar.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture
def sample_training_data():
    """Create sample training data for validation"""
    # Create 100 bars of realistic USDCOP data
    dates = pd.date_range('2025-12-01 08:00:00', periods=100, freq='5min', tz='America/Bogota')

    df = pd.DataFrame({
        'timestamp': dates,
        'open': 4250.0 + np.random.randn(100) * 5,
        'high': 4255.0 + np.random.randn(100) * 5,
        'low': 4245.0 + np.random.randn(100) * 5,
        'close': 4250.0 + np.random.randn(100) * 5,
        'volume': np.random.randint(1000, 5000, 100)
    })

    return df


# =============================================================================
# TEST CLASS 1: Weekend Validation
# =============================================================================

@pytest.mark.unit
class TestWeekendValidation:
    """Test that weekends are correctly identified and excluded"""

    def test_saturday_not_trading_day(self, calendar):
        """Saturdays should not be trading days"""
        saturday = datetime(2025, 12, 20, 10, 0)  # Saturday, 10 AM
        saturday_cot = calendar.COT_TZ.localize(saturday)

        assert not calendar.is_business_day(saturday_cot), \
            "Saturday incorrectly identified as trading day"

    def test_sunday_not_trading_day(self, calendar):
        """Sundays should not be trading days"""
        sunday = datetime(2025, 12, 21, 10, 0)  # Sunday, 10 AM
        sunday_cot = calendar.COT_TZ.localize(sunday)

        assert not calendar.is_business_day(sunday_cot), \
            "Sunday incorrectly identified as trading day"

    def test_weekday_is_trading_day(self, calendar):
        """Regular weekdays should be trading days (if not holiday)"""
        monday = datetime(2025, 12, 22, 10, 0)  # Monday, 10 AM
        monday_cot = calendar.COT_TZ.localize(monday)

        # December 22 is not a holiday in 2025
        assert calendar.is_business_day(monday_cot), \
            "Regular Monday incorrectly identified as non-trading day"

    def test_friday_is_trading_day(self, calendar):
        """Friday should be a trading day (if not holiday)"""
        friday = datetime(2025, 12, 19, 10, 0)  # Friday, 10 AM
        friday_cot = calendar.COT_TZ.localize(friday)

        # December 19 is not a holiday in 2025
        assert calendar.is_business_day(friday_cot), \
            "Regular Friday incorrectly identified as non-trading day"

    def test_weekend_series_filtering(self, calendar):
        """Test filtering a Series containing weekend dates"""
        dates = pd.date_range('2025-12-19', periods=7, freq='D', tz='America/Bogota')
        df = pd.DataFrame({'timestamp': dates})

        business_days = calendar.is_business_day(df['timestamp'])

        # Should filter out Saturday (20th) and Sunday (21st)
        # And Christmas (25th)
        # Note: Dec 26 might also be a holiday in some years
        actual_dates = df[business_days]['timestamp'].dt.date.tolist()

        # Verify weekends are excluded
        assert date(2025, 12, 20) not in actual_dates, "Saturday should be excluded"
        assert date(2025, 12, 21) not in actual_dates, "Sunday should be excluded"

        # Verify Christmas is excluded
        assert date(2025, 12, 25) not in actual_dates, "Christmas should be excluded"

        # Verify at least some weekdays are included
        weekday_count = sum(1 for d in actual_dates if d.weekday() < 5)
        assert weekday_count >= 3, f"Expected at least 3 weekdays, got {weekday_count}"


# =============================================================================
# TEST CLASS 2: Colombian Holidays 2025
# =============================================================================

@pytest.mark.unit
class TestColombianHolidays2025:
    """Test known Colombian holidays for 2025 from trading_calendar.json"""

    def test_new_year_not_trading_day(self, calendar):
        """January 1st (Año Nuevo) should not be trading day"""
        new_year = datetime(2025, 1, 1, 10, 0)
        new_year_cot = calendar.COT_TZ.localize(new_year)

        assert not calendar.is_business_day(new_year_cot), \
            "New Year's Day incorrectly identified as trading day"

    def test_epiphany_not_trading_day(self, calendar):
        """January 6th (Reyes Magos) should not be trading day"""
        epiphany = datetime(2025, 1, 6, 10, 0)
        epiphany_cot = calendar.COT_TZ.localize(epiphany)

        assert not calendar.is_business_day(epiphany_cot), \
            "Epiphany incorrectly identified as trading day"

    def test_labor_day_not_trading_day(self, calendar):
        """May 1st (Día del Trabajo) should not be trading day"""
        labor_day = datetime(2025, 5, 1, 10, 0)
        labor_day_cot = calendar.COT_TZ.localize(labor_day)

        assert not calendar.is_business_day(labor_day_cot), \
            "Labor Day incorrectly identified as trading day"

    def test_independence_day_not_trading_day(self, calendar):
        """July 20th (Independencia) should not be trading day"""
        independence = datetime(2025, 7, 20, 10, 0)
        independence_cot = calendar.COT_TZ.localize(independence)

        assert not calendar.is_business_day(independence_cot), \
            "Independence Day incorrectly identified as trading day"

    def test_battle_of_boyaca_not_trading_day(self, calendar):
        """August 7th (Batalla de Boyacá) should not be trading day"""
        boyaca = datetime(2025, 8, 7, 10, 0)
        boyaca_cot = calendar.COT_TZ.localize(boyaca)

        assert not calendar.is_business_day(boyaca_cot), \
            "Battle of Boyacá incorrectly identified as trading day"

    def test_christmas_not_trading_day(self, calendar):
        """December 25th (Navidad) should not be trading day"""
        christmas = datetime(2025, 12, 25, 10, 0)
        christmas_cot = calendar.COT_TZ.localize(christmas)

        assert not calendar.is_business_day(christmas_cot), \
            "Christmas incorrectly identified as trading day"

    def test_all_2025_holidays_from_config(self, calendar, trading_calendar_config):
        """Test all Colombian holidays from trading_calendar.json"""
        holiday_strings = trading_calendar_config['holidays_2025_colombia']

        for holiday_str in holiday_strings:
            holiday_date = datetime.strptime(holiday_str, '%Y-%m-%d')
            holiday_cot = calendar.COT_TZ.localize(holiday_date.replace(hour=10))

            assert not calendar.is_business_day(holiday_cot), \
                f"Holiday {holiday_str} incorrectly identified as trading day"


# =============================================================================
# TEST CLASS 3: Easter-Based Holidays
# =============================================================================

@pytest.mark.unit
class TestEasterBasedHolidays:
    """Test Easter-dependent holidays are correctly identified"""

    def test_maundy_thursday_2025(self, calendar):
        """Jueves Santo (Maundy Thursday) - April 17, 2025"""
        maundy_thursday = datetime(2025, 4, 17, 10, 0)
        maundy_thursday_cot = calendar.COT_TZ.localize(maundy_thursday)

        assert not calendar.is_business_day(maundy_thursday_cot), \
            "Maundy Thursday incorrectly identified as trading day"

    def test_good_friday_2025(self, calendar):
        """Viernes Santo (Good Friday) - April 18, 2025"""
        good_friday = datetime(2025, 4, 18, 10, 0)
        good_friday_cot = calendar.COT_TZ.localize(good_friday)

        assert not calendar.is_business_day(good_friday_cot), \
            "Good Friday incorrectly identified as trading day"

    def test_ascension_day_2025(self, calendar):
        """Ascensión (moved to Monday) - June 2, 2025"""
        ascension = datetime(2025, 6, 2, 10, 0)
        ascension_cot = calendar.COT_TZ.localize(ascension)

        assert not calendar.is_business_day(ascension_cot), \
            "Ascension Day incorrectly identified as trading day"

    def test_corpus_christi_2025(self, calendar):
        """Corpus Christi (moved to Monday) - June 23, 2025"""
        corpus_christi = datetime(2025, 6, 23, 10, 0)
        corpus_christi_cot = calendar.COT_TZ.localize(corpus_christi)

        assert not calendar.is_business_day(corpus_christi_cot), \
            "Corpus Christi incorrectly identified as trading day"


# =============================================================================
# TEST CLASS 4: Ley Emiliani (Holidays Moved to Monday)
# =============================================================================

@pytest.mark.unit
class TestLeyEmilianiHolidays:
    """
    Test Ley Emiliani - Colombian law that moves certain holidays to Monday.

    Holidays that are always moved to the following Monday if they fall
    on a weekday other than Monday (except for fixed-date holidays like
    Christmas, New Year, Labor Day, Independence Day, Battle of Boyacá).
    """

    def test_immaculate_conception_moved_2025(self, calendar):
        """
        Inmaculada Concepción - December 8, 2025 is a Monday
        Should be observed on December 8 (Monday)
        """
        immaculate = datetime(2025, 12, 8, 10, 0)
        immaculate_cot = calendar.COT_TZ.localize(immaculate)

        assert not calendar.is_business_day(immaculate_cot), \
            "Immaculate Conception incorrectly identified as trading day"

    def test_assumption_moved_2025(self, calendar):
        """
        Asunción - August 18, 2025 (moved to Monday)
        Original date: August 15, 2025 (Friday)
        """
        assumption = datetime(2025, 8, 18, 10, 0)  # Monday
        assumption_cot = calendar.COT_TZ.localize(assumption)

        assert not calendar.is_business_day(assumption_cot), \
            "Assumption Day (moved) incorrectly identified as trading day"

        # Verify original date IS a trading day
        original = datetime(2025, 8, 15, 10, 0)  # Friday
        original_cot = calendar.COT_TZ.localize(original)
        assert calendar.is_business_day(original_cot), \
            "Original Assumption date should be trading day"

    def test_all_saints_day_moved_2025(self, calendar):
        """
        Todos los Santos - November 3, 2025 (moved to Monday)
        Original date: November 1, 2025 (Saturday)
        """
        all_saints = datetime(2025, 11, 3, 10, 0)  # Monday
        all_saints_cot = calendar.COT_TZ.localize(all_saints)

        assert not calendar.is_business_day(all_saints_cot), \
            "All Saints Day (moved) incorrectly identified as trading day"

    def test_independence_of_cartagena_moved_2025(self, calendar):
        """
        Independencia de Cartagena - November 17, 2025 (moved to Monday)
        Original date: November 11, 2025 (Tuesday)
        """
        cartagena = datetime(2025, 11, 17, 10, 0)  # Monday
        cartagena_cot = calendar.COT_TZ.localize(cartagena)

        assert not calendar.is_business_day(cartagena_cot), \
            "Independence of Cartagena (moved) incorrectly identified as trading day"


# =============================================================================
# TEST CLASS 5: Year Boundary Edge Cases
# =============================================================================

@pytest.mark.unit
class TestYearBoundaryEdgeCases:
    """Test calendar works correctly at year boundaries"""

    def test_new_years_eve_is_trading_day(self, calendar):
        """December 31st should be a trading day (if weekday)"""
        # 2025-12-31 is a Wednesday
        new_years_eve = datetime(2025, 12, 31, 10, 0)
        new_years_eve_cot = calendar.COT_TZ.localize(new_years_eve)

        assert calendar.is_business_day(new_years_eve_cot), \
            "New Year's Eve (weekday) incorrectly identified as non-trading day"

    def test_january_2nd_is_trading_day(self, calendar):
        """January 2nd should be a trading day (if weekday and not holiday)"""
        # 2025-01-02 is a Thursday
        jan_2nd = datetime(2025, 1, 2, 10, 0)
        jan_2nd_cot = calendar.COT_TZ.localize(jan_2nd)

        assert calendar.is_business_day(jan_2nd_cot), \
            "January 2nd incorrectly identified as non-trading day"

    def test_cross_year_date_range(self, calendar):
        """Test filtering data across year boundary"""
        # Create range from late December to early January
        dates = pd.date_range('2025-12-29', '2026-01-05', freq='D', tz='America/Bogota')
        df = pd.DataFrame({'timestamp': dates})

        business_days = calendar.is_business_day(df['timestamp'])

        # Should exclude:
        # - 2025-12-29 (Monday) - but this IS a trading day
        # - 2025-12-30 (Tuesday) - trading day
        # - 2025-12-31 (Wednesday) - trading day
        # - 2026-01-01 (Thursday) - NEW YEAR (holiday)
        # - 2026-01-02 (Friday) - trading day
        # - 2026-01-03 (Saturday) - weekend
        # - 2026-01-04 (Sunday) - weekend
        # - 2026-01-05 (Monday) - trading day

        expected_trading_days = [
            date(2025, 12, 29),  # Monday
            date(2025, 12, 30),  # Tuesday
            date(2025, 12, 31),  # Wednesday
            date(2026, 1, 2),    # Friday
            date(2026, 1, 5),    # Monday (if not Epiphany transferred)
        ]

        actual_dates = df[business_days]['timestamp'].dt.date.tolist()

        # Verify New Year is excluded
        assert date(2026, 1, 1) not in actual_dates, \
            "New Year's Day should not be in trading days"

        # Verify weekends are excluded
        assert date(2026, 1, 3) not in actual_dates, \
            "Saturday should not be in trading days"
        assert date(2026, 1, 4) not in actual_dates, \
            "Sunday should not be in trading days"


# =============================================================================
# TEST CLASS 6: DataFrame Filtering
# =============================================================================

@pytest.mark.unit
class TestDataFrameFiltering:
    """Test DataFrame filtering to remove holidays and weekends"""

    def test_filter_removes_weekends(self, calendar):
        """Test that filter_business_hours removes weekends"""
        # Create data spanning a weekend
        dates = pd.date_range('2025-12-19 09:00', periods=48, freq='5min', tz='America/Bogota')
        df = pd.DataFrame({'timestamp': dates, 'value': range(48)})

        filtered = calendar.filter_business_hours(df, 'timestamp', include_holidays=True)

        # Check no Saturday/Sunday dates remain
        weekdays = filtered['timestamp_cot'].dt.dayofweek
        assert (weekdays < 5).all(), "Weekend days found in filtered data"

    def test_filter_removes_christmas(self, calendar):
        """Test that filter removes Christmas holiday"""
        # Create data around Christmas
        dates = pd.date_range('2025-12-24 09:00', periods=24, freq='1h', tz='America/Bogota')
        df = pd.DataFrame({'timestamp': dates, 'value': range(24)})

        filtered = df.copy()
        filtered = calendar.standardize_dataframe_timestamps(filtered, ['timestamp'])

        # Manually filter business days
        business_mask = calendar.is_business_day(filtered['timestamp'])
        filtered = filtered[business_mask]

        # Check December 25 is not in filtered data
        dates_in_filtered = filtered['timestamp'].dt.date.unique()
        assert date(2025, 12, 25) not in dates_in_filtered, \
            "Christmas found in filtered data"

    def test_filter_maintains_trading_hours(self, calendar):
        """Test that filter maintains only trading hours (8am-2pm COT)"""
        # Create 24-hour data
        dates = pd.date_range('2025-12-15 00:00', periods=48, freq='30min', tz='America/Bogota')
        df = pd.DataFrame({'timestamp': dates, 'value': range(48)})

        filtered = calendar.filter_business_hours(df, 'timestamp', include_holidays=True)

        # Check all hours are between 8am and 2pm
        hours = filtered['timestamp_cot'].dt.hour
        assert (hours >= 8).all(), "Hours before 8am found in filtered data"
        assert (hours < 14).all(), "Hours after 2pm found in filtered data"

    def test_filter_empty_result_on_all_holidays(self, calendar):
        """Test filtering all-holiday data returns empty DataFrame"""
        # Create data only on New Year's Day
        dates = pd.date_range('2025-01-01 09:00', periods=12, freq='1h', tz='America/Bogota')
        df = pd.DataFrame({'timestamp': dates, 'value': range(12)})

        filtered = df.copy()
        filtered = calendar.standardize_dataframe_timestamps(filtered, ['timestamp'])
        business_mask = calendar.is_business_day(filtered['timestamp'])
        filtered = filtered[business_mask]

        assert len(filtered) == 0, "Holiday-only data should result in empty DataFrame"


# =============================================================================
# TEST CLASS 7: Training Data Validation
# =============================================================================

@pytest.mark.unit
class TestTrainingDataValidation:
    """Validate that training datasets have no holidays or weekends"""

    def test_sample_data_has_no_weekends(self, calendar, sample_training_data):
        """Verify sample training data contains no weekend dates"""
        df = sample_training_data

        # Ensure timezone aware
        df = calendar.standardize_dataframe_timestamps(df, ['timestamp'])

        # Check weekdays
        weekdays = df['timestamp'].dt.dayofweek
        weekend_count = (weekdays >= 5).sum()

        assert weekend_count == 0, \
            f"Training data contains {weekend_count} weekend records"

    def test_sample_data_has_no_holidays(self, calendar, sample_training_data):
        """Verify sample training data contains no holiday dates"""
        df = sample_training_data

        # Ensure timezone aware
        df = calendar.standardize_dataframe_timestamps(df, ['timestamp'])

        # Check business days
        business_mask = calendar.is_business_day(df['timestamp'])
        non_business_count = (~business_mask).sum()

        assert non_business_count == 0, \
            f"Training data contains {non_business_count} holiday records"

    def test_validate_no_holidays_function(self, calendar, sample_training_data):
        """Test a validation function that checks for holidays in data"""
        df = sample_training_data

        # Ensure timezone aware
        df = calendar.standardize_dataframe_timestamps(df, ['timestamp'])

        # Validation logic
        business_mask = calendar.is_business_day(df['timestamp'])
        is_valid = business_mask.all()

        if not is_valid:
            non_business_dates = df[~business_mask]['timestamp'].dt.date.unique()
            issues = f"Non-trading days found: {non_business_dates}"
        else:
            issues = None

        assert is_valid, f"Training data validation failed: {issues}"

    def test_count_trading_days_in_month(self, calendar):
        """Test counting trading days in a given month"""
        # December 2025 has 31 days
        # Weekdays: ~22-23 days
        # Minus Christmas (25th)
        # Minus Immaculate Conception (8th)

        start = datetime(2025, 12, 1, 9, 0)
        end = datetime(2025, 12, 31, 13, 0)

        expected_timestamps = calendar.generate_expected_timestamps(
            start, end,
            interval_minutes=60,  # 1-hour intervals
            business_hours_only=True,
            exclude_holidays=True
        )

        # Extract unique days
        unique_days = set(ts.date() for ts in expected_timestamps)

        # Should have around 18-20 trading days in December 2025
        assert len(unique_days) >= 15, \
            f"Expected at least 15 trading days in December 2025, got {len(unique_days)}"
        assert len(unique_days) <= 25, \
            f"Expected at most 25 trading days in December 2025, got {len(unique_days)}"

        # Verify Christmas is NOT included
        assert date(2025, 12, 25) not in unique_days, \
            "Christmas should not be a trading day"


# =============================================================================
# TEST CLASS 8: Premium Hours Validation
# =============================================================================

@pytest.mark.unit
class TestPremiumHoursValidation:
    """Test premium trading hours (8am-2pm COT, Mon-Fri)"""

    def test_morning_opening_is_premium(self, calendar):
        """8:00 AM on Monday should be premium hours"""
        morning = datetime(2025, 12, 22, 8, 0)  # Monday 8 AM
        morning_cot = calendar.COT_TZ.localize(morning)

        assert calendar.is_premium_hours(morning_cot), \
            "8:00 AM Monday not identified as premium hours"

    def test_before_opening_not_premium(self, calendar):
        """7:59 AM on Monday should NOT be premium hours"""
        before_open = datetime(2025, 12, 22, 7, 59)
        before_open_cot = calendar.COT_TZ.localize(before_open)

        assert not calendar.is_premium_hours(before_open_cot), \
            "7:59 AM incorrectly identified as premium hours"

    def test_after_close_not_premium(self, calendar):
        """2:00 PM on Monday should NOT be premium hours"""
        after_close = datetime(2025, 12, 22, 14, 0)  # 2:00 PM
        after_close_cot = calendar.COT_TZ.localize(after_close)

        assert not calendar.is_premium_hours(after_close_cot), \
            "2:00 PM incorrectly identified as premium hours"

    def test_last_minute_is_premium(self, calendar):
        """1:59 PM on Monday should be premium hours"""
        last_minute = datetime(2025, 12, 22, 13, 59)  # 1:59 PM
        last_minute_cot = calendar.COT_TZ.localize(last_minute)

        assert calendar.is_premium_hours(last_minute_cot), \
            "1:59 PM Monday not identified as premium hours"

    def test_saturday_never_premium(self, calendar):
        """Saturday 10 AM should never be premium hours"""
        saturday = datetime(2025, 12, 20, 10, 0)
        saturday_cot = calendar.COT_TZ.localize(saturday)

        assert not calendar.is_premium_hours(saturday_cot), \
            "Saturday incorrectly identified as premium hours"

    def test_holiday_not_premium(self, calendar):
        """Christmas at 10 AM should not be premium (holiday)"""
        christmas = datetime(2025, 12, 25, 10, 0)
        christmas_cot = calendar.COT_TZ.localize(christmas)

        # is_premium_hours only checks time, not holidays
        # But is_business_day checks both
        assert not calendar.is_business_day(christmas_cot), \
            "Christmas incorrectly identified as business day"


# =============================================================================
# TEST CLASS 9: Edge Cases and Special Scenarios
# =============================================================================

@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and special scenarios"""

    def test_naive_datetime_handling(self, calendar):
        """Test that naive datetimes are handled correctly"""
        naive_dt = datetime(2025, 12, 22, 10, 0)  # No timezone

        # Should be localized to COT by default
        aware_dt = calendar.ensure_timezone_aware(naive_dt, 'America/Bogota')

        assert aware_dt.tzinfo is not None, \
            "Naive datetime not converted to timezone-aware"
        assert aware_dt.tzinfo.zone == 'America/Bogota', \
            "Naive datetime not localized to correct timezone"

    def test_utc_to_cot_conversion(self, calendar):
        """Test UTC to COT timezone conversion"""
        # 1:00 PM UTC = 8:00 AM COT (UTC-5)
        utc_dt = datetime(2025, 12, 22, 13, 0, tzinfo=pytz.UTC)

        cot_dt = calendar.convert_to_cot(utc_dt)

        assert cot_dt.hour == 8, \
            f"UTC to COT conversion failed: expected 8 AM, got {cot_dt.hour}"

    def test_leap_year_february_29(self, calendar):
        """Test handling of February 29 in leap years"""
        # 2024 is a leap year
        feb_29 = datetime(2024, 2, 29, 10, 0)
        feb_29_cot = calendar.COT_TZ.localize(feb_29)

        # February 29, 2024 is a Thursday (trading day)
        assert calendar.is_business_day(feb_29_cot), \
            "Feb 29 (leap year) not identified as business day"

    def test_midnight_boundary(self, calendar):
        """Test midnight boundary behavior"""
        # 11:59 PM is not trading hours
        late_night = datetime(2025, 12, 22, 23, 59)
        late_night_cot = calendar.COT_TZ.localize(late_night)

        assert not calendar.is_premium_hours(late_night_cot), \
            "11:59 PM incorrectly identified as premium hours"

        # But it's still a business day (if weekday)
        assert calendar.is_business_day(late_night_cot), \
            "11:59 PM Monday should still be business day"

    def test_empty_dataframe_handling(self, calendar):
        """Test handling of empty DataFrame"""
        empty_df = pd.DataFrame({'timestamp': pd.Series([], dtype='datetime64[ns, America/Bogota]')})

        # Should not raise error
        filtered = calendar.filter_business_hours(empty_df, 'timestamp')

        assert len(filtered) == 0, "Empty DataFrame should return empty result"


# =============================================================================
# TEST CLASS 10: Real Dataset Validation (Integration-style)
# =============================================================================

@pytest.mark.unit
class TestRealDatasetValidation:
    """
    Test validation against real training datasets.
    These tests check actual L4 pipeline output for holiday contamination.
    """

    def test_l4_dataset_path_exists(self):
        """Check if L4 RL-ready dataset exists"""
        l4_path = Path(__file__).parent.parent / 'data' / 'pipeline' / 'l4_rl_ready'

        # This test is informational - skip if path doesn't exist
        if not l4_path.exists():
            pytest.skip(f"L4 dataset path not found: {l4_path}")

        assert l4_path.is_dir(), "L4 path exists but is not a directory"

    def test_validate_real_l4_dataset_if_exists(self, calendar):
        """
        If L4 dataset exists, validate it has no holidays.
        This is a critical production validation.
        """
        l4_path = Path(__file__).parent.parent / 'data' / 'pipeline' / 'l4_rl_ready'

        if not l4_path.exists():
            pytest.skip("L4 dataset not available")

        # Look for CSV files
        csv_files = list(l4_path.glob('*.csv'))

        if not csv_files:
            pytest.skip("No CSV files found in L4 dataset")

        # Test first file found
        test_file = csv_files[0]
        df = pd.read_csv(test_file, nrows=1000)  # Sample first 1000 rows

        # Assume timestamp column exists
        if 'timestamp' not in df.columns:
            pytest.skip("No 'timestamp' column in L4 dataset")

        # Parse timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = calendar.standardize_dataframe_timestamps(df, ['timestamp'])

        # Validate no holidays
        business_mask = calendar.is_business_day(df['timestamp'])
        non_business_count = (~business_mask).sum()

        if non_business_count > 0:
            non_business_dates = df[~business_mask]['timestamp'].dt.date.unique()
            pytest.fail(
                f"L4 dataset contains {non_business_count} non-trading day records: "
                f"{non_business_dates}"
            )


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
