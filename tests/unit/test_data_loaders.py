"""
Unit Tests for Unified Data Loaders

Tests the SSOT data loading infrastructure for RL and Forecasting pipelines.

Contract: CTR-DATA-TEST-001
Version: 2.0.0
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.contracts import (
    MACRO_DB_TO_FRIENDLY,
    MACRO_FRIENDLY_TO_DB,
    RL_MACRO_COLUMNS,
    FORECASTING_MACRO_COLUMNS,
    FORECASTING_FEATURES,
    NUM_FORECASTING_FEATURES,
    validate_forecasting_features,
    validate_ohlcv_columns,
)
from src.data.calendar import TradingCalendar, is_trading_day
from src.data.ohlcv_loader import UnifiedOHLCVLoader
from src.data.macro_loader import UnifiedMacroLoader


# =============================================================================
# CONTRACTS TESTS
# =============================================================================

class TestContracts:
    """Tests for data contracts module."""

    def test_macro_mapping_bidirectional(self):
        """Test that DB to friendly and friendly to DB mappings are consistent."""
        for db_col, friendly in MACRO_DB_TO_FRIENDLY.items():
            assert MACRO_FRIENDLY_TO_DB[friendly] == db_col

    def test_rl_macro_columns_are_valid(self):
        """Test that RL macro columns are in the mapping."""
        for col in RL_MACRO_COLUMNS:
            assert col in MACRO_DB_TO_FRIENDLY.values(), f"{col} not in mapping"

    def test_forecasting_macro_columns_are_valid(self):
        """Test that forecasting macro columns are in the mapping."""
        for col in FORECASTING_MACRO_COLUMNS:
            assert col in MACRO_DB_TO_FRIENDLY.values(), f"{col} not in mapping"

    def test_forecasting_features_count(self):
        """Test that there are exactly 19 forecasting features."""
        assert len(FORECASTING_FEATURES) == 19
        assert NUM_FORECASTING_FEATURES == 19

    def test_validate_forecasting_features_pass(self):
        """Test validation passes with all features present."""
        columns = list(FORECASTING_FEATURES) + ['date', 'target_1d']
        is_valid, missing = validate_forecasting_features(columns)
        assert is_valid
        assert len(missing) == 0

    def test_validate_forecasting_features_fail(self):
        """Test validation fails with missing features."""
        columns = list(FORECASTING_FEATURES)[:10]  # Only first 10
        is_valid, missing = validate_forecasting_features(columns)
        assert not is_valid
        assert len(missing) == 9

    def test_validate_ohlcv_columns_pass(self):
        """Test OHLCV validation passes with all columns."""
        columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        is_valid, missing = validate_ohlcv_columns(columns)
        assert is_valid
        assert len(missing) == 0

    def test_validate_ohlcv_columns_fail(self):
        """Test OHLCV validation fails with missing columns."""
        columns = ['time', 'close']
        is_valid, missing = validate_ohlcv_columns(columns)
        assert not is_valid
        assert 'open' in missing
        assert 'high' in missing
        assert 'low' in missing


# =============================================================================
# CALENDAR TESTS
# =============================================================================

class TestTradingCalendar:
    """Tests for trading calendar module."""

    def test_calendar_initialization(self):
        """Test calendar initializes correctly."""
        calendar = TradingCalendar()
        assert calendar is not None

    def test_is_trading_day_weekday(self):
        """Test weekdays are trading days."""
        calendar = TradingCalendar()
        # 2024-01-02 is a Tuesday (weekday, not holiday)
        assert calendar.is_trading_day("2024-01-02")

    def test_is_trading_day_weekend(self):
        """Test weekends are not trading days."""
        calendar = TradingCalendar()
        # 2024-01-06 is a Saturday
        assert not calendar.is_trading_day("2024-01-06")
        # 2024-01-07 is a Sunday
        assert not calendar.is_trading_day("2024-01-07")

    def test_is_trading_day_holiday(self):
        """Test holidays are not trading days."""
        calendar = TradingCalendar()
        # 2024-01-01 is New Year's Day
        assert not calendar.is_trading_day("2024-01-01")
        # 2024-12-25 is Christmas
        assert not calendar.is_trading_day("2024-12-25")

    def test_is_market_hours_true(self):
        """Test timestamps during market hours."""
        calendar = TradingCalendar()
        # 15:00 UTC is within market hours (13:00-18:00)
        ts = pd.Timestamp("2024-01-02 15:00:00", tz='UTC')
        assert calendar.is_market_hours(ts)

    def test_is_market_hours_false(self):
        """Test timestamps outside market hours."""
        calendar = TradingCalendar()
        # 10:00 UTC is before market open (13:00)
        ts = pd.Timestamp("2024-01-02 10:00:00", tz='UTC')
        assert not calendar.is_market_hours(ts)
        # 19:00 UTC is after market close (18:00)
        ts = pd.Timestamp("2024-01-02 19:00:00", tz='UTC')
        assert not calendar.is_market_hours(ts)

    def test_generate_5min_grid(self):
        """Test 5-minute grid generation."""
        calendar = TradingCalendar()
        grid = calendar.generate_5min_grid("2024-01-02", "2024-01-02")

        # Single trading day should have 60 bars (13:00-17:55, 5-min intervals)
        # 5 hours = 300 minutes / 5 = 60 bars
        assert len(grid) == 60

        # First bar should be 13:00
        assert grid[0].hour == 13
        assert grid[0].minute == 0

        # Last bar should be 17:55
        assert grid[-1].hour == 17
        assert grid[-1].minute == 55

    def test_get_trading_days(self):
        """Test getting list of trading days."""
        calendar = TradingCalendar()
        # First week of Jan 2024 (2 is Tue, 3 is Wed, 4 is Thu, 5 is Fri)
        # 1 is New Year's Day (holiday)
        days = calendar.get_trading_days("2024-01-01", "2024-01-05")

        assert len(days) == 4  # Tue, Wed, Thu, Fri
        assert pd.Timestamp("2024-01-01") not in days  # Holiday
        assert pd.Timestamp("2024-01-02") in days


# =============================================================================
# OHLCV LOADER TESTS
# =============================================================================

class TestUnifiedOHLCVLoader:
    """Tests for OHLCV loader."""

    def test_loader_initialization(self):
        """Test loader initializes correctly."""
        loader = UnifiedOHLCVLoader(fallback_csv=True)
        assert loader is not None
        assert loader.fallback_csv == True

    def test_normalize_columns(self):
        """Test column normalization."""
        loader = UnifiedOHLCVLoader()

        df = pd.DataFrame({
            'Time': pd.date_range('2024-01-01', periods=5, freq='5min'),
            'Open': [4000.0] * 5,
            'HIGH': [4010.0] * 5,
            'low': [3990.0] * 5,
            'Close': [4005.0] * 5,
        })

        normalized = loader._normalize_columns(df)

        assert 'time' in normalized.columns
        assert 'open' in normalized.columns
        assert 'high' in normalized.columns
        assert 'low' in normalized.columns
        assert 'close' in normalized.columns
        assert 'volume' in normalized.columns

    def test_resample_to_daily(self):
        """Test resampling 5-min to daily."""
        loader = UnifiedOHLCVLoader()

        # Create mock 5-min data for 2 days
        dates = pd.date_range('2024-01-02 13:00', periods=120, freq='5min')
        df = pd.DataFrame({
            'time': dates,
            'open': [4000 + i for i in range(120)],
            'high': [4010 + i for i in range(120)],
            'low': [3990 + i for i in range(120)],
            'close': [4005 + i for i in range(120)],
            'volume': [100] * 120,
        })

        daily = loader.resample_to_daily(df)

        # Should have 2 days (or 1 depending on how date splits)
        assert len(daily) >= 1
        assert 'date' in daily.columns
        assert 'open' in daily.columns
        assert 'close' in daily.columns

    def test_validate_data_quality_pass(self):
        """Test data quality validation passes for good data."""
        loader = UnifiedOHLCVLoader()

        df = pd.DataFrame({
            'time': pd.date_range('2024-01-02 13:00', periods=5, freq='5min'),
            'open': [4000.0] * 5,
            'high': [4010.0] * 5,
            'low': [3990.0] * 5,
            'close': [4005.0] * 5,
            'volume': [100] * 5,
        })

        result = loader.validate_data_quality(df)

        assert result['is_valid']
        assert len(result['errors']) == 0
        assert result['stats']['rows'] == 5

    def test_validate_data_quality_fail_high_low(self):
        """Test data quality validation fails for high < low."""
        loader = UnifiedOHLCVLoader()

        df = pd.DataFrame({
            'time': pd.date_range('2024-01-02 13:00', periods=5, freq='5min'),
            'open': [4000.0] * 5,
            'high': [3980.0] * 5,  # Invalid: high < low
            'low': [4010.0] * 5,
            'close': [4005.0] * 5,
            'volume': [100] * 5,
        })

        result = loader.validate_data_quality(df)

        assert not result['is_valid']
        assert any('high < low' in err for err in result['errors'])

    def test_load_daily_method_exists(self):
        """Test that load_daily() method exists for Forecasting pipeline."""
        loader = UnifiedOHLCVLoader()
        # Verify method exists
        assert hasattr(loader, 'load_daily')
        assert callable(loader.load_daily)

    def test_resample_to_daily_deprecated_warning(self):
        """Test that resample_to_daily shows deprecation warning."""
        import warnings

        loader = UnifiedOHLCVLoader()
        df = pd.DataFrame({
            'time': pd.date_range('2024-01-02 13:00', periods=60, freq='5min'),
            'open': [4000.0] * 60,
            'high': [4010.0] * 60,
            'low': [3990.0] * 60,
            'close': [4005.0] * 60,
            'volume': [100] * 60,
        })

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = loader.resample_to_daily(df)
            # Check deprecation warning was raised
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "load_daily()" in str(w[0].message)


# =============================================================================
# CONTRACTS TESTS - NEW
# =============================================================================

class TestDataLineage:
    """Tests for data lineage and architecture contracts."""

    def test_rl_ohlcv_table_defined(self):
        """Test RL OHLCV table is defined."""
        from src.data.contracts import RL_OHLCV_TABLE
        assert RL_OHLCV_TABLE == "usdcop_m5_ohlcv"

    def test_forecasting_ohlcv_table_defined(self):
        """Test Forecasting OHLCV table is defined."""
        from src.data.contracts import FORECASTING_OHLCV_TABLE
        assert FORECASTING_OHLCV_TABLE == "bi.dim_daily_usdcop"

    def test_daily_ohlcv_sources(self):
        """Test valid daily OHLCV sources are defined."""
        from src.data.contracts import DAILY_OHLCV_SOURCES
        assert "investing" in DAILY_OHLCV_SOURCES  # Primary
        assert "twelvedata" in DAILY_OHLCV_SOURCES  # Secondary
        assert "resampled_5min" in DAILY_OHLCV_SOURCES  # Fallback

    def test_data_lineage_info(self):
        """Test data lineage documentation."""
        from src.data.contracts import get_data_lineage_info
        info = get_data_lineage_info()

        # RL pipeline
        assert info['rl_pipeline']['frequency'] == "5-minute"
        assert info['rl_pipeline']['loader_method'] == "load_5min()"

        # Forecasting pipeline
        assert info['forecasting_pipeline']['frequency'] == "daily"
        assert info['forecasting_pipeline']['loader_method'] == "load_daily()"
        assert "bi.dim_daily_usdcop" in info['forecasting_pipeline']['ohlcv_table']


# =============================================================================
# MACRO LOADER TESTS
# =============================================================================

class TestUnifiedMacroLoader:
    """Tests for macro loader."""

    def test_loader_initialization(self):
        """Test loader initializes correctly."""
        loader = UnifiedMacroLoader(fallback_parquet=True)
        assert loader is not None
        assert loader.fallback_parquet == True

    def test_apply_friendly_names(self):
        """Test column name mapping."""
        loader = UnifiedMacroLoader()

        df = pd.DataFrame({
            'fecha': pd.date_range('2024-01-01', periods=5),
            'fxrt_index_dxy_usa_d_dxy': [100.0] * 5,
            'volt_vix_usa_d_vix': [20.0] * 5,
        })

        renamed = loader._apply_friendly_names(df)

        assert 'dxy' in renamed.columns
        assert 'vix' in renamed.columns

    def test_add_derived_columns(self):
        """Test derived column calculation."""
        loader = UnifiedMacroLoader()

        df = pd.DataFrame({
            'col10y': [10.0] * 5,
            'ust10y': [4.0] * 5,
            'dxy': [100.0, 101.0, 102.0, 101.5, 102.5],
            'vix': [15.0, 22.0, 28.0, 35.0, 18.0],
        })

        result = loader._add_derived_columns(df)

        # Rate spread
        assert 'rate_spread' in result.columns
        assert result['rate_spread'].iloc[0] == 6.0  # 10 - 4

        # DXY change
        assert 'dxy_change_1d' in result.columns

        # VIX regime
        assert 'vix_regime' in result.columns

    def test_get_available_columns(self):
        """Test getting available column list."""
        loader = UnifiedMacroLoader()
        columns = loader.get_available_columns()

        assert 'dxy' in columns
        assert 'vix' in columns
        assert 'embi' in columns


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestDataLoaderIntegration:
    """Integration tests for data loaders working together."""

    @pytest.fixture
    def mock_ohlcv_data(self):
        """Create mock OHLCV data."""
        dates = pd.date_range('2024-01-02 13:00', periods=60, freq='5min')
        return pd.DataFrame({
            'time': dates,
            'open': [4000 + np.random.randn() * 10 for _ in range(60)],
            'high': [4010 + np.random.randn() * 10 for _ in range(60)],
            'low': [3990 + np.random.randn() * 10 for _ in range(60)],
            'close': [4005 + np.random.randn() * 10 for _ in range(60)],
            'volume': [0] * 60,
        })

    @pytest.fixture
    def mock_macro_data(self):
        """Create mock macro data."""
        dates = pd.date_range('2024-01-01', periods=10)
        return pd.DataFrame({
            'date': dates,
            'dxy': [100 + i * 0.1 for i in range(10)],
            'vix': [20 + i * 0.5 for i in range(10)],
            'wti': [70 + i * 0.2 for i in range(10)],
            'embi': [300 + i * 2 for i in range(10)],
        })

    def test_ohlcv_to_daily_flow(self, mock_ohlcv_data):
        """Test OHLCV loading and resampling to daily."""
        loader = UnifiedOHLCVLoader()

        # Resample to daily
        daily = loader.resample_to_daily(mock_ohlcv_data)

        assert len(daily) >= 1
        assert all(col in daily.columns for col in ['date', 'open', 'high', 'low', 'close'])

    def test_macro_column_consistency(self):
        """Test that RL and Forecasting use same macro column names."""
        # Both should reference friendly names from contracts
        rl_cols = set(RL_MACRO_COLUMNS)
        forecast_cols = set(FORECASTING_MACRO_COLUMNS)

        # DXY should be in both
        assert 'dxy' in rl_cols
        assert 'dxy' in forecast_cols


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
