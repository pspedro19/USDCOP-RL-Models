"""
Temporal Joins Unit Tests
=========================
P0-08: Tests for temporal join operations to prevent lookahead bias.

This module validates that merge_asof operations:
- Use backward direction (only past data)
- Do not have lookahead bias
- Respect tolerance constraints properly

Contract ID: CTR-TEMPORAL-001
Author: Trading Team
Date: 2026-01-17
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src" / "data"))

# Import safe merge operations
try:
    from safe_merge import (
        safe_merge_macro,
        validate_no_future_data,
        FFILL_LIMIT_5MIN,
    )
    SAFE_MERGE_AVAILABLE = True
except ImportError:
    SAFE_MERGE_AVAILABLE = False


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def intraday_ohlcv():
    """
    Create sample intraday OHLCV data (5-minute bars).

    Returns DataFrame with:
    - datetime: 5-minute intervals during trading hours
    - close: Synthetic price data
    """
    dates = pd.date_range("2025-01-10 13:00", periods=50, freq="5min")
    np.random.seed(42)

    return pd.DataFrame({
        "datetime": dates,
        "open": 4250.0 + np.random.randn(50) * 10,
        "high": 4260.0 + np.abs(np.random.randn(50)) * 5,
        "low": 4240.0 - np.abs(np.random.randn(50)) * 5,
        "close": 4250.0 + np.cumsum(np.random.randn(50) * 2),
        "volume": np.random.uniform(1e6, 5e6, 50),
    })


@pytest.fixture
def daily_macro():
    """
    Create sample daily macro data.

    Returns DataFrame with:
    - datetime: Daily dates
    - dxy: Dollar index values
    - vix: Volatility index values
    """
    dates = pd.date_range("2025-01-05", periods=10, freq="D")

    return pd.DataFrame({
        "datetime": dates,
        "dxy": 104.0 + np.arange(10) * 0.1,
        "vix": 18.0 + np.arange(10) * 0.5,
        "embi": 350.0 + np.arange(10) * 2,
    })


@pytest.fixture
def misaligned_macro():
    """
    Create macro data with timestamps that could cause lookahead if not handled properly.

    The timestamps are set to end-of-day which might be after some intraday bars.
    """
    dates = pd.date_range("2025-01-05 23:59:59", periods=10, freq="D")

    return pd.DataFrame({
        "datetime": dates,
        "dxy": 104.0 + np.arange(10) * 0.1,
        "vix": 18.0 + np.arange(10) * 0.5,
    })


# =============================================================================
# Test merge_asof Uses Backward Direction
# =============================================================================

class TestMergeAsofBackwardDirection:
    """
    Tests that merge_asof operations use backward direction.

    This is critical to prevent lookahead bias: we should only use
    data that was available at the time of the observation.
    """

    def test_merge_asof_uses_backward_direction(self, intraday_ohlcv, daily_macro):
        """
        Verify that merge_asof uses direction='backward' to only use past data.

        Given:
        - Intraday data on 2025-01-10
        - Daily macro data from 2025-01-05 to 2025-01-14

        The merge should only use macro data from 2025-01-10 or earlier,
        never from 2025-01-11 onwards.
        """
        # Perform merge using pandas merge_asof with backward direction
        result = pd.merge_asof(
            intraday_ohlcv.sort_values("datetime"),
            daily_macro.sort_values("datetime"),
            on="datetime",
            direction="backward"  # Critical: only use past data
        )

        # Verify: For each row, the macro data date should be <= the ohlcv date
        ohlcv_dates = pd.to_datetime(result["datetime"]).dt.date
        macro_source_dates = pd.to_datetime(daily_macro["datetime"]).dt.date

        # Find which macro date was matched for each row
        for idx, row in result.iterrows():
            row_date = pd.to_datetime(row["datetime"]).date()

            # The DXY value should come from a date on or before row_date
            matched_dxy = row["dxy"]
            if pd.notna(matched_dxy):
                # Find which macro row has this DXY value
                macro_row = daily_macro[daily_macro["dxy"] == matched_dxy]
                if not macro_row.empty:
                    macro_date = pd.to_datetime(macro_row["datetime"].iloc[0]).date()
                    assert macro_date <= row_date, \
                        f"Lookahead detected: macro date {macro_date} > row date {row_date}"

    def test_backward_direction_with_gaps(self, intraday_ohlcv):
        """
        Verify backward direction handles gaps in macro data correctly.

        If there's no macro data for several days, the merge should
        use the most recent available data, not skip ahead.
        """
        # Create macro data with a gap
        macro_with_gap = pd.DataFrame({
            "datetime": pd.to_datetime(["2025-01-05", "2025-01-06", "2025-01-12"]),
            "indicator": [100.0, 101.0, 112.0],
        })

        result = pd.merge_asof(
            intraday_ohlcv.sort_values("datetime"),
            macro_with_gap.sort_values("datetime"),
            on="datetime",
            direction="backward"
        )

        # All rows on 2025-01-10 should use data from 2025-01-06 (not 2025-01-12)
        for idx, row in result.iterrows():
            if pd.notna(row["indicator"]):
                # Should be 101.0 (from 2025-01-06), not 112.0 (from 2025-01-12)
                assert row["indicator"] == 101.0, \
                    f"Expected 101.0 but got {row['indicator']} - possible lookahead"


# =============================================================================
# Test No Lookahead Bias
# =============================================================================

class TestNoLookaheadBias:
    """
    Tests that temporal joins do not introduce lookahead bias.

    Lookahead bias occurs when information from the future is used
    to make decisions about the present.
    """

    def test_no_lookahead_bias_detected(self, intraday_ohlcv, daily_macro):
        """
        Comprehensive test that no lookahead bias exists in temporal joins.

        Tests by modifying future data and verifying present predictions don't change.
        """
        # Original merge
        original_result = pd.merge_asof(
            intraday_ohlcv.sort_values("datetime"),
            daily_macro.sort_values("datetime"),
            on="datetime",
            direction="backward"
        )

        # Modify macro data for future dates (after 2025-01-10)
        modified_macro = daily_macro.copy()
        future_mask = pd.to_datetime(modified_macro["datetime"]) > pd.Timestamp("2025-01-10")
        modified_macro.loc[future_mask, "dxy"] = 999.0  # Significant change
        modified_macro.loc[future_mask, "vix"] = 999.0

        # Merge with modified data
        modified_result = pd.merge_asof(
            intraday_ohlcv.sort_values("datetime"),
            modified_macro.sort_values("datetime"),
            on="datetime",
            direction="backward"
        )

        # Results should be identical since we only changed future data
        pd.testing.assert_frame_equal(
            original_result,
            modified_result,
            check_dtype=False,
            obj="Lookahead bias detected: results changed when future data was modified"
        )

    def test_lookahead_would_fail_with_forward_direction(self, intraday_ohlcv, daily_macro):
        """
        Demonstrate that forward direction WOULD introduce lookahead bias.

        This test shows why backward direction is critical.
        """
        # Merge with forward direction (WRONG - introduces lookahead)
        forward_result = pd.merge_asof(
            intraday_ohlcv.sort_values("datetime"),
            daily_macro.sort_values("datetime"),
            on="datetime",
            direction="forward"
        )

        # Merge with backward direction (CORRECT - no lookahead)
        backward_result = pd.merge_asof(
            intraday_ohlcv.sort_values("datetime"),
            daily_macro.sort_values("datetime"),
            on="datetime",
            direction="backward"
        )

        # Results should be different
        # Forward direction would use 2025-01-11 data for 2025-01-10 rows
        # Backward direction uses 2025-01-10 or earlier
        assert not forward_result.equals(backward_result), \
            "Forward and backward results should differ"

    def test_no_lookahead_at_boundary(self, intraday_ohlcv):
        """
        Test boundary condition: exact timestamp match should use that data.

        When macro data timestamp exactly matches intraday timestamp,
        it should be included (it's not future data).
        """
        # Create macro with exact matching timestamp
        exact_match_macro = pd.DataFrame({
            "datetime": pd.to_datetime(["2025-01-10 13:00"]),  # Exact match
            "indicator": [100.0],
        })

        result = pd.merge_asof(
            intraday_ohlcv.sort_values("datetime"),
            exact_match_macro.sort_values("datetime"),
            on="datetime",
            direction="backward"
        )

        # First row (13:00) should have the indicator
        first_row = result[result["datetime"] == pd.Timestamp("2025-01-10 13:00")]
        assert first_row["indicator"].iloc[0] == 100.0, \
            "Exact timestamp match should be included"


# =============================================================================
# Test Tolerance Handling
# =============================================================================

class TestToleranceRespected:
    """
    Tests that tolerance parameter is respected to prevent data leakage.

    IMPORTANT: In safe merge operations, tolerance should NOT be used
    because it can allow data from slightly after the target time.
    """

    def test_tolerance_respected(self, intraday_ohlcv):
        """
        Verify that merge respects tolerance when specified.

        Note: Our safe_merge_macro intentionally does NOT use tolerance
        to prevent data leakage. This test shows what happens WITH tolerance.
        """
        # Create macro data 10 minutes after OHLCV times
        macro_offset = pd.DataFrame({
            "datetime": pd.to_datetime(["2025-01-10 13:10"]),  # 10 min after start
            "indicator": [100.0],
        })

        # With short tolerance (5 min), no match should occur for 13:00 row
        result_short_tol = pd.merge_asof(
            intraday_ohlcv[intraday_ohlcv["datetime"] == pd.Timestamp("2025-01-10 13:00")],
            macro_offset,
            on="datetime",
            direction="backward",
            tolerance=pd.Timedelta("5min")
        )

        # Should be NaN because macro is 10 min later and tolerance is 5 min
        assert pd.isna(result_short_tol["indicator"].iloc[0]), \
            "Tolerance should prevent match when data is too far in past"

    def test_no_tolerance_allows_old_data(self, intraday_ohlcv):
        """
        Verify that without tolerance, old data is used (as intended).

        This is the correct behavior for macro data that updates infrequently.
        """
        # Create macro data from a week ago
        old_macro = pd.DataFrame({
            "datetime": pd.to_datetime(["2025-01-03"]),  # Week before OHLCV
            "indicator": [100.0],
        })

        result = pd.merge_asof(
            intraday_ohlcv.sort_values("datetime"),
            old_macro.sort_values("datetime"),
            on="datetime",
            direction="backward"
            # No tolerance - will use old data
        )

        # All rows should have the indicator value
        assert result["indicator"].notna().all(), \
            "Without tolerance, old data should be used"
        assert (result["indicator"] == 100.0).all()

    def test_tolerance_prevents_stale_data(self, intraday_ohlcv):
        """
        Demonstrate how tolerance can prevent using stale data.

        When macro data is too old, it might be better to have NaN
        than outdated values.
        """
        # Create macro data from a month ago
        very_old_macro = pd.DataFrame({
            "datetime": pd.to_datetime(["2024-12-01"]),  # Month before
            "indicator": [100.0],
        })

        # With 7-day tolerance, should not match
        result_with_tol = pd.merge_asof(
            intraday_ohlcv.sort_values("datetime"),
            very_old_macro.sort_values("datetime"),
            on="datetime",
            direction="backward",
            tolerance=pd.Timedelta("7D")
        )

        # Should be NaN because data is older than tolerance
        assert result_with_tol["indicator"].isna().all(), \
            "Tolerance should prevent matching very old data"


# =============================================================================
# Test Safe Merge Integration
# =============================================================================

@pytest.mark.skipif(not SAFE_MERGE_AVAILABLE, reason="safe_merge module not available")
class TestSafeMergeIntegration:
    """
    Integration tests with the safe_merge module.

    These tests verify that our production safe_merge_macro function
    properly prevents lookahead bias.
    """

    def test_safe_merge_macro_no_lookahead(self, intraday_ohlcv, daily_macro):
        """Verify safe_merge_macro prevents lookahead bias."""
        result = safe_merge_macro(
            intraday_ohlcv,
            daily_macro,
            datetime_col="datetime",
            track_source=True
        )

        # Verify tracking column exists
        assert "macro_source_date" in result.columns

        # Verify no future data
        for idx, row in result.iterrows():
            target_date = pd.to_datetime(row["datetime"])
            source_date = pd.to_datetime(row["macro_source_date"])

            if pd.notna(source_date):
                assert source_date <= target_date, \
                    f"Lookahead in safe_merge_macro: source {source_date} > target {target_date}"

    def test_validate_no_future_data_function(self):
        """Test the validate_no_future_data helper function."""
        # Valid data (source always before target)
        valid_df = pd.DataFrame({
            "datetime": pd.to_datetime(["2025-01-10", "2025-01-11"]),
            "macro_source_date": pd.to_datetime(["2025-01-09", "2025-01-10"]),
        })

        # Should not raise
        result = validate_no_future_data(valid_df)
        assert result is True

    def test_validate_no_future_data_catches_leakage(self):
        """Test that validate_no_future_data catches data leakage."""
        # Invalid data (source after target - leakage!)
        invalid_df = pd.DataFrame({
            "datetime": pd.to_datetime(["2025-01-10", "2025-01-11"]),
            "macro_source_date": pd.to_datetime(["2025-01-11", "2025-01-10"]),  # First is future!
        })

        with pytest.raises(ValueError, match="DATA LEAKAGE"):
            validate_no_future_data(invalid_df)


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """
    Edge case tests for temporal joins.
    """

    def test_empty_macro_data(self, intraday_ohlcv):
        """Verify behavior with empty macro data."""
        empty_macro = pd.DataFrame({
            "datetime": pd.Series([], dtype="datetime64[ns]"),
            "indicator": pd.Series([], dtype="float64"),
        })

        result = pd.merge_asof(
            intraday_ohlcv.sort_values("datetime"),
            empty_macro.sort_values("datetime"),
            on="datetime",
            direction="backward"
        )

        # All indicator values should be NaN
        assert result["indicator"].isna().all()

    def test_single_macro_point(self, intraday_ohlcv):
        """Verify behavior with single macro data point."""
        single_macro = pd.DataFrame({
            "datetime": pd.to_datetime(["2025-01-08"]),
            "indicator": [100.0],
        })

        result = pd.merge_asof(
            intraday_ohlcv.sort_values("datetime"),
            single_macro.sort_values("datetime"),
            on="datetime",
            direction="backward"
        )

        # All rows should use the single macro value
        assert (result["indicator"] == 100.0).all()

    def test_macro_only_after_ohlcv(self, intraday_ohlcv):
        """Verify behavior when all macro data is after OHLCV data."""
        future_macro = pd.DataFrame({
            "datetime": pd.to_datetime(["2025-01-15", "2025-01-16"]),
            "indicator": [100.0, 101.0],
        })

        result = pd.merge_asof(
            intraday_ohlcv.sort_values("datetime"),
            future_macro.sort_values("datetime"),
            on="datetime",
            direction="backward"
        )

        # All indicator values should be NaN (no past data available)
        assert result["indicator"].isna().all()

    def test_unsorted_input_handling(self, intraday_ohlcv, daily_macro):
        """Verify behavior with unsorted input data."""
        # Shuffle both DataFrames
        shuffled_ohlcv = intraday_ohlcv.sample(frac=1, random_state=42)
        shuffled_macro = daily_macro.sample(frac=1, random_state=42)

        # Merge with sorting
        result = pd.merge_asof(
            shuffled_ohlcv.sort_values("datetime"),
            shuffled_macro.sort_values("datetime"),
            on="datetime",
            direction="backward"
        )

        # Verify no lookahead bias
        for idx, row in result.iterrows():
            if pd.notna(row["dxy"]):
                matched_macro = daily_macro[daily_macro["dxy"] == row["dxy"]]
                macro_date = pd.to_datetime(matched_macro["datetime"].iloc[0]).date()
                row_date = pd.to_datetime(row["datetime"]).date()
                assert macro_date <= row_date


# =============================================================================
# Test Performance
# =============================================================================

class TestPerformance:
    """
    Performance tests for temporal join operations.
    """

    def test_large_dataset_performance(self):
        """Verify reasonable performance with large datasets."""
        import time

        # Create large datasets
        n_ohlcv = 100000  # ~1 year of 5-min bars
        n_macro = 365  # 1 year of daily data

        ohlcv = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=n_ohlcv, freq="5min"),
            "close": np.random.randn(n_ohlcv),
        })

        macro = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=n_macro, freq="D"),
            "indicator": np.random.randn(n_macro),
        })

        # Time the merge
        start = time.time()
        result = pd.merge_asof(
            ohlcv.sort_values("datetime"),
            macro.sort_values("datetime"),
            on="datetime",
            direction="backward"
        )
        elapsed = time.time() - start

        # Should complete in under 5 seconds (usually < 1 second)
        assert elapsed < 5.0, f"Merge took {elapsed:.2f}s - too slow"
        assert len(result) == n_ohlcv


# =============================================================================
# Main Entry Point
# =============================================================================

def run_tests():
    """Run all temporal join tests."""
    print("=" * 60)
    print("Temporal Joins Unit Tests (P0-08)")
    print("=" * 60)

    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x",
    ])

    return exit_code


if __name__ == "__main__":
    sys.exit(run_tests())
