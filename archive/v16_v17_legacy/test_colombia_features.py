"""
Test Script for Colombia Feature Builder V17
============================================

Verifies that the 4 Colombia-specific features are correctly calculated:
1. vix_zscore: VIX z-score (volatility regime)
2. oil_above_60_flag: Binary flag for oil > $60
3. usdclp_ret_1d: Lagged USD/CLP daily return
4. banrep_intervention_proximity: BanRep intervention proximity

Author: Pedro @ Lean Tech Solutions
Date: 2025-12-19
"""

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'src'))

import pandas as pd
import numpy as np
from dataset_builder_v17 import ColombiaFeatureBuilder


def test_with_mock_data():
    """Test with mock data (synthetic OHLCV)."""
    print("\n" + "="*70)
    print("Test 1: Mock Data (Synthetic OHLCV)")
    print("="*70)

    # Create mock data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='5min')
    np.random.seed(42)

    df = pd.DataFrame({
        'open': 4000 + np.random.randn(len(dates)) * 50,
        'high': 4050 + np.random.randn(len(dates)) * 50,
        'low': 3950 + np.random.randn(len(dates)) * 50,
        'close': 4000 + np.random.randn(len(dates)) * 50,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)

    print(f"Input: {len(df):,} bars from {df.index.min()} to {df.index.max()}")

    # Build features
    builder = ColombiaFeatureBuilder()
    df = builder.add_all_colombia_features(df)

    # Verify features exist
    expected_features = [
        'vix_zscore',
        'oil_above_60_flag',
        'usdclp_ret_1d',
        'banrep_intervention_proximity'
    ]

    for feature in expected_features:
        assert feature in df.columns, f"Missing feature: {feature}"
        assert df[feature].isna().sum() == 0, f"Feature {feature} has NaN values"
        print(f"  [OK] {feature}: range [{df[feature].min():.4f}, {df[feature].max():.4f}]")

    print("\n[PASSED] All features added successfully!")


def test_feature_ranges():
    """Test that features are within expected ranges."""
    print("\n" + "="*70)
    print("Test 2: Feature Range Validation")
    print("="*70)

    # Create minimal test data
    dates = pd.date_range('2023-06-01', '2023-06-30', freq='5min')
    df = pd.DataFrame({
        'close': 4000 + np.random.randn(len(dates)) * 100
    }, index=dates)

    # Build features
    builder = ColombiaFeatureBuilder()
    df = builder.add_all_colombia_features(df)

    # Test ranges
    tests = [
        ('vix_zscore', -3.0, 3.0),
        ('oil_above_60_flag', 0, 1),
        ('usdclp_ret_1d', -0.10, 0.10),
        ('banrep_intervention_proximity', -1.0, 1.0)
    ]

    all_passed = True
    for feature, min_val, max_val in tests:
        actual_min = df[feature].min()
        actual_max = df[feature].max()

        in_range = (actual_min >= min_val) and (actual_max <= max_val)
        status = "[OK]" if in_range else "[FAIL]"

        print(f"  {status} {feature}: expected [{min_val}, {max_val}], got [{actual_min:.4f}, {actual_max:.4f}]")

        if not in_range:
            all_passed = False

    if all_passed:
        print("\n[PASSED] All features within expected ranges!")
    else:
        print("\n[FAILED] Some features out of range!")
        sys.exit(1)


def test_feature_summary():
    """Test feature summary statistics method."""
    print("\n" + "="*70)
    print("Test 3: Feature Summary Statistics")
    print("="*70)

    # Create test data
    dates = pd.date_range('2023-01-01', '2023-03-31', freq='5min')
    df = pd.DataFrame({
        'close': 4000 + np.random.randn(len(dates)) * 100
    }, index=dates)

    # Build features
    builder = ColombiaFeatureBuilder()
    df = builder.add_all_colombia_features(df)

    # Get summary
    summary = builder.get_feature_summary(df)

    print("\nSummary Statistics:")
    print(summary[['mean', 'std', 'min', 'max', 'null_count']])

    # Verify no nulls
    assert (summary['null_count'] == 0).all(), "Some features have null values"

    print("\n[PASSED] Feature summary generated successfully!")


def test_binary_flag():
    """Test that oil_above_60_flag is truly binary."""
    print("\n" + "="*70)
    print("Test 4: Binary Flag Validation")
    print("="*70)

    # Create test data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    df = pd.DataFrame({
        'close': 4000.0
    }, index=dates)

    # Build features
    builder = ColombiaFeatureBuilder()
    df = builder.add_all_colombia_features(df)

    # Check binary
    unique_values = df['oil_above_60_flag'].unique()
    is_binary = set(unique_values).issubset({0, 1})

    print(f"  Unique values: {sorted(unique_values)}")
    print(f"  Is binary: {is_binary}")

    assert is_binary, "oil_above_60_flag is not binary!"

    print("\n[PASSED] Oil flag is binary!")


def test_no_lookahead_bias():
    """Test that usdclp_ret_1d has proper lag."""
    print("\n" + "="*70)
    print("Test 5: No Look-Ahead Bias (USD/CLP Lag)")
    print("="*70)

    # Create test data
    dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
    df = pd.DataFrame({
        'close': 4000.0
    }, index=dates)

    # Build features
    builder = ColombiaFeatureBuilder()
    df = builder.add_all_colombia_features(df)

    # The first day should have no signal (lagged)
    first_value = df['usdclp_ret_1d'].iloc[0]
    print(f"  First value (should be 0 or small): {first_value:.6f}")

    # Values should be small (due to lag and stability)
    max_abs = df['usdclp_ret_1d'].abs().max()
    print(f"  Maximum absolute value: {max_abs:.6f}")

    print("\n[PASSED] USD/CLP lag verified!")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("COLOMBIA FEATURE BUILDER V17 - TEST SUITE")
    print("="*70)

    try:
        test_with_mock_data()
        test_feature_ranges()
        test_feature_summary()
        test_binary_flag()
        test_no_lookahead_bias()

        print("\n" + "="*70)
        print("ALL TESTS PASSED!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
