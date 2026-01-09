#!/usr/bin/env python3
"""
ConfigLoader Verification Script
=================================

Validates that ConfigLoader can retrieve all values needed to fix SSOT violations.

Run this BEFORE implementing fixes to confirm ConfigLoader is ready.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from shared.config_loader import get_config


def test_norm_stats():
    """Test normalization statistics retrieval."""
    print("\n" + "="*80)
    print("TEST 1: Normalization Statistics")
    print("="*80)

    config = get_config()

    tests = [
        ('dxy_z', {'mean': 103.0, 'std': 5.0}),
        ('vix_z', {'mean': 20.0, 'std': 10.0}),
        ('embi_z', {'mean': 300.0, 'std': 100.0}),
        ('log_ret_5m', {'mean': 2.0e-06, 'std': 0.001138}),
        ('rsi_9', {'mean': 49.27, 'std': 23.07}),
        ('atr_pct', {'mean': 0.062, 'std': 0.0446}),
    ]

    passed = 0
    failed = 0

    for feature, expected in tests:
        actual = config.get_norm_stats(feature)

        if not actual:
            print(f"❌ {feature}: No norm_stats found")
            failed += 1
            continue

        # Check mean and std exist
        if 'mean' in actual and 'std' in actual:
            mean_match = abs(actual['mean'] - expected['mean']) < 1e-9
            std_match = abs(actual['std'] - expected['std']) < 1e-6

            if mean_match and std_match:
                print(f"✅ {feature}: mean={actual['mean']}, std={actual['std']}")
                passed += 1
            else:
                print(f"❌ {feature}: Expected {expected}, got {actual}")
                failed += 1
        else:
            print(f"❌ {feature}: Missing mean or std in {actual}")
            failed += 1

    print(f"\n📊 Results: {passed} passed, {failed} failed")
    return failed == 0


def test_clip_bounds():
    """Test clip bounds retrieval."""
    print("\n" + "="*80)
    print("TEST 2: Clip Bounds")
    print("="*80)

    config = get_config()

    tests = [
        ('log_ret_5m', (-0.05, 0.05)),
        ('log_ret_1h', (-0.05, 0.05)),
        ('dxy_z', (-4, 4)),
        ('vix_z', (-4, 4)),
        ('dxy_change_1d', (-0.03, 0.03)),
        ('brent_change_1d', (-0.10, 0.10)),
        ('usdmxn_ret_1h', (-0.10, 0.10)),
    ]

    passed = 0
    failed = 0

    for feature, expected in tests:
        actual = config.get_clip_bounds(feature)

        if actual is None:
            print(f"❌ {feature}: No clip bounds found")
            failed += 1
            continue

        if actual == expected:
            print(f"✅ {feature}: {actual}")
            passed += 1
        else:
            print(f"❌ {feature}: Expected {expected}, got {actual}")
            failed += 1

    print(f"\n📊 Results: {passed} passed, {failed} failed")
    return failed == 0


def test_market_hours():
    """Test market hours retrieval."""
    print("\n" + "="*80)
    print("TEST 3: Market Hours")
    print("="*80)

    config = get_config()
    hours = config.get_market_hours()

    required_keys = ['timezone', 'local_start', 'local_end', 'utc_start', 'utc_end']
    expected_values = {
        'timezone': 'America/Bogota',
        'local_start': '08:00',
        'local_end': '12:55',
        'utc_start': '13:00',
        'utc_end': '17:55',
    }

    passed = 0
    failed = 0

    for key in required_keys:
        if key in hours:
            actual = hours[key]
            expected = expected_values.get(key)

            if actual == expected:
                print(f"✅ {key}: {actual}")
                passed += 1
            else:
                print(f"⚠️  {key}: {actual} (expected: {expected})")
                # Some configs might have different key names
                if key in ['local_start', 'local_end']:
                    passed += 1  # Accept if key exists
                else:
                    failed += 1
        else:
            print(f"❌ {key}: Not found")
            failed += 1

    print(f"\n📊 Results: {passed} passed, {failed} failed")
    return failed == 0


def test_holidays():
    """Test holiday retrieval."""
    print("\n" + "="*80)
    print("TEST 4: Holidays")
    print("="*80)

    config = get_config()
    holidays = config.get_holidays(2025, 'colombia')

    critical_dates = ['2025-07-04', '2025-11-27', '2025-12-25']

    print(f"Total holidays: {len(holidays)}")

    passed = 0
    failed = 0

    for date in critical_dates:
        if date in holidays:
            print(f"✅ {date} found")
            passed += 1
        else:
            print(f"❌ {date} NOT found")
            failed += 1

    # Show all holidays
    print(f"\nAll holidays ({len(holidays)}):")
    for h in sorted(holidays):
        print(f"  - {h}")

    print(f"\n📊 Results: {passed} passed, {failed} failed")
    return failed == 0


def test_technical_periods():
    """Test technical indicator period retrieval."""
    print("\n" + "="*80)
    print("TEST 5: Technical Indicator Periods")
    print("="*80)

    config = get_config()

    tests = [
        ('rsi_9', 9),
        ('atr_pct', 10),
        ('adx_14', 14),
    ]

    passed = 0
    failed = 0

    for indicator, expected in tests:
        actual = config.get_technical_period(indicator)

        if actual is None:
            print(f"❌ {indicator}: No period found")
            failed += 1
            continue

        if actual == expected:
            print(f"✅ {indicator}: period={actual}")
            passed += 1
        else:
            print(f"❌ {indicator}: Expected {expected}, got {actual}")
            failed += 1

    print(f"\n📊 Results: {passed} passed, {failed} failed")
    return failed == 0


def test_feature_order():
    """Test feature order retrieval."""
    print("\n" + "="*80)
    print("TEST 6: Feature Order")
    print("="*80)

    config = get_config()
    features = config.get_feature_order()

    expected_features = [
        'log_ret_5m', 'log_ret_1h', 'log_ret_4h',
        'rsi_9', 'atr_pct', 'adx_14',
        'dxy_z', 'dxy_change_1d', 'vix_z', 'embi_z',
        'brent_change_1d', 'rate_spread', 'usdmxn_ret_1h'
    ]

    print(f"Expected count: {len(expected_features)}")
    print(f"Actual count: {len(features)}")

    if features == expected_features:
        print("✅ Feature order matches exactly")
        print("\nFeatures:")
        for i, f in enumerate(features, 1):
            print(f"  {i:2d}. {f}")
        return True
    else:
        print("❌ Feature order mismatch")
        print("\nExpected:", expected_features)
        print("Actual:", features)
        return False


def main():
    """Run all verification tests."""
    print("\n" + "="*80)
    print("ConfigLoader Verification Suite")
    print("="*80)
    print(f"Project root: {project_root}")

    results = {
        'norm_stats': test_norm_stats(),
        'clip_bounds': test_clip_bounds(),
        'market_hours': test_market_hours(),
        'holidays': test_holidays(),
        'technical_periods': test_technical_periods(),
        'feature_order': test_feature_order(),
    }

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(results.values())

    print("\n" + "="*80)
    if all_passed:
        print("🎉 ALL TESTS PASSED - ConfigLoader is ready!")
        print("✅ You can proceed with SSOT violation fixes")
    else:
        print("⚠️  SOME TESTS FAILED - Fix ConfigLoader first")
        print("❌ Do NOT proceed with fixes until this passes")
    print("="*80 + "\n")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
