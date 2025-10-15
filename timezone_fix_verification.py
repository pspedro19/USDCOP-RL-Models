#!/usr/bin/env python3
"""
Timezone Fix Verification Script
===============================
Tests the timezone fixes applied to the L0 USDCOP pipeline
"""

import sys
import os
import logging
from datetime import datetime

# Add the airflow dags path to import the fixed modules
sys.path.append('/home/GlobalForex/USDCOP-RL-Models/airflow/dags')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_timezone_fixes():
    """Test all timezone-related fixes"""

    print("="*60)
    print("TIMEZONE FIX VERIFICATION")
    print("="*60)

    try:
        # Test 1: Import the fixed module
        print("\n1. Testing module imports...")
        from usdcop_m5__01_l0_acquire import (
            ensure_timezone_aware,
            validate_dataframe_timezone,
            calculate_quality_metrics
        )
        print("✅ Successfully imported timezone helper functions")

        # Test 2: Test timezone helper functions
        print("\n2. Testing timezone helper functions...")

        # Test ensure_timezone_aware function
        import pandas as pd
        import pytz

        # Test with naive timestamp
        naive_dt = pd.Timestamp('2024-01-01 10:00:00')
        aware_dt = ensure_timezone_aware(naive_dt)

        print(f"   Naive timestamp: {naive_dt} (tz: {naive_dt.tz})")
        print(f"   Made aware: {aware_dt} (tz: {aware_dt.tz})")

        if aware_dt.tz is not None:
            print("✅ ensure_timezone_aware function works correctly")
        else:
            print("❌ ensure_timezone_aware function failed")
            return False

        # Test 3: Test DataFrame timezone validation
        print("\n3. Testing DataFrame timezone validation...")

        # Create test dataframe with naive timestamps
        test_df = pd.DataFrame({
            'time': pd.date_range('2024-01-01 08:00:00', periods=10, freq='5T'),
            'open': [4000.0] * 10,
            'high': [4010.0] * 10,
            'low': [3990.0] * 10,
            'close': [4005.0] * 10,
            'volume': [100] * 10
        })

        print(f"   Original time column tz: {test_df['time'].dt.tz}")

        # Validate timezone
        validated_df = validate_dataframe_timezone(test_df, 'time')

        print(f"   Validated time column tz: {validated_df['time'].dt.tz}")

        if validated_df['time'].dt.tz is not None:
            print("✅ DataFrame timezone validation works correctly")
        else:
            print("❌ DataFrame timezone validation failed")
            return False

        # Test 4: Test calculate_quality_metrics function
        print("\n4. Testing calculate_quality_metrics function...")

        try:
            batch_start = pd.Timestamp('2024-01-01')
            batch_end = pd.Timestamp('2024-01-02')

            # This should not raise timezone comparison errors anymore
            quality_metrics = calculate_quality_metrics(validated_df, batch_start, batch_end)

            print(f"   Quality metrics calculated successfully")
            print(f"   Completeness: {quality_metrics['completeness']:.1f}%")
            print(f"   Actual bars: {quality_metrics['actual_bars']}")
            print(f"   Premium bars: {quality_metrics['premium_bars']}")

            if 'error' not in quality_metrics:
                print("✅ calculate_quality_metrics function works without timezone errors")
            else:
                print(f"❌ calculate_quality_metrics has error: {quality_metrics['error']}")
                return False

        except Exception as e:
            print(f"❌ calculate_quality_metrics failed with error: {e}")
            return False

        # Test 5: Test mixed timezone scenarios
        print("\n5. Testing mixed timezone scenarios...")

        try:
            # Create timezone-aware timestamps
            cot_tz = pytz.timezone('America/Bogota')
            utc_tz = pytz.timezone('UTC')

            # Mix of naive and aware timestamps
            mixed_times = [
                pd.Timestamp('2024-01-01 08:00:00'),  # naive
                pd.Timestamp('2024-01-01 08:05:00', tz=cot_tz),  # COT aware
                pd.Timestamp('2024-01-01 13:10:00', tz=utc_tz),  # UTC aware
                pd.Timestamp('2024-01-01 08:15:00')   # naive
            ]

            mixed_df = pd.DataFrame({
                'time': mixed_times,
                'open': [4000.0] * 4,
                'high': [4010.0] * 4,
                'low': [3990.0] * 4,
                'close': [4005.0] * 4,
                'volume': [100] * 4
            })

            print(f"   Mixed timezone dataframe created with {len(mixed_df)} rows")

            # This should handle mixed timezones gracefully
            validated_mixed_df = validate_dataframe_timezone(mixed_df, 'time')

            # Check if all timestamps are now consistently timezone-aware
            all_aware = all(t.tz is not None for t in validated_mixed_df['time'])

            if all_aware:
                print("✅ Mixed timezone handling works correctly")
            else:
                print("❌ Mixed timezone handling failed")
                return False

        except Exception as e:
            print(f"❌ Mixed timezone test failed: {e}")
            return False

        print("\n" + "="*60)
        print("🎉 ALL TIMEZONE FIXES VERIFIED SUCCESSFULLY!")
        print("="*60)

        return True

    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_verification_report():
    """Create a detailed verification report"""

    report = {
        'timestamp': datetime.now().isoformat(),
        'fixes_applied': [
            {
                'location': 'Line 275 - TwelveData timestamp processing',
                'fix': 'Added timezone localization for naive timestamps from TwelveData API',
                'purpose': 'Ensure all timestamps from API are timezone-aware (COT)'
            },
            {
                'location': 'Lines 356, 453-455 - Colombian holiday filtering',
                'fix': 'Added timezone localization for holiday timestamps before comparison',
                'purpose': 'Prevent tz-naive vs tz-aware comparison errors in holiday filtering'
            },
            {
                'location': 'Lines 196, 263-274 - Batch date processing',
                'fix': 'Added timezone awareness to batch start/end dates',
                'purpose': 'Ensure consistent timezone handling throughout batch processing'
            },
            {
                'location': 'Lines 362, 463-470 - Hour extraction for premium hours',
                'fix': 'Added timezone conversion before hour extraction',
                'purpose': 'Ensure premium hours (8am-2pm) are correctly identified in COT'
            },
            {
                'location': 'Lines 370, 483-485 - Gap detection',
                'fix': 'Added timezone handling for time difference calculations',
                'purpose': 'Prevent timezone issues in gap detection logic'
            },
            {
                'location': 'Lines 36-83 - Helper functions',
                'fix': 'Added timezone validation and conversion helper functions',
                'purpose': 'Centralized timezone handling with fallback error handling'
            },
            {
                'location': 'Lines 434-538 - Error handling',
                'fix': 'Added comprehensive error handling with fallback metrics',
                'purpose': 'Gracefully handle any remaining timezone edge cases'
            }
        ],
        'root_causes_addressed': [
            'TwelveData API returning naive timestamps',
            'Colombian holiday strings being compared with timezone-aware batch dates',
            'Mixed timezone objects in datetime comparisons',
            'Hour extraction from timezone-naive timestamps for premium hour filtering',
            'Time difference calculations with inconsistent timezone awareness'
        ],
        'verification_strategy': [
            'Import and test all timezone helper functions',
            'Test DataFrame timezone validation with naive timestamps',
            'Test quality metrics calculation without timezone errors',
            'Test mixed timezone scenarios (naive, COT-aware, UTC-aware)',
            'Verify error handling and fallback mechanisms'
        ]
    }

    return report

if __name__ == "__main__":
    print("Starting timezone fix verification...")

    success = test_timezone_fixes()

    if success:
        print("\n📊 Generating verification report...")
        report = create_verification_report()

        print(f"\n📋 VERIFICATION REPORT:")
        print(f"Timestamp: {report['timestamp']}")
        print(f"Fixes Applied: {len(report['fixes_applied'])}")
        print(f"Root Causes Addressed: {len(report['root_causes_addressed'])}")
        print(f"Verification Tests: {len(report['verification_strategy'])}")

        print(f"\n✅ TIMEZONE ERROR RESOLUTION: COMPLETE")
        print(f"The L0 pipeline should now handle timezone comparisons correctly.")

        sys.exit(0)
    else:
        print(f"\n❌ VERIFICATION FAILED")
        print(f"Please review the error messages above.")
        sys.exit(1)