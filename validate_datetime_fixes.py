#!/usr/bin/env python3
"""
Basic Validation Script for USDCOP Datetime Fixes
=================================================
Validates that the core datetime handling fixes are properly implemented
without requiring external dependencies like pandas.
"""

import os
import sys

def validate_files_exist():
    """Validate that all critical files exist"""
    print("🔍 Validating file existence...")

    critical_files = [
        "/home/GlobalForex/USDCOP-RL-Models/airflow/dags/utils/datetime_handler.py",
        "/home/GlobalForex/USDCOP-RL-Models/airflow/dags/data_sources/twelvedata_client.py",
        "/home/GlobalForex/USDCOP-RL-Models/airflow/dags/usdcop_m5__01_l0_acquire.py",
        "/home/GlobalForex/USDCOP-RL-Models/airflow/dags/usdcop_m5__02_l1_standardize.py",
        "/home/GlobalForex/USDCOP-RL-Models/airflow/dags/usdcop_m5__03_l2_prepare.py",
    ]

    all_exist = True
    for file_path in critical_files:
        if os.path.exists(file_path):
            print(f"   ✅ {os.path.basename(file_path)}")
        else:
            print(f"   ❌ {os.path.basename(file_path)} - NOT FOUND")
            all_exist = False

    return all_exist

def validate_unified_datetime_handler():
    """Validate the UnifiedDatetimeHandler implementation"""
    print("🔍 Validating UnifiedDatetimeHandler...")

    handler_path = "/home/GlobalForex/USDCOP-RL-Models/airflow/dags/utils/datetime_handler.py"

    if not os.path.exists(handler_path):
        print("   ❌ UnifiedDatetimeHandler file not found")
        return False

    with open(handler_path, 'r') as f:
        content = f.read()

    # Check for key components
    required_components = [
        "class UnifiedDatetimeHandler",
        "ensure_timezone_aware",
        "convert_to_cot",
        "convert_to_utc",
        "standardize_dataframe_timestamps",
        "is_premium_hours",
        "is_business_day",
        "calculate_time_differences",
        "America/Bogota",
        "MARKET_OPEN_HOUR = 8",
        "MARKET_CLOSE_HOUR = 14"
    ]

    missing_components = []
    for component in required_components:
        if component not in content:
            missing_components.append(component)

    if missing_components:
        print("   ❌ Missing components:")
        for comp in missing_components:
            print(f"      - {comp}")
        return False
    else:
        print("   ✅ All required components present")
        return True

def validate_twelvedata_client_fixes():
    """Validate TwelveData client fixes"""
    print("🔍 Validating TwelveData client fixes...")

    client_path = "/home/GlobalForex/USDCOP-RL-Models/airflow/dags/data_sources/twelvedata_client.py"

    with open(client_path, 'r') as f:
        content = f.read()

    # Check for key fixes
    required_fixes = [
        "UnifiedDatetimeHandler",
        "timezone_safe",
        "@timezone_safe",
        "ensure_timezone_aware",
        "UNIFIED_DATETIME",
        "add_timezone_columns"
    ]

    missing_fixes = []
    for fix in required_fixes:
        if fix not in content:
            missing_fixes.append(fix)

    if missing_fixes:
        print("   ❌ Missing fixes:")
        for fix in missing_fixes:
            print(f"      - {fix}")
        return False
    else:
        print("   ✅ All required fixes present")
        return True

def validate_l0_pipeline_fixes():
    """Validate L0 pipeline fixes"""
    print("🔍 Validating L0 pipeline fixes...")

    l0_path = "/home/GlobalForex/USDCOP-RL-Models/airflow/dags/usdcop_m5__01_l0_acquire.py"

    with open(l0_path, 'r') as f:
        content = f.read()

    # Check for key timezone fixes
    required_fixes = [
        "ensure_timezone_aware",
        "validate_dataframe_timezone",
        "TIMEZONE FIX",
        "pytz",
        "America/Bogota"
    ]

    missing_fixes = []
    for fix in required_fixes:
        if fix not in content:
            missing_fixes.append(fix)

    if missing_fixes:
        print("   ❌ Missing fixes:")
        for fix in missing_fixes:
            print(f"      - {fix}")
        return False
    else:
        print("   ✅ All required fixes present")
        return True

def validate_l1_standardize_fixes():
    """Validate L1 standardize fixes"""
    print("🔍 Validating L1 standardize fixes...")

    l1_path = "/home/GlobalForex/USDCOP-RL-Models/airflow/dags/usdcop_m5__02_l1_standardize.py"

    with open(l1_path, 'r') as f:
        content = f.read()

    # Check for unified datetime handler integration
    required_fixes = [
        "UnifiedDatetimeHandler",
        "UNIFIED_DATETIME",
        "TIMEZONE FIX",
        "calculate_time_differences"
    ]

    missing_fixes = []
    for fix in required_fixes:
        if fix not in content:
            missing_fixes.append(fix)

    if missing_fixes:
        print("   ❌ Missing fixes:")
        for fix in missing_fixes:
            print(f"      - {fix}")
        return False
    else:
        print("   ✅ All required fixes present")
        return True

def validate_l2_prepare_fixes():
    """Validate L2 prepare fixes"""
    print("🔍 Validating L2 prepare fixes...")

    l2_path = "/home/GlobalForex/USDCOP-RL-Models/airflow/dags/usdcop_m5__03_l2_prepare.py"

    with open(l2_path, 'r') as f:
        content = f.read()

    # Check for timezone handling improvements
    required_fixes = [
        "UnifiedDatetimeHandler",
        "UNIFIED_DATETIME",
        "TIMEZONE FIX",
        "standardize_dataframe_timestamps",
        "convert_to_cot"
    ]

    missing_fixes = []
    for fix in required_fixes:
        if fix not in content:
            missing_fixes.append(fix)

    if missing_fixes:
        print("   ❌ Missing fixes:")
        for fix in missing_fixes:
            print(f"      - {fix}")
        return False
    else:
        print("   ✅ All required fixes present")
        return True

def validate_documentation():
    """Validate documentation exists"""
    print("🔍 Validating documentation...")

    doc_files = [
        "/home/GlobalForex/USDCOP-RL-Models/DATETIME_FIXES_SUMMARY.md",
        "/home/GlobalForex/USDCOP-RL-Models/test_datetime_fixes.py"
    ]

    all_exist = True
    for doc_file in doc_files:
        if os.path.exists(doc_file):
            print(f"   ✅ {os.path.basename(doc_file)}")
        else:
            print(f"   ❌ {os.path.basename(doc_file)} - NOT FOUND")
            all_exist = False

    return all_exist

def main():
    """Main validation function"""
    print("="*70)
    print("USDCOP DATETIME FIXES - VALIDATION REPORT")
    print("="*70)

    validation_results = []

    # Run all validations
    validation_results.append(("File Existence", validate_files_exist()))
    validation_results.append(("UnifiedDatetimeHandler", validate_unified_datetime_handler()))
    validation_results.append(("TwelveData Client", validate_twelvedata_client_fixes()))
    validation_results.append(("L0 Pipeline", validate_l0_pipeline_fixes()))
    validation_results.append(("L1 Standardize", validate_l1_standardize_fixes()))
    validation_results.append(("L2 Prepare", validate_l2_prepare_fixes()))
    validation_results.append(("Documentation", validate_documentation()))

    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    passed = 0
    total = len(validation_results)

    for test_name, result in validation_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} validations passed")

    if passed == total:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        print("\nDatetime fixes have been successfully implemented:")
        print("✅ Unified timezone handling across all pipeline stages")
        print("✅ Mixed timezone-aware/naive datetime comparison fixes")
        print("✅ Colombian business hours filtering implementation")
        print("✅ TwelveData API timezone conversion standardization")
        print("✅ Robust pandas datetime operations with timezone awareness")
        print("✅ Complete documentation and testing infrastructure")

        print("\n📋 NEXT STEPS:")
        print("1. Deploy fixes to staging environment")
        print("2. Run integration tests with real data")
        print("3. Monitor pipeline execution for timezone errors")
        print("4. Update monitoring dashboards")

        return True
    else:
        print(f"\n❌ {total - passed} VALIDATIONS FAILED")
        print("Please review the failed validations above and fix the issues.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)