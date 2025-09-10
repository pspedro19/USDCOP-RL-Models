#!/usr/bin/env python3
"""
Verification script to ensure all critical fixes are in place
Run this before re-executing L4 and L5 pipelines
"""

import os
import sys
import re
import json
from pathlib import Path

def check_file_contains(filepath, pattern, description):
    """Check if file contains a specific pattern"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            if re.search(pattern, content):
                print(f"[OK] {description}")
                return True
            else:
                print(f"[FAIL] {description} - NOT FOUND!")
                return False
    except Exception as e:
        print(f"[ERROR] Error checking {filepath}: {e}")
        return False

def main():
    print("=" * 80)
    print("CRITICAL FIXES VERIFICATION SCRIPT")
    print("=" * 80)
    
    base_path = Path("C:/Users/pedro/OneDrive/Documents/ALGO TRADING/USDCOP/USDCOP_Trading_RL")
    l4_dag = base_path / "airflow/dags/usdcop_m5__05_l4_rlready.py"
    l5_dag = base_path / "airflow/dags/usdcop_m5__06_l5_serving.py"
    
    all_checks_passed = True
    
    print("\nL4 CRITICAL CHECKS:")
    print("-" * 40)
    
    # Check 1: REWARD_SPEC window
    if not check_file_contains(
        l4_dag,
        r"REWARD_SPEC\s*=\s*\{[^}]*'window'\s*:\s*\[12,\s*24\]",
        "REWARD_SPEC window is [12, 24]"
    ):
        all_checks_passed = False
        print("  WARNING: This is THE MOST CRITICAL fix - without it, nothing works!")
    
    # Check 2: forward_window in reward_spec dict
    if not check_file_contains(
        l4_dag,
        r"'forward_window'\s*:\s*\[12,\s*24\]",
        "forward_window in reward_spec dict is [12, 24]"
    ):
        all_checks_passed = False
    
    # Check 3: price_type is mid
    if not check_file_contains(
        l4_dag,
        r"'price_type'\s*:\s*'mid'",
        "price_type is 'mid' (not OHLC4)"
    ):
        all_checks_passed = False
    
    # Check 4: spread bounds reduced
    if not check_file_contains(
        l4_dag,
        r"'spread_bps_p95_bounds'\s*:\s*\[2,\s*15\]",
        "spread_bps_p95_bounds is [2, 15] (not [2, 25])"
    ):
        all_checks_passed = False
    
    # Check 5: Multi-scale features function exists
    if not check_file_contains(
        l4_dag,
        r"def add_multiscale_features",
        "Multi-scale features function exists"
    ):
        all_checks_passed = False
    
    # Check 6: Multi-scale features called
    if not check_file_contains(
        l4_dag,
        r"df = add_multiscale_features\(df\)",
        "Multi-scale features function is called"
    ):
        all_checks_passed = False
    
    # Check 7: mid_t shifts use REWARD_SPEC
    if not check_file_contains(
        l4_dag,
        r"shift\(-REWARD_SPEC\['window'\]\[0\]\)",
        "mid_t1 uses REWARD_SPEC['window'][0] shift"
    ):
        all_checks_passed = False
    
    if not check_file_contains(
        l4_dag,
        r"shift\(-REWARD_SPEC\['window'\]\[1\]\)",
        "mid_t2 uses REWARD_SPEC['window'][1] shift"
    ):
        all_checks_passed = False
    
    print("\nL5 CONFIGURATION CHECKS:")
    print("-" * 40)
    
    # Check 8: Total timesteps
    if not check_file_contains(
        l5_dag,
        r"L5_TOTAL_TIMESTEPS.*default_var=\"1000000\"",
        "L5_TOTAL_TIMESTEPS default is 1000000"
    ):
        all_checks_passed = False
    
    # Check 9: Entropy coefficient
    if not check_file_contains(
        l5_dag,
        r"L5_ENT_COEF.*default_var=\"0\.03\"",
        "L5_ENT_COEF default is 0.03"
    ):
        all_checks_passed = False
    
    # Check 10: Optimizer kwargs
    if not check_file_contains(
        l5_dag,
        r"\"betas\":\s*\(l5_adam_beta1,\s*l5_adam_beta2\)",
        "Adam optimizer uses beta variables"
    ):
        all_checks_passed = False
    
    # Check 11: Action mapping fix
    if not check_file_contains(
        l5_dag,
        r"sell_ratio = action_counts\.get\(0, 0\)",
        "Action mapping: 0=Sell (fixed)"
    ):
        all_checks_passed = False
    
    # Check 12: MLflow artifact path fix
    if not check_file_contains(
        l5_dag,
        r"artifact_path=f\"models/\{model_filename\}\"",
        "MLflow downloads from 'models/' path"
    ):
        all_checks_passed = False
    
    print("\n" + "=" * 80)
    
    if all_checks_passed:
        print("[SUCCESS] ALL CRITICAL FIXES ARE IN PLACE!")
        print("\nYou can now proceed with:")
        print("1. Re-run L4: airflow dags trigger usdcop_m5__05_l4_rlready")
        print("2. After L4 completes, re-run L5: airflow dags trigger usdcop_m5__06_l5_serving")
        print("\nExpected results:")
        print("- Sanity check will be positive (~0.002)")
        print("- No more 100% HOLD policy")
        print("- Sortino > 0 (targeting > 1.3)")
        return 0
    else:
        print("[FAILURE] SOME CRITICAL FIXES ARE MISSING!")
        print("\nDO NOT proceed with pipeline execution until all checks pass.")
        print("Review the failed checks above and apply the necessary fixes.")
        return 1

if __name__ == "__main__":
    sys.exit(main())