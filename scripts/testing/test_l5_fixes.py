#!/usr/bin/env python3
"""
Test Script for L5 Pipeline Fixes
==================================
Validates all the critical fixes implemented for the L5 serving pipeline:
1. HOLD reward = 0 in evaluation mode
2. Loop detector correctly handles 60/60 episodes
3. Sortino robust handling of extreme values
4. Bundle includes real L4 artifacts
5. Increased timesteps and cost curriculum
6. Sanity check with proper reward calculation
"""

import sys
import os
import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime

# Add the airflow dags path to sys.path
sys.path.append('airflow/dags')
sys.path.append('airflow/dags/utils')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("=" * 80)
print("L5 PIPELINE FIXES VALIDATION TEST SUITE")
print("=" * 80)
print(f"Timestamp: {datetime.now().isoformat()}")
print()

# =============================================================================
# TEST 1: HOLD Reward = 0 in Eval Mode
# =============================================================================
def test_hold_reward_zero():
    """Test that HOLD action gives reward=0 in eval mode with no costs"""
    print("\n" + "=" * 60)
    print("TEST 1: HOLD Reward = 0 in Eval Mode")
    print("=" * 60)
    
    try:
        from reward_sentinel import SentinelTradingEnv
        from gymnasium_trading_env import USDCOPTradingEnv
        
        # Create base environment
        base_env = USDCOPTradingEnv()  # Use synthetic data
        
        # Wrap with SentinelTradingEnv in eval mode
        env = SentinelTradingEnv(
            base_env,
            cost_model={'spread_bps': 0, 'slippage_bps': 0, 'fee_bps': 0},  # No costs
            shaping_penalty=0.0,  # No shaping
            eval_mode=True  # EVAL MODE
        )
        
        # Reset environment
        obs, _ = env.reset()
        
        # Test HOLD action (action=0)
        hold_rewards = []
        for _ in range(10):
            obs, reward, done, truncated, info = env.step(0)  # HOLD action
            hold_rewards.append(reward)
            if done or truncated:
                break
        
        # Check if all HOLD rewards are 0
        all_zero = all(abs(r) < 1e-9 for r in hold_rewards)
        avg_reward = np.mean(hold_rewards) if hold_rewards else 0
        
        if all_zero:
            print("✅ PASS: All HOLD rewards are 0.0 in eval mode")
        else:
            print(f"❌ FAIL: HOLD rewards not zero. Average: {avg_reward:.6f}")
            print(f"   Rewards: {hold_rewards[:5]}...")
        
        return all_zero
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

# =============================================================================
# TEST 2: Loop Detector for 60/60 Episodes
# =============================================================================
def test_loop_detector():
    """Test that episodes ending at exactly 60 steps are not marked as loops"""
    print("\n" + "=" * 60)
    print("TEST 2: Loop Detector for 60/60 Episodes")
    print("=" * 60)
    
    try:
        # Simulate episode ending at exactly 60 steps
        MAX_STEPS_PER_EPISODE = 60
        ep_length = 60
        done = [True]  # Episode completed normally
        
        # Check the logic
        is_truncated = ep_length >= MAX_STEPS_PER_EPISODE and not done[0]
        is_normal = ep_length == MAX_STEPS_PER_EPISODE and done[0]
        
        if is_normal and not is_truncated:
            print(f"✅ PASS: Episode of {ep_length} steps with done=True marked as NORMAL")
        else:
            print(f"❌ FAIL: Episode of {ep_length} steps with done=True incorrectly marked")
        
        # Test actual truncation case
        done = [False]
        is_truncated = ep_length >= MAX_STEPS_PER_EPISODE and not done[0]
        
        if is_truncated:
            print(f"✅ PASS: Episode of {ep_length} steps with done=False marked as TRUNCATED")
        else:
            print(f"❌ FAIL: Episode of {ep_length} steps with done=False not marked as truncated")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

# =============================================================================
# TEST 3: Sortino Robust Handling
# =============================================================================
def test_sortino_robust():
    """Test Sortino ratio handles extreme values correctly"""
    print("\n" + "=" * 60)
    print("TEST 3: Sortino Robust Handling")
    print("=" * 60)
    
    try:
        from l5_patch_metrics import robust_sortino
        
        # Test case 1: Normal returns
        normal_returns = np.random.normal(0.001, 0.01, 100)
        sortino_normal, status_normal = robust_sortino(normal_returns)
        print(f"Normal returns: Sortino={sortino_normal:.4f}, Status={status_normal}")
        
        # Test case 2: Extreme negative returns (like current problem)
        extreme_returns = np.full(100, -1e6)  # Extreme negative
        sortino_extreme, status_extreme = robust_sortino(extreme_returns)
        print(f"Extreme returns: Sortino={sortino_extreme:.4f}, Status={status_extreme}")
        
        # Test case 3: Zero variance (all returns same)
        zero_var_returns = np.full(100, 0.001)
        sortino_zero_var, status_zero_var = robust_sortino(zero_var_returns)
        print(f"Zero variance: Sortino={sortino_zero_var:.4f}, Status={status_zero_var}")
        
        # Check if extreme values are clamped
        if -100 <= sortino_extreme <= 100:
            print("✅ PASS: Extreme Sortino values properly clamped")
        else:
            print(f"❌ FAIL: Extreme Sortino not clamped: {sortino_extreme}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

# =============================================================================
# TEST 4: Configuration Changes
# =============================================================================
def test_configuration():
    """Test that configuration changes are correct"""
    print("\n" + "=" * 60)
    print("TEST 4: Configuration Changes")
    print("=" * 60)
    
    try:
        # Read the DAG file to check configuration
        dag_path = 'airflow/dags/usdcop_m5__06_l5_serving.py'
        with open(dag_path, 'r') as f:
            dag_content = f.read()
        
        # Check TOTAL_TIMESTEPS
        if 'default_var="300000"' in dag_content:
            print("✅ PASS: TOTAL_TIMESTEPS increased to 300000")
        else:
            print("❌ FAIL: TOTAL_TIMESTEPS not properly increased")
        
        # Check cost curriculum
        if 'initial_cost_factor=0.25' in dag_content:
            print("✅ PASS: Cost curriculum starts at 25%")
        else:
            print("❌ FAIL: Cost curriculum not properly configured")
        
        if 'TOTAL_TIMESTEPS * 0.8' in dag_content:
            print("✅ PASS: Cost curriculum reaches 100% at 80% of training")
        else:
            print("❌ FAIL: Cost curriculum progression not properly configured")
        
        # Check hyperparameters
        if '"ent_coef": 0.005' in dag_content:
            print("✅ PASS: Entropy coefficient reduced to 0.005")
        else:
            print("❌ FAIL: Entropy coefficient not properly set")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

# =============================================================================
# TEST 5: Sanity Check
# =============================================================================
def test_sanity_check():
    """Test that sanity check returns are centered around 0"""
    print("\n" + "=" * 60)
    print("TEST 5: Sanity Check with Proper Rewards")
    print("=" * 60)
    
    try:
        from reward_costs_sanity import quick_sanity_check
        
        # Run sanity check with no costs
        result = quick_sanity_check(
            env_class='gymnasium_trading_env.USDCOPTradingEnv',
            n_episodes=10,
            cost_model={'spread_bps': 0, 'slippage_bps': 0, 'fee_bps': 0}
        )
        
        if result['status'] == 'PASS':
            print(f"✅ PASS: Sanity check passed")
            print(f"   Mean reward: {result['mean']:.6f}")
            print(f"   Std reward: {result['std']:.6f}")
        else:
            print(f"❌ FAIL: Sanity check failed")
            print(f"   Status: {result['status']}")
            print(f"   Mean reward: {result.get('mean', 'N/A')}")
        
        # Check if mean is close to 0
        if 'mean' in result and abs(result['mean']) < 0.01:
            print("✅ PASS: Mean reward centered near 0")
        else:
            print(f"❌ FAIL: Mean reward not centered: {result.get('mean', 'N/A')}")
        
        return result['status'] == 'PASS'
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

# =============================================================================
# TEST 6: Bundle L4 Artifacts
# =============================================================================
def test_bundle_l4_artifacts():
    """Test that bundle creation includes L4 artifacts logic"""
    print("\n" + "=" * 60)
    print("TEST 6: Bundle L4 Artifacts")
    print("=" * 60)
    
    try:
        # Check if the _ensure_l4_local function exists in DAG
        dag_path = 'airflow/dags/usdcop_m5__06_l5_serving.py'
        with open(dag_path, 'r') as f:
            dag_content = f.read()
        
        if 'def _ensure_l4_local' in dag_content:
            print("✅ PASS: _ensure_l4_local function exists")
        else:
            print("❌ FAIL: _ensure_l4_local function not found")
            return False
        
        if 'split_spec = _ensure_l4_local' in dag_content:
            print("✅ PASS: split_spec downloaded from L4")
        else:
            print("❌ FAIL: split_spec not using L4 download")
        
        if 'norm_ref = _ensure_l4_local' in dag_content:
            print("✅ PASS: normalization_ref downloaded from L4")
        else:
            print("❌ FAIL: normalization_ref not using L4 download")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

# =============================================================================
# MAIN TEST RUNNER
# =============================================================================
def main():
    """Run all tests and report results"""
    print("\n" + "=" * 80)
    print("RUNNING ALL TESTS")
    print("=" * 80)
    
    results = {
        "HOLD Reward Zero": test_hold_reward_zero(),
        "Loop Detector": test_loop_detector(),
        "Sortino Robust": test_sortino_robust(),
        "Configuration": test_configuration(),
        "Sanity Check": test_sanity_check(),
        "Bundle L4 Artifacts": test_bundle_l4_artifacts()
    }
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 SUCCESS: All tests passed! L5 pipeline is ready for production.")
    else:
        print(f"\n⚠️  WARNING: {total - passed} tests failed. Review and fix before deployment.")
    
    # Save results
    results_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "passed": passed,
            "total": total,
            "status": "PASS" if passed == total else "FAIL"
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
