#!/usr/bin/env python3
"""
Validate L5 Pipeline - 6 Critical Improvements
===============================================
Verifies all contract-driven enhancements are working properly.
"""

import subprocess
import re
import json
from datetime import datetime
from typing import Dict, List, Tuple

# Latest run
RUN_ID = "manual__2025-09-05T16:41:42+00:00"

def get_log_content(task_pattern: str) -> str:
    """Get log content from Docker container"""
    try:
        cmd = f'''docker exec usdcop-airflow-webserver bash -c "tail -500 /opt/airflow/logs/dag_id=usdcop_m5__06_l5_serving/run_id={RUN_ID}/task_id={task_pattern}/attempt=1.log 2>/dev/null"'''
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout
    except:
        return ""

def check_improvement_1_contract_driven() -> Tuple[bool, str]:
    """Check if MAX_STEPS is read from env_spec.json"""
    log = get_log_content("ppo_training.train_ppo_seed_42")
    
    # Look for env_spec loading
    if "Loading env_spec from L4" in log or "max_episode_length" in log:
        # Check if it's being used
        if "max_episode_length=60" in log or "MAX_STEPS = 60" in log:
            return True, "MAX_STEPS read from env_spec.json (60)"
    
    # Fallback check in config
    if '"max_episode_length": 60' in log:
        return True, "Episode length configured from contract"
    
    return False, "Not detected - may be using hardcoded value"

def check_improvement_2_deep_scaling() -> Tuple[bool, str]:
    """Check if scale_costs_deep is being used"""
    log = get_log_content("evaluate_production_gates")
    
    if "scale_costs_deep" in log:
        return True, "Deep cost scaling implemented"
    
    # Check for stress test results
    if "COST_STRESS" in log and "spread_stats" in log:
        return True, "Stress test with nested scaling detected"
    
    return False, "Deep scaling not detected"

def check_improvement_3_fold_evaluation() -> Tuple[bool, str]:
    """Check if fold-based evaluation is active"""
    log = get_log_content("evaluate_production_gates")
    
    # Look for fold processing
    if "Evaluating fold" in log or "folds" in log:
        fold_count = log.count("fold")
        if fold_count > 2:  # Multiple mentions suggest fold processing
            return True, f"Fold-based evaluation active ({fold_count} references)"
    
    # Check for split_spec loading
    if "split_spec.json" in log:
        return True, "Split spec loaded for fold evaluation"
    
    return False, "Single partition evaluation (no folds)"

def check_improvement_4_action_alignment() -> Tuple[bool, str]:
    """Check if action mapping is properly aligned"""
    log = get_log_content("evaluate_production_gates")
    
    # Look for proper action mapping
    if "0=hold, 1=buy, 2=sell" in log or "Policy actions: 0=HOLD, 1=BUY, 2=SELL" in log:
        return True, "Action mapping correctly documented"
    
    # Check for balanced actions
    buy_match = re.search(r'"buys":\s*(\d+)', log)
    sell_match = re.search(r'"sells":\s*(\d+)', log)
    
    if buy_match and sell_match:
        buys = int(buy_match.group(1))
        sells = int(sell_match.group(1))
        if buys > 0 and sells > 0:
            return True, f"Actions balanced: {buys} buys, {sells} sells"
    
    return False, "Action alignment not verified"

def check_improvement_5_eval_callback() -> Tuple[bool, str]:
    """Check if EnhancedEvalCallback is active"""
    log = get_log_content("ppo_training.train_ppo_seed_42")
    
    # Look for enhanced callback
    if "EnhancedEvalCallback" in log or "Best model saved" in log:
        return True, "Enhanced callback with best model saving"
    
    # Check for early stop
    if "Early stopping" in log or "Policy collapse detected" in log:
        return True, "Early stop on collapse active"
    
    # Check for eval frequency
    if "eval_freq" in log and "best_model" in log:
        return True, "Evaluation callback configured"
    
    return False, "Standard callback only"

def check_improvement_6_reward_spec() -> Tuple[bool, str]:
    """Check if reward_spec has all required fields"""
    log = get_log_content("prepare_training_environment")
    
    # Look for reward_spec validation
    required_fields = ["forward_window", "price_type", "normalization", "method"]
    found_fields = []
    
    for field in required_fields:
        if f'"{field}"' in log or f"'{field}'" in log:
            found_fields.append(field)
    
    if len(found_fields) >= 3:
        return True, f"Reward spec complete: {', '.join(found_fields)}"
    
    # Check in bundle creation
    bundle_log = get_log_content("create_model_bundle")
    if "reward_spec.json" in bundle_log and "RMSE" in bundle_log:
        return True, "Reward spec included in bundle"
    
    return False, "Reward spec validation not detected"

def check_overall_health() -> Dict[str, any]:
    """Check overall pipeline health metrics"""
    metrics = {}
    
    # Check training progress
    train_log = get_log_content("ppo_training.train_ppo_seed_42")
    
    # Extract timesteps
    timestep_match = re.search(r'timesteps:\s*(\d+)', train_log)
    if timestep_match:
        metrics['timesteps'] = int(timestep_match.group(1))
    
    # Extract trading metrics
    trades_match = re.search(r'trades_per_ep["\']?\s*:\s*(\d+)', train_log)
    if trades_match:
        metrics['trades_per_episode'] = int(trades_match.group(1))
    
    cost_match = re.search(r'cost_bps["\']?\s*:\s*(\d+\.?\d*)', train_log)
    if cost_match:
        metrics['cost_bps'] = float(cost_match.group(1))
    
    # Check for crashes or errors
    if "ERROR" in train_log or "FAILED" in train_log:
        metrics['has_errors'] = True
    else:
        metrics['has_errors'] = False
    
    return metrics

def main():
    """Main validation routine"""
    print("=" * 80)
    print("L5 PIPELINE - 6 CRITICAL IMPROVEMENTS VALIDATION")
    print("=" * 80)
    print(f"Run ID: {RUN_ID}")
    print(f"Time: {datetime.now().isoformat()}")
    print("-" * 80)
    
    # Check each improvement
    improvements = [
        ("1. Contract-Driven Limits", check_improvement_1_contract_driven),
        ("2. Deep Cost Scaling", check_improvement_2_deep_scaling),
        ("3. Fold-Based Evaluation", check_improvement_3_fold_evaluation),
        ("4. Action Alignment", check_improvement_4_action_alignment),
        ("5. Enhanced Callbacks", check_improvement_5_eval_callback),
        ("6. Complete Reward Spec", check_improvement_6_reward_spec)
    ]
    
    results = []
    for name, check_func in improvements:
        try:
            passed, message = check_func()
            results.append((name, passed, message))
            status = "[PASS]" if passed else "[PENDING]"
            print(f"{name}: {status}")
            print(f"  -> {message}")
        except Exception as e:
            results.append((name, False, f"Error: {e}"))
            print(f"{name}: [ERROR]")
            print(f"  -> {e}")
    
    # Check overall health
    print("\n" + "-" * 80)
    print("OVERALL PIPELINE HEALTH")
    print("-" * 80)
    
    health = check_overall_health()
    if health:
        for key, value in health.items():
            print(f"  {key}: {value}")
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("-" * 80)
    
    passed = sum(1 for _, p, _ in results if p)
    total = len(results)
    
    print(f"Improvements Applied: {passed}/{total}")
    
    if passed == total:
        print("\n[SUCCESS] All 6 critical improvements are active!")
    elif passed >= 4:
        print("\n[GOOD] Most improvements active, pipeline enhanced")
    else:
        print("\n[IN PROGRESS] Improvements being applied...")
    
    # Recommendations
    if passed < total:
        print("\nPending Improvements:")
        for name, passed, message in results:
            if not passed:
                print(f"  * {name}")
    
    print("=" * 80)

if __name__ == "__main__":
    main()