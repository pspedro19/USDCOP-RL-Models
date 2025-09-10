#!/usr/bin/env python3
"""
Monitor Impact of Trading Fixes on L5 Pipeline
==============================================
Tracks key metrics to validate over-trading reduction and metric stability.
"""

import os
import time
import re
from datetime import datetime
from pathlib import Path

# Configuration
RUN_ID = "manual__2025-09-05T16:01:52+00:00"
LOG_DIR = Path("airflow/logs/dag_id=usdcop_m5__06_l5_serving") / f"run_id={RUN_ID}"

def extract_metrics_from_log(log_path):
    """Extract key metrics from log file"""
    if not log_path.exists():
        return None
    
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except:
        return None
    
    metrics = {}
    
    # Extract trades per episode
    trades_match = re.search(r"trades_per_ep['\"]?\s*:\s*(\d+)", content)
    if trades_match:
        metrics['trades_per_episode'] = int(trades_match.group(1))
    
    # Extract cost sum
    cost_match = re.search(r"cost_bps_sum['\"]?\s*:\s*(\d+\.?\d*)", content)
    if cost_match:
        metrics['cost_bps'] = float(cost_match.group(1))
    
    # Extract hold percentage
    hold_match = re.search(r"%hold['\"]?\s*:\s*(\d+\.?\d*)", content)
    if hold_match:
        metrics['hold_percentage'] = float(hold_match.group(1))
    
    # Extract mean reward
    reward_match = re.search(r"mean_reward['\"]?\s*:\s*(-?\d+\.?\d*(?:e[+-]?\d+)?)", content)
    if reward_match:
        metrics['mean_reward'] = float(reward_match.group(1))
    
    # Extract Sortino ratios
    sortino_train_match = re.search(r"sortino_train['\"]?\s*:\s*(-?\d+\.?\d*(?:e[+-]?\d+)?|inf|-inf)", content)
    if sortino_train_match:
        val = sortino_train_match.group(1)
        metrics['sortino_train'] = float('inf') if val == 'inf' else float('-inf') if val == '-inf' else float(val)
    
    sortino_test_match = re.search(r"sortino_test['\"]?\s*:\s*(-?\d+\.?\d*(?:e[+-]?\d+)?|inf|-inf)", content)
    if sortino_test_match:
        val = sortino_test_match.group(1)
        metrics['sortino_test'] = float('inf') if val == 'inf' else float('-inf') if val == '-inf' else float(val)
    
    # Check for trade penalty messages
    if "Trade penalty applied" in content:
        metrics['trade_penalty_active'] = True
    
    # Check for curriculum settings
    if "initial_cost_factor=0.10" in content:
        metrics['new_curriculum'] = True
    
    return metrics

def monitor_training_progress():
    """Monitor all training tasks"""
    print("=" * 80)
    print("MONITORING L5 PIPELINE WITH TRADING FIXES")
    print("=" * 80)
    print(f"Run ID: {RUN_ID}")
    print(f"Time: {datetime.now().isoformat()}")
    print("-" * 80)
    
    seeds = [42, 123, 456]
    all_metrics = {}
    
    for seed in seeds:
        task_path = LOG_DIR / f"task_id=ppo_training.train_ppo_seed_{seed}" / "attempt=1.log"
        metrics = extract_metrics_from_log(task_path)
        
        if metrics:
            all_metrics[f'seed_{seed}'] = metrics
            print(f"\n[SEED {seed}]")
            print(f"  Trades/Episode: {metrics.get('trades_per_episode', 'N/A')}")
            print(f"  Cost (bps): {metrics.get('cost_bps', 'N/A')}")
            print(f"  Hold %: {metrics.get('hold_percentage', 'N/A')}")
            print(f"  Mean Reward: {metrics.get('mean_reward', 'N/A')}")
            print(f"  Sortino Train: {metrics.get('sortino_train', 'N/A')}")
            print(f"  Sortino Test: {metrics.get('sortino_test', 'N/A')}")
            print(f"  Trade Penalty: {'Active' if metrics.get('trade_penalty_active') else 'Not detected'}")
        else:
            print(f"\n[SEED {seed}] - Not started or no data yet")
    
    # Check gates
    gates_path = LOG_DIR / "task_id=evaluate_production_gates" / "attempt=1.log"
    gates_metrics = extract_metrics_from_log(gates_path)
    
    if gates_metrics:
        print("\n" + "=" * 80)
        print("PRODUCTION GATES STATUS")
        print("-" * 80)
        
        # Check for specific patterns
        with open(gates_path, 'r', encoding='utf-8', errors='ignore') as f:
            gates_content = f.read()
            
        if "PRODUCTION GATE STATUS: PASS" in gates_content:
            print("[SUCCESS] Gates PASSED!")
        elif "FAIL_METRICS_INVALID" in gates_content:
            print("[FAILURE] Gates failed - metrics invalid")
        else:
            print("[PENDING] Gates evaluation in progress")
        
        # Extract gate counts
        gates_match = re.search(r"(\d+) gates passed", gates_content)
        if gates_match:
            print(f"Gates Passed: {gates_match.group(1)}/8")
    
    # Summary and recommendations
    print("\n" + "=" * 80)
    print("IMPACT ASSESSMENT")
    print("-" * 80)
    
    improvements = []
    issues = []
    
    for seed_key, metrics in all_metrics.items():
        if metrics.get('trades_per_episode', 100) < 10:
            improvements.append(f"{seed_key}: Low trading frequency achieved (<10 trades/ep)")
        elif metrics.get('trades_per_episode', 100) > 20:
            issues.append(f"{seed_key}: Still over-trading (>20 trades/ep)")
        
        if metrics.get('cost_bps', 100) < 30:
            improvements.append(f"{seed_key}: Low costs achieved (<30 bps)")
        
        if metrics.get('hold_percentage', 0) > 70:
            improvements.append(f"{seed_key}: High hold percentage (>70%)")
        
        sortino_train = metrics.get('sortino_train', -100)
        sortino_test = metrics.get('sortino_test', -100)
        
        if isinstance(sortino_train, (int, float)) and isinstance(sortino_test, (int, float)):
            if abs(sortino_train) < 100 and abs(sortino_test) < 100:
                improvements.append(f"{seed_key}: Sortino values normalized (no inf)")
    
    if improvements:
        print("IMPROVEMENTS DETECTED:")
        for imp in improvements:
            print(f"  + {imp}")
    
    if issues:
        print("\nISSUES REMAINING:")
        for issue in issues:
            print(f"  - {issue}")
    
    if not improvements and not issues:
        print("Waiting for training to complete...")
    
    print("\n" + "=" * 80)

def main():
    """Main monitoring loop"""
    # Initial check
    monitor_training_progress()
    
    # Continue monitoring every 30 seconds
    print("\nContinuing to monitor... (Press Ctrl+C to stop)")
    try:
        while True:
            time.sleep(30)
            print("\n" + "="*40 + " UPDATE " + "="*40)
            monitor_training_progress()
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    main()