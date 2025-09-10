#!/usr/bin/env python3
"""
Monitor Final L5 Pipeline Run with All Fixes
=============================================
Tracks the complete L5 execution with all 20+ improvements applied.
"""

import subprocess
import re
import time
from datetime import datetime

# Configuration
L5_RUN_ID = "manual__2025-09-05T17:20:47+00:00"

def get_task_logs(task_id, attempt=1):
    """Get logs for a specific task"""
    cmd = f'''docker exec usdcop-airflow-webserver bash -c "tail -200 /opt/airflow/logs/dag_id=usdcop_m5__06_l5_serving/run_id={L5_RUN_ID}/task_id={task_id}/attempt={attempt}.log 2>/dev/null"'''
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout

def get_pipeline_status():
    """Get overall pipeline status"""
    cmd = f"docker exec usdcop-airflow-webserver airflow tasks states-for-dag-run usdcop_m5__06_l5_serving {L5_RUN_ID}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    status = {
        'running': [],
        'success': [],
        'failed': [],
        'up_for_retry': []
    }
    
    for line in result.stdout.split('\n'):
        if 'running' in line:
            task = line.split('|')[2].strip()
            status['running'].append(task)
        elif 'success' in line:
            task = line.split('|')[2].strip()
            status['success'].append(task)
        elif 'failed' in line:
            task = line.split('|')[2].strip()
            status['failed'].append(task)
        elif 'up_for_retry' in line:
            task = line.split('|')[2].strip()
            status['up_for_retry'].append(task)
    
    return status

def extract_training_metrics(seed):
    """Extract metrics from training logs"""
    log = get_task_logs(f"ppo_training.train_ppo_seed_{seed}")
    
    metrics = {}
    
    # Extract trades per episode
    trades_match = re.search(r'trades_per_ep["\']?\s*:\s*(\d+)', log)
    if trades_match:
        metrics['trades_per_episode'] = int(trades_match.group(1))
    
    # Extract costs
    cost_match = re.search(r'cost_bps["\']?\s*:\s*(\d+\.?\d*)', log)
    if cost_match:
        metrics['cost_bps'] = float(cost_match.group(1))
    
    # Extract buy/sell counts
    buy_match = re.search(r'"buys":\s*(\d+)', log)
    sell_match = re.search(r'"sells":\s*(\d+)', log)
    if buy_match and sell_match:
        metrics['buys'] = int(buy_match.group(1))
        metrics['sells'] = int(sell_match.group(1))
    
    # Extract hold percentage
    hold_match = re.search(r'%hold["\']?\s*:\s*(\d+\.?\d*)', log)
    if hold_match:
        metrics['hold_pct'] = float(hold_match.group(1))
    
    # Extract mean reward
    reward_match = re.search(r'mean_reward["\']?\s*:\s*(-?\d+\.?\d*(?:e[+-]?\d+)?)', log)
    if reward_match:
        metrics['mean_reward'] = float(reward_match.group(1))
    
    # Check for improvements
    if 'EnhancedEvalCallback' in log:
        metrics['enhanced_callback'] = True
    if 'fold' in log.lower():
        metrics['fold_evaluation'] = True
    if 'scale_costs_deep' in log:
        metrics['deep_scaling'] = True
    
    return metrics

def check_gates():
    """Check production gates status"""
    log = get_task_logs("evaluate_production_gates")
    
    gates = {
        'status': 'PENDING',
        'gates_passed': 0,
        'total_gates': 8
    }
    
    if 'PRODUCTION GATE STATUS: PASS' in log:
        gates['status'] = 'PASSED'
    elif 'FAIL_METRICS_INVALID' in log:
        gates['status'] = 'FAILED_INVALID'
    elif 'FAIL' in log:
        gates['status'] = 'FAILED'
    
    gates_match = re.search(r'(\d+) gates passed', log)
    if gates_match:
        gates['gates_passed'] = int(gates_match.group(1))
    
    return gates

def main():
    print("=" * 80)
    print("FINAL L5 PIPELINE MONITORING")
    print("=" * 80)
    print(f"Run ID: {L5_RUN_ID}")
    print(f"Time: {datetime.now().isoformat()}")
    print("-" * 80)
    
    # Get pipeline status
    status = get_pipeline_status()
    
    print("\n[PIPELINE STATUS]")
    print(f"  Success: {len(status['success'])} tasks")
    print(f"  Running: {len(status['running'])} tasks")
    print(f"  Failed: {len(status['failed'])} tasks")
    print(f"  Retrying: {len(status['up_for_retry'])} tasks")
    
    if status['success']:
        print(f"\n  Completed: {', '.join(status['success'][:3])}")
    if status['running']:
        print(f"  In Progress: {', '.join(status['running'][:3])}")
    
    # Check training metrics for each seed
    print("\n[TRAINING METRICS]")
    seeds = [42, 123, 456]
    
    all_balanced = True
    total_trades = []
    total_costs = []
    
    for seed in seeds:
        print(f"\n  Seed {seed}:")
        metrics = extract_training_metrics(seed)
        
        if metrics:
            if 'trades_per_episode' in metrics:
                print(f"    Trades/Episode: {metrics['trades_per_episode']}")
                total_trades.append(metrics['trades_per_episode'])
            
            if 'cost_bps' in metrics:
                print(f"    Costs: {metrics['cost_bps']:.1f} bps")
                total_costs.append(metrics['cost_bps'])
            
            if 'buys' in metrics and 'sells' in metrics:
                buys = metrics['buys']
                sells = metrics['sells']
                print(f"    Actions: {buys} buys, {sells} sells")
                
                if buys == 0 or sells == 0:
                    print(f"    [WARNING] Policy imbalance detected!")
                    all_balanced = False
                elif buys > 0 and sells > 0:
                    ratio = min(buys, sells) / max(buys, sells)
                    if ratio < 0.1:
                        print(f"    [WARNING] Severe imbalance: {ratio:.1%}")
                        all_balanced = False
            
            if 'hold_pct' in metrics:
                print(f"    Hold: {metrics['hold_pct']:.1f}%")
            
            if 'mean_reward' in metrics:
                print(f"    Mean Reward: {metrics['mean_reward']:.2e}")
        else:
            print("    [PENDING]")
    
    # Check improvements
    print("\n[IMPROVEMENTS DETECTED]")
    improvements = []
    
    for seed in seeds:
        metrics = extract_training_metrics(seed)
        if metrics:
            if metrics.get('enhanced_callback'):
                improvements.append("Enhanced Callbacks")
            if metrics.get('fold_evaluation'):
                improvements.append("Fold Evaluation")
            if metrics.get('deep_scaling'):
                improvements.append("Deep Cost Scaling")
            break
    
    if improvements:
        for imp in set(improvements):
            print(f"  [ACTIVE] {imp}")
    
    # Check gates
    print("\n[PRODUCTION GATES]")
    gates = check_gates()
    print(f"  Status: {gates['status']}")
    print(f"  Gates Passed: {gates['gates_passed']}/{gates['total_gates']}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("-" * 80)
    
    if total_trades:
        avg_trades = sum(total_trades) / len(total_trades)
        print(f"  Avg Trades/Episode: {avg_trades:.1f}")
    
    if total_costs:
        avg_costs = sum(total_costs) / len(total_costs)
        print(f"  Avg Costs: {avg_costs:.1f} bps")
    
    print(f"  Policy Balance: {'BALANCED' if all_balanced else 'IMBALANCED'}")
    
    if gates['status'] == 'PASSED':
        print("\n[SUCCESS] Pipeline ready for production!")
    elif gates['status'] == 'PENDING':
        print("\n[IN PROGRESS] Waiting for completion...")
    else:
        print("\n[ATTENTION] Gates failed - review metrics")
    
    print("=" * 80)

if __name__ == "__main__":
    while True:
        main()
        time.sleep(30)
        print("\n" + "="*40 + " UPDATE " + "="*40)