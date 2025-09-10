#!/usr/bin/env python3
"""
Monitor L4-L5 Integration Fixes
================================
Tracks the impact of configuration alignment and policy balance fixes.
"""

import os
import time
import re
import subprocess
from datetime import datetime
from pathlib import Path

# Configuration
L4_RUN_ID = "manual__2025-09-05T16:26:44+00:00"
L5_RUN_ID = "manual__2025-09-05T16:26:54+00:00"

def get_docker_logs(dag_id, run_id, task_pattern=None):
    """Extract logs from Docker container"""
    try:
        cmd = f'docker exec usdcop-airflow-webserver bash -c "ls /opt/airflow/logs/dag_id={dag_id}/run_id={run_id}/"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            return None
        
        tasks = result.stdout.strip().split('\n')
        logs = {}
        
        for task_line in tasks:
            if 'task_id=' not in task_line:
                continue
            task_id = task_line.split('task_id=')[1]
            
            if task_pattern and task_pattern not in task_id:
                continue
            
            # Get the log content
            log_cmd = f'docker exec usdcop-airflow-webserver bash -c "tail -200 /opt/airflow/logs/dag_id={dag_id}/run_id={run_id}/task_id={task_id}/attempt=1.log 2>/dev/null"'
            log_result = subprocess.run(log_cmd, shell=True, capture_output=True, text=True)
            
            if log_result.returncode == 0:
                logs[task_id] = log_result.stdout
        
        return logs
    except Exception as e:
        print(f"Error getting logs: {e}")
        return None

def analyze_l4_fixes(logs):
    """Analyze L4 pipeline fixes"""
    issues_fixed = {
        'clip_rate_ok': False,
        'spread_order_fixed': False,
        'reward_formula_consistent': False,
        'no_dtype_warnings': False
    }
    
    metrics = {}
    
    if not logs:
        return issues_fixed, metrics
    
    for task_id, content in logs.items():
        # Check clip rate
        clip_match = re.search(r'top_clip_rates.*?obs_\d+.*?(\d+\.\d+)', content)
        if clip_match:
            clip_val = float(clip_match.group(1))
            metrics['max_clip_rate'] = clip_val
            if clip_val < 0.005:  # < 0.5%
                issues_fixed['clip_rate_ok'] = True
        
        # Check for spread warnings
        if 'spread_proxy_bps_norm not found' not in content:
            issues_fixed['spread_order_fixed'] = True
        
        # Check reward formula consistency
        if 'ret_forward_1' in content and '+1e-8' not in content:
            issues_fixed['reward_formula_consistent'] = True
        
        # Check for dtype warnings
        if 'FutureWarning' not in content and 'dtype' not in content.lower():
            issues_fixed['no_dtype_warnings'] = True
        
        # Extract spread stats
        spread_match = re.search(r'spread_p95.*?(\d+\.?\d*)', content)
        if spread_match:
            metrics['spread_p95_bps'] = float(spread_match.group(1))
    
    return issues_fixed, metrics

def analyze_l5_fixes(logs):
    """Analyze L5 pipeline fixes"""
    fixes_applied = {
        'episode_length_60': False,
        'ppo_config_aligned': False,
        'sortino_robust': False,
        'seeds_fixed': False,
        'policy_balanced': False
    }
    
    metrics = {}
    
    if not logs:
        return fixes_applied, metrics
    
    for task_id, content in logs.items():
        # Check episode length
        if 'MAX_STEPS_PER_EPISODE=60' in content or 'max_episode_length=60' in content:
            fixes_applied['episode_length_60'] = True
        
        # Check PPO config
        if 'n_steps=1920' in content or '"n_steps": 1920' in content:
            fixes_applied['ppo_config_aligned'] = True
        
        # Check for Sortino robustness
        if 'sortino' in content.lower():
            sortino_match = re.search(r'sortino.*?(-?\d+\.?\d*|inf|-inf)', content)
            if sortino_match:
                val = sortino_match.group(1)
                if val not in ['inf', '-inf']:
                    fixes_applied['sortino_robust'] = True
                    metrics['sortino'] = float(val)
        
        # Check seed setting
        if 'random.seed' in content or 'np.random.seed' in content:
            fixes_applied['seeds_fixed'] = True
        
        # Check policy balance
        buy_match = re.search(r'"buys":\s*(\d+)', content)
        sell_match = re.search(r'"sells":\s*(\d+)', content)
        
        if buy_match and sell_match:
            buys = int(buy_match.group(1))
            sells = int(sell_match.group(1))
            metrics['buys'] = buys
            metrics['sells'] = sells
            
            if buys > 0 and sells > 0:
                total_trades = buys + sells
                buy_ratio = buys / total_trades if total_trades > 0 else 0
                sell_ratio = sells / total_trades if total_trades > 0 else 0
                
                metrics['buy_ratio'] = buy_ratio
                metrics['sell_ratio'] = sell_ratio
                
                # Balanced if both actions represent at least 5% of trades
                if buy_ratio >= 0.05 and sell_ratio >= 0.05:
                    fixes_applied['policy_balanced'] = True
        
        # Extract trading metrics
        trades_match = re.search(r'trades_per_ep.*?(\d+)', content)
        if trades_match:
            metrics['trades_per_episode'] = int(trades_match.group(1))
        
        cost_match = re.search(r'cost_bps.*?(\d+\.?\d*)', content)
        if cost_match:
            metrics['cost_bps'] = float(cost_match.group(1))
    
    return fixes_applied, metrics

def main():
    """Main monitoring loop"""
    print("=" * 80)
    print("MONITORING L4-L5 INTEGRATION FIXES")
    print("=" * 80)
    print(f"Time: {datetime.now().isoformat()}")
    print(f"L4 Run: {L4_RUN_ID}")
    print(f"L5 Run: {L5_RUN_ID}")
    print("-" * 80)
    
    # Monitor L4
    print("\n[L4 PIPELINE STATUS]")
    l4_logs = get_docker_logs("usdcop_m5__05_l4_rlready", L4_RUN_ID)
    
    if l4_logs:
        l4_fixes, l4_metrics = analyze_l4_fixes(l4_logs)
        
        print("Fixes Applied:")
        for fix, status in l4_fixes.items():
            symbol = "[PASS]" if status else "[PENDING]"
            print(f"  {fix}: {symbol}")
        
        if l4_metrics:
            print("\nMetrics:")
            for key, val in l4_metrics.items():
                print(f"  {key}: {val}")
    else:
        print("  Pipeline not started or no logs available")
    
    # Monitor L5
    print("\n[L5 PIPELINE STATUS]")
    l5_logs = get_docker_logs("usdcop_m5__06_l5_serving", L5_RUN_ID)
    
    if l5_logs:
        l5_fixes, l5_metrics = analyze_l5_fixes(l5_logs)
        
        print("Fixes Applied:")
        for fix, status in l5_fixes.items():
            symbol = "[PASS]" if status else "[PENDING]"
            print(f"  {fix}: {symbol}")
        
        if l5_metrics:
            print("\nMetrics:")
            for key, val in l5_metrics.items():
                print(f"  {key}: {val}")
        
        # Check policy balance
        if 'buys' in l5_metrics and 'sells' in l5_metrics:
            buys = l5_metrics['buys']
            sells = l5_metrics['sells']
            
            if buys == 0:
                print("\n  [WARNING] Policy collapse detected: NO BUYS!")
            elif sells == 0:
                print("\n  [WARNING] Policy collapse detected: NO SELLS!")
            elif l5_fixes['policy_balanced']:
                print("\n  [SUCCESS] Policy is balanced!")
    else:
        print("  Pipeline not started or no logs available")
    
    # Summary
    print("\n" + "=" * 80)
    print("INTEGRATION STATUS")
    print("-" * 80)
    
    if l4_logs and l5_logs:
        l4_ready = all(l4_fixes.values())
        l5_ready = all(l5_fixes.values())
        
        if l4_ready and l5_ready:
            print("[SUCCESS] Both pipelines fully fixed and integrated!")
        elif l4_ready:
            print("[PARTIAL] L4 fixed, L5 in progress...")
        elif l5_ready:
            print("[PARTIAL] L5 fixed, L4 in progress...")
        else:
            print("[IN PROGRESS] Fixes being applied...")
    else:
        print("[WAITING] Pipelines starting...")
    
    print("=" * 80)

if __name__ == "__main__":
    main()