#!/usr/bin/env python3
"""
Monitor L5 Anti-Collapse Improvements
======================================
Real-time tracking of policy behavior with all fixes applied.
"""

import subprocess
import re
import time
from datetime import datetime

# Configuration
RUN_ID = "manual__2025-09-05T18:43:16+00:00"

def get_log(task_id, attempt=1):
    """Get task log content"""
    cmd = f'''docker exec usdcop-airflow-webserver bash -c "tail -300 /opt/airflow/logs/dag_id=usdcop_m5__06_l5_serving/run_id={RUN_ID}/task_id={task_id}/attempt={attempt}.log 2>/dev/null"'''
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout

def check_variables():
    """Check Airflow variables are set correctly"""
    variables = {
        'L5_TOTAL_TIMESTEPS': None,
        'L5_ENT_COEF': None,
        'L5_SHAPING_PENALTY': None,
        'l4_dataset_prefix': None
    }
    
    for var in variables:
        cmd = f'docker exec usdcop-airflow-webserver airflow variables get {var} 2>/dev/null'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            variables[var] = result.stdout.strip()
    
    return variables

def extract_training_progress(seed):
    """Extract training metrics and check for improvements"""
    log = get_log(f"ppo_training.train_ppo_seed_{seed}")
    
    metrics = {
        'status': 'PENDING',
        'trades_per_ep': [],
        'cost_bps': [],
        'buys': 0,
        'sells': 0,
        'hold_pct': 100,
        'action_wrapper': False,
        'using_variables': False
    }
    
    # Check if DiscreteToSignedAction is being used
    if 'DiscreteToSignedAction' in log:
        metrics['action_wrapper'] = True
    
    # Check if using Airflow variables
    if 'L5_TOTAL_TIMESTEPS' in log or 'L5_ENT_COEF' in log:
        metrics['using_variables'] = True
    
    # Extract all trades per episode values
    trades_matches = re.findall(r'trades_per_ep["\']?\s*:\s*(\d+)', log)
    if trades_matches:
        metrics['trades_per_ep'] = [int(x) for x in trades_matches[-10:]]  # Last 10 values
    
    # Extract costs
    cost_matches = re.findall(r'cost_bps["\']?\s*:\s*(\d+\.?\d*)', log)
    if cost_matches:
        metrics['cost_bps'] = [float(x) for x in cost_matches[-10:]]
    
    # Extract buy/sell counts
    buy_matches = re.findall(r'"buys":\s*(\d+)', log)
    sell_matches = re.findall(r'"sells":\s*(\d+)', log)
    if buy_matches and sell_matches:
        metrics['buys'] = int(buy_matches[-1])
        metrics['sells'] = int(sell_matches[-1])
    
    # Extract hold percentage
    hold_matches = re.findall(r'%hold["\']?\s*:\s*(\d+\.?\d*)', log)
    if hold_matches:
        metrics['hold_pct'] = float(hold_matches[-1])
    
    # Check completion
    if 'Training completed successfully' in log or 'Done. Returned value was' in log:
        metrics['status'] = 'COMPLETED'
    elif 'ERROR' in log or 'failed' in log.lower():
        metrics['status'] = 'FAILED'
    elif metrics['trades_per_ep']:
        metrics['status'] = 'TRAINING'
    
    return metrics

def check_action_histogram():
    """Check if action histogram is being collected"""
    log = get_log("evaluate_production_gates")
    
    histogram = {
        'collected': False,
        'buy_actions': 0,
        'sell_actions': 0,
        'hold_actions': 0,
        'balanced': False
    }
    
    # Look for action histogram
    hist_match = re.search(r'ACTION_HIST_EVAL.*?{.*?"-1":\s*(\d+).*?"0":\s*(\d+).*?"1":\s*(\d+)', log)
    if hist_match:
        histogram['collected'] = True
        histogram['sell_actions'] = int(hist_match.group(1))
        histogram['hold_actions'] = int(hist_match.group(2))
        histogram['buy_actions'] = int(hist_match.group(3))
        
        total = sum([histogram['sell_actions'], histogram['hold_actions'], histogram['buy_actions']])
        if total > 0:
            hold_ratio = histogram['hold_actions'] / total
            trade_ratio = (histogram['sell_actions'] + histogram['buy_actions']) / total
            histogram['balanced'] = hold_ratio <= 0.8 and trade_ratio >= 0.2
    
    return histogram

def check_degenerate_policy():
    """Check if degenerate policy detection is working"""
    log = get_log("evaluate_production_gates")
    
    detection = {
        'active': False,
        'triggered': False,
        'reason': None
    }
    
    if 'FAIL_NONDEGENERATE' in log:
        detection['active'] = True
        detection['triggered'] = True
        
        if '0 trades' in log:
            detection['reason'] = 'Zero trades detected'
        elif 'HOLD>=95%' in log or 'hold% >= 95' in log:
            detection['reason'] = 'Excessive holding (>=95%)'
    elif 'degenerate_policy' in log:
        detection['active'] = True
    
    return detection

def main():
    print("=" * 80)
    print("L5 ANTI-COLLAPSE MONITORING")
    print("=" * 80)
    print(f"Run ID: {RUN_ID}")
    print(f"Time: {datetime.now().isoformat()}")
    print("-" * 80)
    
    # Check variables
    print("\n[AIRFLOW VARIABLES]")
    variables = check_variables()
    for var, value in variables.items():
        if value:
            print(f"  {var}: {value}")
        else:
            print(f"  {var}: [NOT SET]")
    
    # Check training progress for each seed
    print("\n[TRAINING PROGRESS]")
    seeds = [42, 123, 456]
    
    any_trading = False
    all_completed = True
    
    for seed in seeds:
        metrics = extract_training_progress(seed)
        print(f"\n  Seed {seed}: {metrics['status']}")
        
        if metrics['action_wrapper']:
            print(f"    [OK] DiscreteToSignedAction wrapper active")
        
        if metrics['using_variables']:
            print(f"    [OK] Using Airflow variables")
        
        if metrics['trades_per_ep']:
            recent_trades = metrics['trades_per_ep'][-5:]
            avg_trades = sum(recent_trades) / len(recent_trades) if recent_trades else 0
            print(f"    Recent trades/ep: {recent_trades}")
            print(f"    Avg: {avg_trades:.1f}")
            
            if avg_trades > 0:
                any_trading = True
        
        if metrics['buys'] > 0 and metrics['sells'] > 0:
            print(f"    Actions: {metrics['buys']} buys, {metrics['sells']} sells")
            print(f"    [SUCCESS] Policy using both buy and sell!")
        elif metrics['buys'] == 0 and metrics['sells'] == 0:
            print(f"    [WARNING] No trading detected")
        
        if metrics['hold_pct'] < 100:
            print(f"    Hold: {metrics['hold_pct']:.1f}%")
        
        if metrics['status'] != 'COMPLETED':
            all_completed = False
    
    # Check evaluation metrics
    if all_completed:
        print("\n[EVALUATION CHECKS]")
        
        # Action histogram
        histogram = check_action_histogram()
        if histogram['collected']:
            print(f"  Action Histogram:")
            print(f"    Sell: {histogram['sell_actions']}")
            print(f"    Hold: {histogram['hold_actions']}")
            print(f"    Buy: {histogram['buy_actions']}")
            
            if histogram['balanced']:
                print(f"    [SUCCESS] Policy is balanced!")
            else:
                print(f"    [WARNING] Policy may be imbalanced")
        
        # Degenerate policy check
        detection = check_degenerate_policy()
        if detection['active']:
            print(f"\n  Degenerate Policy Detection: ACTIVE")
            if detection['triggered']:
                print(f"    [TRIGGERED] {detection['reason']}")
            else:
                print(f"    [PASSED] Policy not degenerate")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("-" * 80)
    
    if any_trading:
        print("[SUCCESS] Policy is actively trading!")
    else:
        print("[WARNING] Limited or no trading detected")
    
    if all_completed:
        print("[INFO] Training completed - check evaluation results")
    else:
        print("[INFO] Training in progress...")
    
    print("=" * 80)

if __name__ == "__main__":
    while True:
        main()
        time.sleep(30)
        print("\n" + "="*35 + " REFRESH " + "="*36)