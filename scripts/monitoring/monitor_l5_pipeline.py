#!/usr/bin/env python3
"""
Monitor L5 Pipeline Execution in Real-Time
==========================================
This script monitors the L5 pipeline execution and provides real-time status updates.
"""

import os
import json
import time
import re
from datetime import datetime
from pathlib import Path
import subprocess

# Configuration
LOG_DIR = Path("airflow/logs/dag_id=usdcop_m5__06_l5_serving")
TASKS_TO_MONITOR = [
    "validate_l4_outputs",
    "prepare_training_environment", 
    "ppo_training.train_ppo_seed_42",
    "ppo_training.train_ppo_seed_123",
    "ppo_training.train_ppo_seed_456",
    "evaluate_production_gates",
    "create_model_bundle",
    "finalize_deployment"
]

def get_latest_run():
    """Find the latest run_id"""
    if not LOG_DIR.exists():
        return None
    
    runs = [d for d in LOG_DIR.iterdir() if d.is_dir() and d.name.startswith("run_id=")]
    if not runs:
        return None
    
    # Sort by modification time to get the latest
    latest_run = sorted(runs, key=lambda x: x.stat().st_mtime)[-1]
    return latest_run.name.replace("run_id=", "")

def read_task_log(run_id, task_id, attempt=1):
    """Read the log file for a specific task"""
    log_path = LOG_DIR / f"run_id={run_id}" / f"task_id={task_id}" / f"attempt={attempt}.log"
    
    if not log_path.exists():
        return None
    
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        return f"Error reading log: {e}"

def extract_task_status(log_content):
    """Extract task status from log content"""
    if not log_content:
        return "NOT_STARTED"
    
    if "Task exited with return code 0" in log_content:
        return "SUCCESS"
    elif "ERROR - Task failed with exception" in log_content:
        return "FAILED"
    elif "Marking task as FAILED" in log_content:
        return "FAILED"
    elif "Marking task as SUCCESS" in log_content:
        return "SUCCESS"
    elif "Starting attempt" in log_content:
        return "RUNNING"
    else:
        return "UNKNOWN"

def extract_key_metrics(log_content):
    """Extract key metrics from training logs"""
    metrics = {}
    
    if not log_content:
        return metrics
    
    # Extract timesteps
    timestep_match = re.search(r"Training PPO for (\d+) timesteps", log_content)
    if timestep_match:
        metrics['total_timesteps'] = int(timestep_match.group(1))
    
    # Extract Sortino ratio
    sortino_matches = re.findall(r"sortino[_\s]+(?:train|test)?\s*[:\s]+(-?\d+\.?\d*(?:e[+-]?\d+)?)", log_content, re.IGNORECASE)
    if sortino_matches:
        metrics['sortino_values'] = sortino_matches[-5:]  # Last 5 values
    
    # Extract gate status
    gate_match = re.search(r"PRODUCTION GATE STATUS:\s*(\w+)", log_content)
    if gate_match:
        metrics['gate_status'] = gate_match.group(1)
    
    # Extract percentage of holds
    hold_match = re.search(r"%hold['\"]?\s*:\s*(\d+\.?\d*)", log_content)
    if hold_match:
        metrics['hold_percentage'] = float(hold_match.group(1))
    
    # Extract infinite loop warnings
    if "possible infinite loops" in log_content:
        metrics['infinite_loop_warning'] = True
    
    # Extract JSON serialization errors
    if "Object of type type is not JSON serializable" in log_content:
        metrics['json_error'] = True
    
    # Extract reward statistics
    reward_match = re.search(r"mean_reward['\"]?\s*:\s*(-?\d+\.?\d*(?:e[+-]?\d+)?)", log_content)
    if reward_match:
        metrics['mean_reward'] = float(reward_match.group(1))
    
    return metrics

def monitor_pipeline():
    """Main monitoring loop"""
    print("=" * 80)
    print("L5 PIPELINE MONITOR")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Get latest run
    run_id = get_latest_run()
    if not run_id:
        print("No pipeline runs found!")
        return
    
    print(f"Monitoring Run: {run_id}")
    print("-" * 80)
    
    # Monitor each task
    overall_status = "SUCCESS"
    task_statuses = {}
    all_metrics = {}
    
    for task_id in TASKS_TO_MONITOR:
        log_content = read_task_log(run_id, task_id)
        status = extract_task_status(log_content)
        task_statuses[task_id] = status
        
        # Extract metrics
        metrics = extract_key_metrics(log_content)
        if metrics:
            all_metrics[task_id] = metrics
        
        # Update overall status
        if status == "FAILED":
            overall_status = "FAILED"
        elif status in ["RUNNING", "NOT_STARTED"] and overall_status != "FAILED":
            overall_status = "RUNNING"
        
        # Display status
        status_icon = {
            "SUCCESS": "[OK]",
            "FAILED": "[FAIL]",
            "RUNNING": "[RUN]",
            "NOT_STARTED": "[WAIT]",
            "UNKNOWN": "[?]"
        }.get(status, "[?]")
        
        print(f"{status_icon} {task_id:.<50} {status}")
        
        # Display key metrics for this task
        if task_id in all_metrics:
            for key, value in all_metrics[task_id].items():
                print(f"    {key}: {value}")
    
    print("-" * 80)
    print(f"Overall Status: {overall_status}")
    
    # Summary of critical issues
    print("\n" + "=" * 80)
    print("CRITICAL ISSUES DETECTED:")
    print("-" * 80)
    
    issues_found = False
    
    # Check for JSON errors
    for task_id, metrics in all_metrics.items():
        if metrics.get('json_error'):
            print(f"[ERROR] JSON serialization error in {task_id}")
            issues_found = True
    
    # Check for infinite loops
    for task_id, metrics in all_metrics.items():
        if metrics.get('infinite_loop_warning'):
            print(f"[WARNING] Possible infinite loops detected in {task_id}")
            issues_found = True
    
    # Check for high hold percentage
    for task_id, metrics in all_metrics.items():
        if metrics.get('hold_percentage', 0) > 90:
            print(f"[WARNING] Very high hold percentage ({metrics['hold_percentage']:.1f}%) in {task_id}")
            issues_found = True
    
    # Check for gate failures
    for task_id, metrics in all_metrics.items():
        if metrics.get('gate_status') == 'FAIL_METRICS_INVALID':
            print(f"[ERROR] Production gates failed with FAIL_METRICS_INVALID in {task_id}")
            issues_found = True
    
    # Check for negative rewards
    for task_id, metrics in all_metrics.items():
        if metrics.get('mean_reward', 0) < -0.001:
            print(f"[WARNING] Negative mean reward ({metrics['mean_reward']:.6f}) in {task_id}")
            issues_found = True
    
    if not issues_found:
        print("[OK] No critical issues detected")
    
    print("\n" + "=" * 80)
    
    # Save status to file
    status_file = f"pipeline_status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(status_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "run_id": run_id,
            "overall_status": overall_status,
            "task_statuses": task_statuses,
            "metrics": all_metrics
        }, f, indent=2)
    
    print(f"\nStatus saved to: {status_file}")
    
    return overall_status, task_statuses, all_metrics

def trigger_new_run():
    """Trigger a new run of the L5 pipeline via Airflow CLI"""
    print("\nTriggering new L5 pipeline run...")
    
    try:
        # Use docker exec to trigger the DAG
        cmd = [
            "docker", "exec", "usdcop-airflow-webserver",
            "airflow", "dags", "trigger", "usdcop_m5__06_l5_serving"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("[OK] New pipeline run triggered successfully!")
            print(f"Output: {result.stdout}")
            return True
        else:
            print(f"[FAIL] Failed to trigger pipeline: {result.stderr}")
            return False
    except Exception as e:
        print(f"[ERROR] Exception triggering pipeline: {e}")
        return False

if __name__ == "__main__":
    # First, check current status
    overall_status, task_statuses, metrics = monitor_pipeline()
    
    # If the last run failed due to JSON error, trigger a new one
    should_trigger_new = False
    for task_id, task_metrics in metrics.items():
        if task_metrics.get('json_error'):
            print("\n" + "=" * 80)
            print("JSON ERROR DETECTED - FIX HAS BEEN APPLIED")
            print("A new run should be triggered to test the fix")
            print("=" * 80)
            should_trigger_new = True
            break
    
    if should_trigger_new:
        response = input("\nTrigger new pipeline run? (y/n): ")
        if response.lower() == 'y':
            if trigger_new_run():
                print("\nWaiting 30 seconds for pipeline to start...")
                time.sleep(30)
                print("\nMonitoring new run...")
                monitor_pipeline()
    
    print("\nMonitoring complete!")