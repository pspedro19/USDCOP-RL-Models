"""
Script to trigger and test L4 RL-Ready DAG in Airflow
"""

import os
import sys
import json
import time
from datetime import datetime
import subprocess
import requests

# Configuration
AIRFLOW_URL = "http://localhost:8080"
AIRFLOW_USER = "airflow"
AIRFLOW_PASS = "airflow"
DAG_ID = "usdcop_m5__05_l4_rlready"


def check_airflow_status():
    """Check if Airflow is running"""
    try:
        response = requests.get(f"{AIRFLOW_URL}/health", timeout=5)
        if response.status_code == 200:
            print("[OK] Airflow is running")
            return True
    except:
        pass
    
    print("[WARNING] Airflow is not accessible at", AIRFLOW_URL)
    print("[INFO] You can trigger the DAG manually with:")
    print(f"       docker exec -it usdcop-airflow-webserver airflow dags trigger {DAG_ID}")
    return False


def trigger_dag_cli():
    """Trigger DAG using CLI"""
    execution_date = datetime.now().strftime('%Y-%m-%d')
    
    # Try docker exec first
    cmd = f"docker exec -it usdcop-airflow-webserver airflow dags trigger {DAG_ID} --exec-date {execution_date}"
    
    print(f"\n[INFO] Triggering DAG: {DAG_ID}")
    print(f"[INFO] Execution date: {execution_date}")
    print(f"[INFO] Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("[OK] DAG triggered successfully")
            print(result.stdout)
            return True
        else:
            print("[ERROR] Failed to trigger DAG")
            print(result.stderr)
            
            # Try alternative command
            alt_cmd = f"airflow dags trigger {DAG_ID} --exec-date {execution_date}"
            print(f"\n[INFO] Trying alternative command: {alt_cmd}")
            
            result = subprocess.run(alt_cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print("[OK] DAG triggered successfully")
                print(result.stdout)
                return True
            else:
                print("[ERROR] Alternative command also failed")
                print(result.stderr)
    
    except Exception as e:
        print(f"[ERROR] Exception triggering DAG: {e}")
    
    return False


def check_dag_status():
    """Check DAG run status"""
    cmd = f"docker exec -it usdcop-airflow-webserver airflow dags list-runs -d {DAG_ID}"
    
    print(f"\n[INFO] Checking DAG runs for {DAG_ID}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("[OK] DAG runs:")
            print(result.stdout)
            return True
        else:
            print("[WARNING] Could not get DAG runs")
            print(result.stderr)
    
    except Exception as e:
        print(f"[ERROR] Exception checking DAG status: {e}")
    
    return False


def test_dag_parsing():
    """Test if DAG parses correctly"""
    dag_file = "airflow/dags/usdcop_m5__05_l4_rlready.py"
    
    print(f"\n[INFO] Testing DAG parsing: {dag_file}")
    
    cmd = f"python {dag_file}"
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd="..")
        
        if result.returncode == 0:
            print("[OK] DAG parsed successfully")
            return True
        else:
            print("[ERROR] DAG parsing failed")
            print(result.stderr)
            
            # Check for common issues
            if "ModuleNotFoundError" in result.stderr:
                print("\n[HINT] Missing dependencies. Install with:")
                print("       pip install apache-airflow-providers-amazon")
                print("       pip install pyarrow pandas numpy scipy")
            
            return False
    
    except Exception as e:
        print(f"[ERROR] Exception testing DAG: {e}")
        return False


def main():
    """Main function"""
    print("=" * 60)
    print("L4 RL-Ready DAG Trigger Script")
    print("=" * 60)
    
    # Test DAG parsing
    if not test_dag_parsing():
        print("\n[WARNING] DAG has parsing issues but might still work in Airflow")
    
    # Check Airflow status
    airflow_running = check_airflow_status()
    
    if not airflow_running:
        print("\n[INFO] Starting Airflow services...")
        print("[INFO] Run: docker-compose up -d")
        print("[INFO] Wait for services to start (about 30 seconds)")
    
    # Trigger DAG
    print("\n" + "=" * 60)
    if trigger_dag_cli():
        print("\n[SUCCESS] DAG triggered successfully!")
        
        # Check status
        time.sleep(2)
        check_dag_status()
        
        print("\n[INFO] Monitor the DAG at:")
        print(f"       {AIRFLOW_URL}/dags/{DAG_ID}/grid")
        print("\n[INFO] Check MinIO for outputs at:")
        print("       http://localhost:9000")
        print("       Bucket: ds-usdcop-rlready")
    else:
        print("\n[INFO] Manual trigger instructions:")
        print("1. Open Airflow UI: http://localhost:8080")
        print("2. Find DAG: usdcop_m5__05_l4_rlready")
        print("3. Click trigger button")
        print("\nOR use CLI:")
        print(f"   docker exec -it usdcop-airflow-webserver airflow dags trigger {DAG_ID}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()