"""
Trigger L4 RL-Ready DAG
"""

import requests
from datetime import datetime
import json

# Airflow configuration
AIRFLOW_URL = "http://localhost:8080"
AUTH = ("airflow", "airflow")

def trigger_l4_dag():
    """Trigger the L4 RL-Ready DAG"""
    
    dag_id = "usdcop_m5__05_l4_rlready"
    
    # Trigger endpoint
    url = f"{AIRFLOW_URL}/api/v1/dags/{dag_id}/dagRuns"
    
    # Execution configuration
    payload = {
        "dag_run_id": f"manual__{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "logical_date": datetime.now().isoformat(),
        "conf": {
            "process_historical": True,
            "start_date": "2020-01-01",
            "end_date": "2025-08-22"
        }
    }
    
    print(f"Triggering DAG: {dag_id}")
    print(f"Configuration: {json.dumps(payload['conf'], indent=2)}")
    
    try:
        response = requests.post(
            url,
            json=payload,
            auth=AUTH,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print(f"\n✅ DAG triggered successfully!")
            print(f"DAG Run ID: {payload['dag_run_id']}")
            print(f"\nView progress at: {AIRFLOW_URL}/dags/{dag_id}/grid")
        else:
            print(f"\n❌ Failed to trigger DAG")
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"\n❌ Error triggering DAG: {e}")
        print("\nAlternative: Run this command in terminal:")
        print(f"docker exec -it usdcop-airflow-webserver airflow dags trigger {dag_id}")

if __name__ == "__main__":
    trigger_l4_dag()