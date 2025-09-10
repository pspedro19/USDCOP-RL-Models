#!/usr/bin/env python3
"""
Test Airflow DAG for Windows compatibility
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# Default arguments
default_args = {
    'owner': 'pedro',
    'depends_on_past': False,
    'start_date': datetime(2025, 9, 7),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'test_windows_compatibility',
    default_args=default_args,
    description='Test DAG for Windows compatibility',
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=['test', 'windows']
)

def test_python_task():
    """Simple Python task to test functionality"""
    print("Testing Python task execution on Windows")
    print("Task executed successfully!")
    return "success"

def test_system_info():
    """Test system information gathering"""
    import platform
    import os
    
    print(f"Platform: {platform.system()}")
    print(f"Platform version: {platform.version()}")
    print(f"Python version: {platform.python_version()}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Environment variables: {len(os.environ)}")
    
    return {
        'platform': platform.system(),
        'python_version': platform.python_version()
    }

# Define tasks
python_task = PythonOperator(
    task_id='test_python_task',
    python_callable=test_python_task,
    dag=dag
)

system_info_task = PythonOperator(
    task_id='test_system_info',
    python_callable=test_system_info,
    dag=dag
)

bash_task = BashOperator(
    task_id='test_bash_task',
    bash_command='echo "Testing Bash task execution on Windows"',
    dag=dag
)

# Set task dependencies
python_task >> system_info_task >> bash_task