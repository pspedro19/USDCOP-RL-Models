"""
DAG Factory - Creates DAGs from YAML configurations
====================================================
This module creates standardized DAGs that read their configuration
from YAML files, ensuring DAG_ID == MinIO prefix
"""

import yaml
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
import uuid
import logging


def load_dag_config(dag_id: str) -> Dict[str, Any]:
    """Load DAG configuration from YAML file"""
    config_path = Path(f"/opt/airflow/configs/{dag_id}.yml")
    
    # Fallback to local path if running outside container
    if not config_path.exists():
        config_path = Path(__file__).parent.parent / "configs" / f"{dag_id}.yml"
    
    if not config_path.exists():
        # Return minimal config if file not found
        return {
            'dag': {
                'id': dag_id,
                'schedule': None,
                'owner': 'data-platform',
                'start_date': '2024-01-01',
                'tags': ['auto-generated'],
                'description': f'Auto-generated DAG for {dag_id}'
            }
        }
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_run_id(**context) -> str:
    """Generate unique run_id for this execution"""
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    uuid_short = str(uuid.uuid4())[:8]
    run_id = f"{timestamp}-{uuid_short}"
    context['ti'].xcom_push(key='run_id', value=run_id)
    logging.info(f"Generated run_id: {run_id}")
    return run_id


def placeholder_task(**context) -> str:
    """Placeholder task for DAGs not yet fully implemented"""
    dag_id = context['dag'].dag_id
    logging.info(f"Executing placeholder task for {dag_id}")
    return f"Placeholder execution for {dag_id}"


def create_dag_from_config(dag_id: str) -> DAG:
    """Create a DAG from YAML configuration"""
    config = load_dag_config(dag_id)
    dag_config = config['dag']
    
    default_args = {
        'owner': dag_config.get('owner', 'data-platform'),
        'depends_on_past': False,
        'start_date': datetime.strptime(dag_config['start_date'], '%Y-%m-%d'),
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': dag_config.get('retries', 1),
        'retry_delay': timedelta(minutes=5),
    }
    
    dag = DAG(
        dag_id=dag_config['id'],
        default_args=default_args,
        description=dag_config.get('description', ''),
        schedule_interval=dag_config.get('schedule', None),
        catchup=False,
        max_active_runs=1,
        tags=dag_config.get('tags', []),
    )
    
    with dag:
        # Start task
        start = DummyOperator(
            task_id='start'
        )
        
        # Generate run_id
        task_generate_run_id = PythonOperator(
            task_id='generate_run_id',
            python_callable=generate_run_id,
        )
        
        # Placeholder main task
        task_main = PythonOperator(
            task_id='main_process',
            python_callable=placeholder_task,
        )
        
        # End task
        end = DummyOperator(
            task_id='end'
        )
        
        # Define dependencies
        start >> task_generate_run_id >> task_main >> end
    
    return dag