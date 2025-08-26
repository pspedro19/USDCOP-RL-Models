"""
Base DAG Template that reads configuration from YAML
====================================================
This is a reusable template for creating DAGs that read their
configuration from YAML files, ensuring DAG_ID == MinIO prefix
"""

import yaml
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import uuid
import logging


class YamlConfiguredDAG:
    """Base class for YAML-configured DAGs"""
    
    def __init__(self, dag_id: str):
        self.dag_id = dag_id
        self.config = self._load_config()
        self.dag = self._create_dag()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load DAG configuration from YAML file"""
        config_path = Path(f"/opt/airflow/configs/{self.dag_id}.yml")
        
        # Fallback to local path if running outside container
        if not config_path.exists():
            config_path = Path(__file__).parent.parent / "configs" / f"{self.dag_id}.yml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        logging.info(f"Loaded configuration for {self.dag_id}")
        return config
    
    def _create_dag(self) -> DAG:
        """Create DAG from configuration"""
        dag_config = self.config['dag']
        
        default_args = {
            'owner': dag_config.get('owner', 'data-platform'),
            'depends_on_past': False,
            'start_date': datetime.strptime(dag_config['start_date'], '%Y-%m-%d'),
            'email_on_failure': True,
            'email_on_retry': False,
            'retries': dag_config.get('retries', 2),
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
        
        return dag
    
    def get_bucket(self) -> str:
        """Get MinIO bucket from config"""
        return self.config['minio']['bucket']
    
    def get_prefix(self) -> str:
        """Get MinIO prefix (should equal dag_id)"""
        prefix = self.config['minio']['prefix']
        # Ensure prefix equals dag_id for traceability
        if '{{ dag.id }}' in prefix:
            prefix = prefix.replace('{{ dag.id }}', self.dag_id)
        assert prefix == self.dag_id, f"Prefix must equal dag_id for traceability. Got {prefix} != {self.dag_id}"
        return prefix
    
    def get_inputs(self) -> list:
        """Get input paths from config"""
        return self.config['io'].get('inputs', [])
    
    def get_outputs(self) -> list:
        """Get output paths from config"""
        return self.config['io'].get('outputs', [])
    
    def get_rules(self) -> Dict[str, Any]:
        """Get contract rules from config"""
        return self.config.get('contracts', {}).get('rules', {})
    
    def build_path(self, template: str, **kwargs) -> str:
        """Build path from template and context"""
        path = template
        
        # Replace dag.id
        path = path.replace('{{ dag.id }}', self.dag_id)
        
        # Replace minio partitions
        for key, value in self.config['minio']['partitions'].items():
            path = path.replace(f'{{{{ minio.partitions.{key} }}}}', str(value))
        
        # Replace provided kwargs
        for key, value in kwargs.items():
            path = path.replace(f'{{{{ {key} }}}}', str(value))
            
        return path


def generate_run_id(**context) -> str:
    """Generate unique run_id for this execution"""
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    uuid_short = str(uuid.uuid4())[:8]
    run_id = f"{timestamp}-{uuid_short}"
    context['ti'].xcom_push(key='run_id', value=run_id)
    logging.info(f"Generated run_id: {run_id}")
    return run_id


def create_ready_signal(dag_config: YamlConfiguredDAG, **context) -> None:
    """Create READY signal in MinIO"""
    run_id = context['ti'].xcom_pull(key='run_id')
    execution_date = context['ds']
    
    s3_hook = S3Hook(aws_conn_id='minio_conn')
    
    # Find signal_ready in outputs
    for output in dag_config.get_outputs():
        if isinstance(output, dict) and 'signal_ready' in output:
            signal_template = output['signal_ready']
        elif isinstance(output, str) and 'signal_ready' in output:
            signal_template = output
        else:
            continue
            
        # Build signal path
        signal_path = dag_config.build_path(
            signal_template,
            date=execution_date,
            run_id=run_id
        )
        
        # Create signal
        signal_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'dag_id': dag_config.dag_id,
            'run_id': run_id,
            'execution_date': execution_date,
            'status': 'READY'
        }
        
        import json
        s3_hook.load_string(
            string_data=json.dumps(signal_data, indent=2),
            key=signal_path,
            bucket_name=dag_config.get_bucket(),
            replace=True
        )
        
        logging.info(f"Created READY signal at {signal_path}")
        break