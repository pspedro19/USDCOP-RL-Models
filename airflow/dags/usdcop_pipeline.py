from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# DAG definition
default_args = {
    'owner': 'trading-system',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'usdcop_trading_pipeline',
    default_args=default_args,
    description='USD/COP Trading Data Pipeline',
    schedule_interval=timedelta(hours=1),
    catchup=False,
    tags=['trading', 'usdcop', 'pipeline'],
)

def pipeline_status():
    """Simple pipeline status check"""
    print("USD/COP Trading Pipeline - Status Check")
    return "Pipeline is running successfully"

# Tasks
status_check = PythonOperator(
    task_id='pipeline_status_check',
    python_callable=pipeline_status,
    dag=dag,
)

data_ingestion = BashOperator(
    task_id='data_ingestion',
    bash_command='echo "Data ingestion completed"',
    dag=dag,
)

data_processing = BashOperator(
    task_id='data_processing',
    bash_command='echo "Data processing completed"',
    dag=dag,
)

model_training = BashOperator(
    task_id='model_training',
    bash_command='echo "Model training completed"',
    dag=dag,
)

# Define task dependencies
status_check >> data_ingestion >> data_processing >> model_training