"""
DAG Dependencies - ExternalTaskSensor utilities.
GEMINI-T6 | Plan Item: P1-1

Elimina dependencias implicitas entre DAGs reemplazando
catchup=False con ExternalTaskSensor para coordinacion explicita.
"""

from datetime import timedelta
from typing import List, Optional, Dict, Any

try:
    from airflow.sensors.external_task import ExternalTaskSensor
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    ExternalTaskSensor = None


def create_dag_dependency(
    task_id: str,
    external_dag_id: str,
    external_task_id: str,
    timeout_hours: int = 1,
    execution_delta: Optional[timedelta] = None,
    dag: Any = None,
) -> Any:
    """
    Crea sensor de dependencia entre DAGs.

    Args:
        task_id: ID del task sensor
        external_dag_id: DAG del que dependemos
        external_task_id: Task especifico del que dependemos
        timeout_hours: Timeout en horas
        execution_delta: Delta de tiempo para ejecucion (si DAGs no estan sincronizados)
        dag: DAG object (required for Airflow)

    Returns:
        ExternalTaskSensor configurado

    Example:
        wait_for_features = create_dag_dependency(
            task_id='wait_for_feature_pipeline',
            external_dag_id='l4_feature_pipeline',
            external_task_id='validate_features',
            dag=dag,
        )
    """
    if not AIRFLOW_AVAILABLE:
        raise ImportError("Airflow is required for ExternalTaskSensor")

    return ExternalTaskSensor(
        task_id=task_id,
        external_dag_id=external_dag_id,
        external_task_id=external_task_id,
        timeout=timeout_hours * 3600,
        execution_delta=execution_delta or timedelta(hours=0),
        mode="reschedule",  # No bloquea worker mientras espera
        poke_interval=60,
        dag=dag,
    )


# Dependencias estandar del proyecto
# Architecture:
#   L0: Data (OHLCV, Macro) - No dependencies
#   L1: Feature refresh - Depends on L0
#   L2: Preprocessing - Depends on L0
#   L3: Model Training - Depends on L2 (needs preprocessed datasets)
#   L5: Inference - Depends on L3 (needs trained model)
DAG_DEPENDENCIES: Dict[str, List[Dict[str, str]]] = {
    "v3.l3_model_training": [
        {
            "task_id": "wait_for_preprocessing",
            "external_dag_id": "v3.l2_preprocessing_pipeline",
            "external_task_id": "validate_datasets",
        },
    ],
    "l5_multi_model_inference": [
        {
            "task_id": "wait_for_features",
            "external_dag_id": "l4_feature_pipeline",
            "external_task_id": "validate_features",
        },
        {
            "task_id": "wait_for_macro",
            "external_dag_id": "l3_macro_ingest",
            "external_task_id": "final_validation",
        },
    ],
    "l4_feature_pipeline": [
        {
            "task_id": "wait_for_ohlcv",
            "external_dag_id": "l2_ohlcv_ingest",
            "external_task_id": "validate_ohlcv",
        },
    ],
    "l3_macro_ingest": [
        # No upstream dependencies
    ],
    "l2_ohlcv_ingest": [
        # No upstream dependencies
    ],
}


def get_upstream_sensors(
    dag_id: str,
    dag: Any = None,
    timeout_hours: int = 1,
) -> List[Any]:
    """
    Creates all upstream sensors for a given DAG.

    Args:
        dag_id: ID of the DAG needing dependencies
        dag: The DAG object
        timeout_hours: Timeout for each sensor

    Returns:
        List of ExternalTaskSensor objects

    Example:
        with DAG('l5_multi_model_inference', ...) as dag:
            sensors = get_upstream_sensors('l5_multi_model_inference', dag)
            start_task = DummyOperator(task_id='start')
            for sensor in sensors:
                sensor >> start_task
    """
    if dag_id not in DAG_DEPENDENCIES:
        return []

    sensors = []
    for dep in DAG_DEPENDENCIES[dag_id]:
        sensor = create_dag_dependency(
            task_id=dep["task_id"],
            external_dag_id=dep["external_dag_id"],
            external_task_id=dep["external_task_id"],
            timeout_hours=timeout_hours,
            dag=dag,
        )
        sensors.append(sensor)

    return sensors


def validate_dag_dependencies() -> Dict[str, List[str]]:
    """
    Validates that all referenced DAGs and tasks exist.
    Returns dict of errors by DAG.

    This should be run during CI/CD to catch broken dependencies.
    """
    errors: Dict[str, List[str]] = {}

    # This would need actual DAG inspection which requires Airflow context
    # For now, just validate the structure of DAG_DEPENDENCIES

    for dag_id, deps in DAG_DEPENDENCIES.items():
        dag_errors = []

        for dep in deps:
            if "task_id" not in dep:
                dag_errors.append("Missing 'task_id' in dependency")
            if "external_dag_id" not in dep:
                dag_errors.append("Missing 'external_dag_id' in dependency")
            if "external_task_id" not in dep:
                dag_errors.append("Missing 'external_task_id' in dependency")

        if dag_errors:
            errors[dag_id] = dag_errors

    return errors
