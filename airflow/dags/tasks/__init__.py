"""
Airflow Custom Tasks for USD/COP Trading System
=================================================

Reusable task modules for Airflow DAGs. These provide encapsulated
business logic that can be used across multiple DAGs.

Available Tasks:
- L5InferenceTask: Feature retrieval and inference execution with L1 fallback

Usage:
    from tasks import L5InferenceTask, run_l5_inference

    # Use the task class directly
    task = L5InferenceTask()
    features = task.get_features_for_inference("USD/COP", timestamp)

    # Or use as PythonOperator callable
    inference_task = PythonOperator(
        task_id='run_inference',
        python_callable=run_l5_inference,
        provide_context=True,
    )

Author: Trading Team
Version: 1.0.0
Created: 2026-01-17
"""

from tasks.l5_inference_task import (
    L5InferenceTask,
    run_l5_inference,
    get_inference_task,
    reset_inference_task,
    L1_FALLBACK_ALERT_THRESHOLD,
    DEFAULT_SYMBOL,
    MAX_FEATURE_AGE_MINUTES,
)

__all__ = [
    # Main task class
    'L5InferenceTask',

    # Airflow entry point
    'run_l5_inference',

    # Singleton accessors
    'get_inference_task',
    'reset_inference_task',

    # Constants
    'L1_FALLBACK_ALERT_THRESHOLD',
    'DEFAULT_SYMBOL',
    'MAX_FEATURE_AGE_MINUTES',
]
