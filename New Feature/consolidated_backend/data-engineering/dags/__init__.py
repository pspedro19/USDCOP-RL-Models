# data-engineering/dags/__init__.py
"""
Airflow DAGs for USD/COP Forecasting Pipeline.

DAGs included:
    - dag_monthly_training: Monthly model retraining (1st Sunday of month)
    - dag_weekly_inference: Weekly forecast generation
    - dag_data_reconciliation: Data validation and reconciliation
    - dag_backup_postgresql: PostgreSQL backup
    - l0_macro_unified: Macro economic data ingestion
"""

__all__ = []
