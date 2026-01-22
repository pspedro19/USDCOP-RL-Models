# data-engineering/__init__.py
"""
Data Engineering Module.

This package contains Airflow DAGs and web scrapers for data collection.

Subpackages:
    - dags: Airflow DAG definitions
    - scrapers: Web scraping utilities for data sources

DAGs:
    - dag_monthly_training: Monthly model retraining
    - dag_weekly_inference: Weekly forecast generation
    - dag_data_reconciliation: Data quality reconciliation
    - dag_backup_postgresql: Database backup
    - l0_macro_unified: Macro data ingestion
"""

__all__ = []
