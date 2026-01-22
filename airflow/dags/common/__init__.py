"""
Common utilities for Airflow DAGs
=================================
Shared modules for all DAGs including:
- BaseDAGFactory: DRY factory for creating DAGs
- SSOTImports: Centralized SSOT import management
- SecretsManager: Secrets management
- Database connections and API clients

Usage:
    from common import BaseDAGFactory, SSOTImports, get_ssot

    # Load SSOT
    ssot = get_ssot()

    # Create DAG with factory
    dag = BaseDAGFactory.create_dag('my_dag', schedule='@daily')
"""

from .secrets import (
    SecretsManager,
    DatabaseCredentials,
    APICredentials,
    get_secrets,
    get_trading_db,
    get_twelvedata_key,
    get_slack_webhook,
)

from .dag_factory import (
    BaseDAGFactory,
    SSOTImports,
    PROJECT_ROOT,
    get_ssot,
    create_trading_dag,
)

__all__ = [
    # DAG Factory (DRY)
    "BaseDAGFactory",
    "SSOTImports",
    "PROJECT_ROOT",
    "get_ssot",
    "create_trading_dag",
    # Secrets
    "SecretsManager",
    "DatabaseCredentials",
    "APICredentials",
    "get_secrets",
    "get_trading_db",
    "get_twelvedata_key",
    "get_slack_webhook",
]
