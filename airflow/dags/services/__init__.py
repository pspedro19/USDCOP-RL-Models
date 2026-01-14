"""
DAG Services Module
===================
Reusable services for Airflow DAGs.
"""

from .validation_strategies import (
    ValidationStrategy,
    StandardValidationStrategy,
    ComparisonValidationStrategy,
    StrictValidationStrategy,
    WalkForwardValidationStrategy,
    ValidationStrategyRegistry,
)

from .backtest_factory import (
    AbstractBacktestRunner,
    OrchestratorBacktestRunner,
    MockBacktestRunner,
    BacktestRunnerFactory,
    BacktestConfigBuilder,
    create_backtest_runner,
)

from .alert_service import (
    AbstractNotifier,
    SlackNotifier,
    EmailNotifier,
    LogNotifier,
    WebhookNotifier,
    AlertService,
    AlertBuilder,
    get_alert_service,
    send_alert,
)

__all__ = [
    # Validation Strategies
    "ValidationStrategy",
    "StandardValidationStrategy",
    "ComparisonValidationStrategy",
    "StrictValidationStrategy",
    "WalkForwardValidationStrategy",
    "ValidationStrategyRegistry",
    # Backtest Factory
    "AbstractBacktestRunner",
    "OrchestratorBacktestRunner",
    "MockBacktestRunner",
    "BacktestRunnerFactory",
    "BacktestConfigBuilder",
    "create_backtest_runner",
    # Alert Service
    "AbstractNotifier",
    "SlackNotifier",
    "EmailNotifier",
    "LogNotifier",
    "WebhookNotifier",
    "AlertService",
    "AlertBuilder",
    "get_alert_service",
    "send_alert",
]
