"""
DAG Services Module
===================
Reusable services for Airflow DAGs.

Includes:
- Backtest validation strategies
- Alert/notification services
- L0 data pipeline services (UPSERT, extraction, validation)
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

from .upsert_service import UpsertService
from .seed_service import SeedService

# Macro extraction service (lazy import to avoid circular deps)
try:
    from .macro_extraction_service import MacroExtractionService
except ImportError:
    MacroExtractionService = None

# Dead-letter-queue service
try:
    from .dlq_service import (
        DeadLetterQueueService,
        DeadLetterEntry,
        DLQStatus,
        RetryResult,
        get_dlq_service,
    )
except ImportError:
    DeadLetterQueueService = None
    DeadLetterEntry = None
    DLQStatus = None
    RetryResult = None
    get_dlq_service = None

# L2 Data Quality Report
try:
    from .l2_data_quality_report import (
        L2DataQualityReportGenerator,
        L2DataQualityReport,
        VariableReport,
        DataQualityLevel,
        generate_l2_report,
    )
except ImportError:
    L2DataQualityReportGenerator = None
    L2DataQualityReport = None
    VariableReport = None
    DataQualityLevel = None
    generate_l2_report = None

# Metrics exporter
try:
    from .metrics_exporter import (
        MetricsExporter,
        get_metrics,
    )
except ImportError:
    MetricsExporter = None
    get_metrics = None

__all__ = [
    # Validation Strategies (Backtest)
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
    # L0 Pipeline Services
    "UpsertService",
    "SeedService",
    "MacroExtractionService",
    # Dead-Letter-Queue
    "DeadLetterQueueService",
    "DeadLetterEntry",
    "DLQStatus",
    "RetryResult",
    "get_dlq_service",
    # L2 Data Quality Report
    "L2DataQualityReportGenerator",
    "L2DataQualityReport",
    "VariableReport",
    "DataQualityLevel",
    "generate_l2_report",
    # Metrics Exporter
    "MetricsExporter",
    "get_metrics",
]
