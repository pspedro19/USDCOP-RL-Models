"""
Data Validation Package
Contract: CTR-VALID

Provides validation services for data integrity.

Components:
- BacktestDataValidator: Validate data for backtesting
- UnifiedBacktestEngine: Unified backtesting with slippage (Week 1)
- SmokeTest: Pre-deployment model validation (Week 1)
"""

from src.validation.backtest_data_validator import (
    BacktestDataValidator,
    BacktestDataValidationResult,
    DataValidationError,
    ValidationIssue,
    ValidationSeverity,
)

# Unified Backtest Engine (Week 1)
from src.validation.backtest_engine import (
    SignalType,
    BacktestConfig,
    Trade,
    BacktestMetrics,
    BacktestResult,
    UnifiedBacktestEngine,
    create_backtest_engine,
)

# Smoke Test (Week 1)
from src.validation.smoke_test import (
    ValidationStatus,
    ValidationCheck,
    SmokeTestConfig,
    SmokeTestResult,
    SmokeTest,
    run_smoke_test,
)

# Great Expectations Validation Suite (P2)
from src.validation.great_expectations_suite import (
    ExpectationResult,
    ValidationReport,
    FeatureValidationSuite,
    validate_features,
    validate_training_data,
    get_validation_summary,
)

__all__ = [
    # Backtest Data Validator
    "BacktestDataValidator",
    "BacktestDataValidationResult",
    "DataValidationError",
    "ValidationIssue",
    "ValidationSeverity",
    # Unified Backtest Engine (Week 1)
    "SignalType",
    "BacktestConfig",
    "Trade",
    "BacktestMetrics",
    "BacktestResult",
    "UnifiedBacktestEngine",
    "create_backtest_engine",
    # Smoke Test (Week 1)
    "ValidationStatus",
    "ValidationCheck",
    "SmokeTestConfig",
    "SmokeTestResult",
    "SmokeTest",
    "run_smoke_test",
    # Great Expectations (P2)
    "ExpectationResult",
    "ValidationReport",
    "FeatureValidationSuite",
    "validate_features",
    "validate_training_data",
    "get_validation_summary",
]
