"""
Backtest Validation Contracts
=============================
Pydantic models defining clear contracts between pipeline stages.

SOLID Principles:
- Single Responsibility: Each contract represents one data structure
- Open/Closed: Extensible via inheritance
- Liskov Substitution: All contracts inherit from BaseContract
- Interface Segregation: Focused, minimal interfaces
- Dependency Inversion: Stages depend on contracts, not implementations

Design Patterns:
- Value Object: Immutable Pydantic models
- Contract Pattern: Explicit interfaces between stages

Author: Trading Team
Version: 1.0.0
Created: 2025-01-12
"""

from datetime import datetime, date
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


# =============================================================================
# ENUMS - Type-safe constants
# =============================================================================

class BacktestPeriodType(str, Enum):
    """Type of backtest period"""
    VALIDATION = "validation"
    TEST = "test"
    OUT_OF_SAMPLE = "out_of_sample"
    CUSTOM = "custom"


class BacktestStatus(str, Enum):
    """Status of backtest execution"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"


class AlertSeverity(str, Enum):
    """Severity level for alerts"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class ValidationResult(str, Enum):
    """Result of validation check"""
    PASSED = "passed"
    DEGRADED = "degraded"
    FAILED = "failed"


# =============================================================================
# BASE CONTRACTS
# =============================================================================

class BaseContract(BaseModel):
    """Base contract with common configuration"""
    model_config = ConfigDict(
        frozen=True,  # Immutable
        extra="forbid",  # No extra fields
        validate_assignment=True,
    )


class TimestampedContract(BaseContract):
    """Contract with timestamp metadata"""
    created_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# INPUT CONTRACTS - What stages receive
# =============================================================================

class BacktestRequest(BaseContract):
    """
    Contract: Request to run a backtest.

    Used by: DAG trigger, API endpoint
    Validated: At pipeline entry point
    """
    model_id: str = Field(..., min_length=1, description="Model identifier")
    start_date: date = Field(..., description="Backtest start date")
    end_date: date = Field(..., description="Backtest end date")
    period_type: BacktestPeriodType = Field(default=BacktestPeriodType.OUT_OF_SAMPLE)
    force_regenerate: bool = Field(default=False, description="Force regeneration even if cached")

    @model_validator(mode='after')
    def validate_dates(self) -> 'BacktestRequest':
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        return self


class ValidationThresholds(BaseContract):
    """
    Contract: Thresholds for validation checks.

    Used by: ValidationStrategy
    Source: Config or defaults
    """
    min_sharpe_ratio: float = Field(default=0.5, ge=-5.0, le=10.0)
    max_drawdown_pct: float = Field(default=0.20, ge=0.0, le=1.0)
    min_win_rate: float = Field(default=0.40, ge=0.0, le=1.0)
    min_profit_factor: float = Field(default=1.0, ge=0.0)
    min_trades: int = Field(default=10, ge=1)
    max_consecutive_losses: int = Field(default=10, ge=1)


class ModelComparisonRequest(BaseContract):
    """
    Contract: Request to compare multiple models.

    Used by: Model comparison stage
    """
    model_ids: List[str] = Field(..., min_length=2, max_length=10)
    start_date: date
    end_date: date
    baseline_model_id: Optional[str] = Field(default=None, description="Model to compare against")


# =============================================================================
# OUTPUT CONTRACTS - What stages produce
# =============================================================================

class TradeRecord(BaseContract):
    """
    Contract: Single trade record.

    Produced by: BacktestOrchestrator
    Consumed by: Metrics calculation, persistence
    """
    trade_id: Optional[int] = None
    entry_time: datetime
    exit_time: Optional[datetime] = None
    side: str = Field(..., pattern="^(LONG|SHORT)$")
    entry_price: float = Field(..., gt=0)
    exit_price: Optional[float] = Field(default=None, gt=0)
    quantity: float = Field(default=1.0, gt=0)
    pnl_usd: Optional[float] = None
    pnl_pct: Optional[float] = None
    equity_at_entry: float = Field(..., gt=0)
    equity_at_exit: Optional[float] = None
    entry_confidence: Optional[float] = Field(default=None, ge=-1, le=1)
    exit_reason: Optional[str] = None

    @property
    def duration_minutes(self) -> Optional[float]:
        if self.exit_time and self.entry_time:
            return (self.exit_time - self.entry_time).total_seconds() / 60
        return None

    @property
    def is_winner(self) -> Optional[bool]:
        if self.pnl_usd is not None:
            return self.pnl_usd > 0
        return None


class BacktestMetrics(BaseContract):
    """
    Contract: Aggregated backtest metrics.

    Produced by: Metrics calculator
    Consumed by: Validation, alerts, MLflow
    """
    # Core metrics
    total_trades: int = Field(..., ge=0)
    winning_trades: int = Field(..., ge=0)
    losing_trades: int = Field(..., ge=0)

    # Returns
    total_pnl_usd: float
    total_return_pct: float

    # Risk metrics
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown_pct: float = Field(..., ge=0, le=1)
    max_drawdown_usd: float = Field(..., ge=0)

    # Trade metrics
    win_rate: float = Field(..., ge=0, le=1)
    profit_factor: Optional[float] = None
    avg_win_usd: Optional[float] = None
    avg_loss_usd: Optional[float] = None
    avg_trade_duration_minutes: Optional[float] = None
    max_consecutive_wins: int = Field(default=0, ge=0)
    max_consecutive_losses: int = Field(default=0, ge=0)

    # Period info
    start_date: date
    end_date: date
    trading_days: int = Field(..., ge=0)

    @property
    def expectancy(self) -> Optional[float]:
        """Expected value per trade"""
        if self.total_trades > 0:
            return self.total_pnl_usd / self.total_trades
        return None


class BacktestResult(TimestampedContract):
    """
    Contract: Complete backtest result.

    Produced by: BacktestOrchestrator
    Consumed by: Next pipeline stage
    """
    # Request info
    model_id: str
    period_type: BacktestPeriodType

    # Status
    status: BacktestStatus
    source: str = Field(..., pattern="^(generated|database|cache)$")
    processing_time_ms: float = Field(..., ge=0)

    # Results
    trades: List[TradeRecord] = Field(default_factory=list)
    metrics: Optional[BacktestMetrics] = None

    # Errors
    error_message: Optional[str] = None

    @property
    def trade_count(self) -> int:
        return len(self.trades)

    @property
    def is_successful(self) -> bool:
        return self.status == BacktestStatus.COMPLETED and self.error_message is None


class ValidationCheckResult(BaseContract):
    """
    Contract: Result of a single validation check.

    Produced by: ValidationStrategy
    Consumed by: Alert service
    """
    check_name: str
    result: ValidationResult
    actual_value: float
    threshold_value: float
    message: str
    severity: AlertSeverity = AlertSeverity.INFO

    @property
    def is_passing(self) -> bool:
        return self.result == ValidationResult.PASSED


class ValidationReport(TimestampedContract):
    """
    Contract: Complete validation report.

    Produced by: Validation stage
    Consumed by: Alert service, MLflow
    """
    model_id: str
    backtest_result: BacktestResult
    thresholds: ValidationThresholds

    # Validation results
    checks: List[ValidationCheckResult] = Field(default_factory=list)
    overall_result: ValidationResult = ValidationResult.PASSED

    # Comparison (optional)
    baseline_model_id: Optional[str] = None
    baseline_metrics: Optional[BacktestMetrics] = None
    relative_performance: Optional[Dict[str, float]] = None

    @property
    def passed_checks(self) -> int:
        return sum(1 for c in self.checks if c.is_passing)

    @property
    def failed_checks(self) -> int:
        return len(self.checks) - self.passed_checks

    @property
    def critical_failures(self) -> List[ValidationCheckResult]:
        return [c for c in self.checks if c.severity == AlertSeverity.CRITICAL and not c.is_passing]


class ModelComparisonResult(TimestampedContract):
    """
    Contract: Result of comparing multiple models.

    Produced by: Comparison stage
    Consumed by: Reporting, alerts
    """
    models: Dict[str, BacktestMetrics]  # model_id -> metrics
    baseline_model_id: Optional[str] = None
    best_model_id: str
    ranking: List[Tuple[str, float]] = Field(default_factory=list)  # [(model_id, score)]

    comparison_metrics: List[str] = Field(
        default=["sharpe_ratio", "total_return_pct", "max_drawdown_pct", "win_rate"]
    )


# =============================================================================
# ALERT CONTRACTS
# =============================================================================

class Alert(TimestampedContract):
    """
    Contract: Alert to be sent.

    Produced by: Alert service
    Consumed by: Notification channels
    """
    title: str
    message: str
    severity: AlertSeverity
    source: str = "backtest_validation"

    # Context
    model_id: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    validation_report: Optional[ValidationReport] = None

    # Notification
    channels: List[str] = Field(default=["slack", "email"])
    sent: bool = False
    sent_at: Optional[datetime] = None


# =============================================================================
# PIPELINE CONTEXT CONTRACT
# =============================================================================

class PipelineContext(BaseModel):
    """
    Contract: Context passed through pipeline stages.

    Mutable container for accumulating results.
    Not frozen - designed to be updated by stages.
    """
    model_config = ConfigDict(extra="allow")

    # Initial config
    request: BacktestRequest
    thresholds: ValidationThresholds = Field(default_factory=ValidationThresholds)

    # Accumulated results
    backtest_result: Optional[BacktestResult] = None
    validation_report: Optional[ValidationReport] = None
    comparison_result: Optional[ModelComparisonResult] = None
    alerts: List[Alert] = Field(default_factory=list)

    # Metadata
    dag_run_id: Optional[str] = None
    execution_date: Optional[datetime] = None

    def add_alert(self, alert: Alert) -> None:
        """Add alert to context"""
        self.alerts.append(alert)

    def to_xcom(self) -> Dict[str, Any]:
        """Serialize for XCom"""
        return self.model_dump(mode="json")

    @classmethod
    def from_xcom(cls, data: Dict[str, Any]) -> "PipelineContext":
        """Deserialize from XCom"""
        return cls.model_validate(data)


# =============================================================================
# FACTORY CONTRACTS - Configuration for factories
# =============================================================================

class BacktestConfig(BaseContract):
    """
    Contract: Configuration for backtest execution.

    Used by: BacktestFactory
    """
    # Model
    model_id: str
    model_path: Optional[str] = None
    norm_stats_path: Optional[str] = None

    # Simulation
    initial_capital: float = Field(default=10_000.0, gt=0)
    transaction_cost_bps: float = Field(default=75.0, ge=0)  # realistic USDCOP spread
    slippage_bps: float = Field(default=15.0, ge=0)  # realistic slippage

    # Thresholds (wider HOLD zone)
    long_entry_threshold: float = Field(default=0.33, ge=-1, le=1)
    short_entry_threshold: float = Field(default=-0.33, ge=-1, le=1)
    exit_threshold: float = Field(default=0.15, ge=0, le=1)

    # Risk
    stop_loss_pct: float = Field(default=0.02, ge=0, le=1)
    take_profit_pct: float = Field(default=0.03, ge=0, le=1)
    max_position_bars: int = Field(default=20, ge=1)


class StrategyConfig(BaseContract):
    """
    Contract: Configuration for validation strategy.

    Used by: StrategyFactory
    """
    strategy_type: str = Field(default="standard")
    thresholds: ValidationThresholds = Field(default_factory=ValidationThresholds)
    compare_to_baseline: bool = Field(default=True)
    baseline_model_id: Optional[str] = None
    alert_on_degradation: bool = Field(default=True)
    alert_channels: List[str] = Field(default=["slack"])
