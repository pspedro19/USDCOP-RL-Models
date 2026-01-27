"""
Signal Bridge Contracts - Backend Pydantic Models
==================================================

Defines contracts for the SignalBridge orchestration layer that connects
inference signals to exchange execution.

SSOT:
- Action values from src/core/contracts (0=SELL, 1=HOLD, 2=BUY for inference)
- TradingMode from src/config/trading_flags.py
- ExecutionStatus from execution.py

Author: Trading Team
Version: 1.0.0
Date: 2026-01-22
"""

from datetime import datetime
from enum import Enum, IntEnum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from uuid import UUID

from .common import BaseContract
from .exchange import SupportedExchange
from .execution import ExecutionStatus, OrderSide


# =============================================================================
# Enums - SSOT Aligned
# =============================================================================

class InferenceAction(IntEnum):
    """
    Inference model action types - SSOT from src/core/contracts.

    Note: Different from SignalAction in signal.py which uses 1/2/3/0.
    This aligns with the inference model output format.
    """
    SELL = 0
    HOLD = 1
    BUY = 2


class TradingMode(str, Enum):
    """
    Trading execution modes - SSOT from src/config/trading_flags.py.
    Priority order from highest to lowest.
    """
    KILLED = "KILLED"       # Emergency stop - highest priority
    DISABLED = "DISABLED"   # Trading disabled
    SHADOW = "SHADOW"       # Signal logging only, no execution
    PAPER = "PAPER"         # Simulated trading
    STAGING = "STAGING"     # Pre-production with real data
    LIVE = "LIVE"           # Full production trading


class BridgeEventType(str, Enum):
    """Event types for execution audit trail."""
    SIGNAL_RECEIVED = "SIGNAL_RECEIVED"
    RISK_CHECK_STARTED = "RISK_CHECK_STARTED"
    RISK_CHECK_PASSED = "RISK_CHECK_PASSED"
    RISK_CHECK_FAILED = "RISK_CHECK_FAILED"
    EXECUTION_STARTED = "EXECUTION_STARTED"
    EXECUTION_SUBMITTED = "EXECUTION_SUBMITTED"
    EXECUTION_FILLED = "EXECUTION_FILLED"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    EXECUTION_CANCELLED = "EXECUTION_CANCELLED"
    KILL_SWITCH_ACTIVATED = "KILL_SWITCH_ACTIVATED"
    KILL_SWITCH_DEACTIVATED = "KILL_SWITCH_DEACTIVATED"


class RiskDecision(str, Enum):
    """Risk enforcement decision aligned with src/trading/risk_enforcer.py."""
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    REDUCE = "REDUCE"  # Allow but reduce size


class RiskReason(str, Enum):
    """Reasons for risk decisions - aligned with src/trading/risk_enforcer.py."""
    APPROVED = "approved"
    KILL_SWITCH = "kill_switch_active"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    TRADE_LIMIT = "trade_limit_reached"
    COOLDOWN = "cooldown_active"
    MAX_POSITION = "max_position_exceeded"
    LOW_CONFIDENCE = "low_confidence"
    EXPOSURE_LIMIT = "exposure_limit"
    MARKET_CLOSED = "market_closed"
    SHORT_DISABLED = "short_disabled"
    USER_LIMIT = "user_limit_exceeded"


# =============================================================================
# Input Contracts
# =============================================================================

class InferenceSignalCreate(BaseModel):
    """
    Create signal from inference API.

    This is the input format from the WebSocket bridge when receiving
    predictions from the inference API.
    """
    signal_id: UUID = Field(description="Unique signal identifier")
    model_id: str = Field(min_length=1, max_length=100, description="Model that generated the signal")
    action: int = Field(ge=0, le=2, description="0=SELL, 1=HOLD, 2=BUY (from SSOT)")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence score")
    symbol: str = Field(min_length=2, max_length=20, description="Trading pair (e.g., BTCUSDT)")
    credential_id: UUID = Field(description="Exchange credential to use")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Signal timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional signal data")

    @property
    def action_enum(self) -> InferenceAction:
        """Convert action int to enum."""
        return InferenceAction(self.action)

    @property
    def action_name(self) -> str:
        """Get human-readable action name."""
        return InferenceAction(self.action).name


class ManualSignalCreate(BaseModel):
    """
    Create signal manually via API.

    Used when submitting signals through the REST API instead of
    the WebSocket inference bridge.
    """
    model_id: str = Field(default="manual", min_length=1, max_length=100)
    action: int = Field(ge=0, le=2, description="0=SELL, 1=HOLD, 2=BUY")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    symbol: str = Field(min_length=2, max_length=20)
    credential_id: UUID
    quantity: Optional[float] = Field(None, gt=0, description="Override position size")
    stop_loss: Optional[float] = Field(None, ge=0)
    take_profit: Optional[float] = Field(None, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KillSwitchRequest(BaseModel):
    """Request to activate/deactivate kill switch."""
    activate: bool = Field(description="True to activate, False to deactivate")
    reason: str = Field(min_length=1, max_length=500, description="Reason for the action")
    confirm: bool = Field(default=False, description="Must be True for deactivation")


# =============================================================================
# Response Contracts
# =============================================================================

class RiskCheckResult(BaseContract):
    """
    Result of a risk check.
    Aligned with src/trading/risk_enforcer.py RiskCheckResult.
    """
    decision: RiskDecision
    reason: RiskReason
    message: str
    adjusted_size: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def is_allowed(self) -> bool:
        """Check if trade is allowed."""
        return self.decision in (RiskDecision.ALLOW, RiskDecision.REDUCE)

    @property
    def is_blocked(self) -> bool:
        """Check if trade is blocked."""
        return self.decision == RiskDecision.BLOCK


class ExecutionResult(BaseContract):
    """
    Result of processing a signal through the bridge.
    """
    success: bool
    execution_id: Optional[UUID] = None
    signal_id: UUID
    status: ExecutionStatus
    exchange: Optional[SupportedExchange] = None
    symbol: Optional[str] = None
    side: Optional[OrderSide] = None
    requested_quantity: float = 0
    filled_quantity: float = 0
    filled_price: float = 0
    commission: float = 0
    risk_check: Optional[RiskCheckResult] = None
    error_message: Optional[str] = None
    executed_at: Optional[datetime] = None
    processing_time_ms: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BridgeStatus(BaseContract):
    """
    Current status of the Signal Bridge.
    Used by /signal-bridge/status endpoint.
    """
    is_active: bool = Field(description="Whether the bridge is processing signals")
    kill_switch_active: bool = Field(description="Emergency stop status")
    kill_switch_reason: Optional[str] = None
    trading_mode: TradingMode = Field(description="Current trading mode")
    connected_exchanges: List[SupportedExchange] = Field(default_factory=list)
    pending_executions: int = Field(ge=0, default=0)
    last_signal_at: Optional[datetime] = None
    last_execution_at: Optional[datetime] = None
    inference_ws_connected: bool = Field(default=False, description="WebSocket to inference API")
    uptime_seconds: float = Field(ge=0, default=0)
    stats: Optional[Dict[str, Any]] = None


class BridgeHealthCheck(BaseContract):
    """Health check response for the bridge components."""
    status: str  # "healthy", "degraded", "unhealthy"
    database: bool
    redis: bool
    vault: bool
    inference_ws: bool
    exchanges: Dict[str, bool] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)


# =============================================================================
# User Risk Limits
# =============================================================================

class UserRiskLimits(BaseContract):
    """
    User-specific risk limits stored in database.
    These override the default RiskEnforcer limits for a specific user.
    """
    id: Optional[UUID] = None
    user_id: UUID
    max_daily_loss_pct: float = Field(default=2.00, ge=0, le=100)
    max_trades_per_day: int = Field(default=10, ge=1, le=1000)
    max_position_size_usd: float = Field(default=1000.00, ge=0)
    cooldown_minutes: int = Field(default=15, ge=0, le=1440)
    enable_short: bool = Field(default=False)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class UserRiskLimitsCreate(BaseModel):
    """Create user risk limits."""
    max_daily_loss_pct: float = Field(default=2.00, ge=0, le=100)
    max_trades_per_day: int = Field(default=10, ge=1, le=1000)
    max_position_size_usd: float = Field(default=1000.00, ge=0)
    cooldown_minutes: int = Field(default=15, ge=0, le=1440)
    enable_short: bool = Field(default=False)


class UserRiskLimitsUpdate(BaseModel):
    """Update user risk limits (partial)."""
    max_daily_loss_pct: Optional[float] = Field(None, ge=0, le=100)
    max_trades_per_day: Optional[int] = Field(None, ge=1, le=1000)
    max_position_size_usd: Optional[float] = Field(None, ge=0)
    cooldown_minutes: Optional[int] = Field(None, ge=0, le=1440)
    enable_short: Optional[bool] = None


# =============================================================================
# Execution Audit
# =============================================================================

class ExecutionAuditEvent(BaseContract):
    """
    Audit event for execution tracking.
    Used for compliance and debugging.
    """
    id: Optional[UUID] = None
    execution_id: UUID
    event_type: BridgeEventType
    event_data: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ExecutionAuditCreate(BaseModel):
    """Create an audit event."""
    execution_id: UUID
    event_type: BridgeEventType
    event_data: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# WebSocket Message Contracts
# =============================================================================

class WebSocketMessage(BaseModel):
    """Base WebSocket message format."""
    type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = Field(default_factory=dict)


class InferencePredictionMessage(WebSocketMessage):
    """Message format from inference API WebSocket."""
    type: str = "prediction"
    signal_id: str
    model_id: str
    action: int
    confidence: float
    symbol: str
    features: Optional[Dict[str, float]] = None


class ExecutionNotification(WebSocketMessage):
    """Notification sent to frontend about execution status."""
    type: str = "execution_update"
    execution_id: str
    status: ExecutionStatus
    filled_quantity: float = 0
    filled_price: float = 0
    error_message: Optional[str] = None


class RiskAlertNotification(WebSocketMessage):
    """Risk alert notification to frontend."""
    type: str = "risk_alert"
    alert_type: RiskReason
    message: str
    severity: str = "warning"  # "info", "warning", "critical"


class KillSwitchNotification(WebSocketMessage):
    """Kill switch status change notification."""
    type: str = "kill_switch"
    active: bool
    reason: Optional[str] = None
    activated_by: Optional[str] = None


# =============================================================================
# History and Statistics
# =============================================================================

class ExecutionHistoryFilter(BaseModel):
    """Filter parameters for execution history."""
    exchange: Optional[SupportedExchange] = None
    symbol: Optional[str] = None
    status: Optional[ExecutionStatus] = None
    model_id: Optional[str] = None
    since: Optional[datetime] = None
    until: Optional[datetime] = None
    page: int = Field(ge=1, default=1)
    limit: int = Field(ge=1, le=100, default=20)


class BridgeStatistics(BaseContract):
    """Statistics for the signal bridge."""
    total_signals_received: int = 0
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    blocked_by_risk: int = 0
    total_volume_usd: float = 0.0
    total_pnl_usd: float = 0.0
    avg_execution_time_ms: float = 0.0
    period_start: datetime
    period_end: datetime
    by_exchange: Dict[str, int] = Field(default_factory=dict)
    by_model: Dict[str, int] = Field(default_factory=dict)


# =============================================================================
# API Key Validation
# =============================================================================

class APIKeyValidationResult(BaseContract):
    """Result of API key validation."""
    is_valid: bool
    exchange: SupportedExchange
    has_trading_permission: bool = False
    has_withdraw_permission: bool = False  # Should be False for safety
    permissions: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


class UserTradingState(BaseContract):
    """
    User trading state for risk management and session tracking.
    Tracks daily P&L, trade count, and position status.
    """
    user_id: UUID
    daily_pnl: float = Field(default=0.0, description="Daily P&L in USD")
    daily_trades: int = Field(default=0, ge=0, description="Number of trades today")
    open_positions: int = Field(default=0, ge=0, description="Number of open positions")
    total_exposure_usd: float = Field(default=0.0, ge=0, description="Total exposure in USD")
    last_trade_at: Optional[datetime] = None
    cooldown_until: Optional[datetime] = None
    is_in_cooldown: bool = Field(default=False)
