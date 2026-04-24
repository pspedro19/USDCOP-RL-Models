"""
Pydantic contracts for SignalBridge API.
Contract-First Design - SSOT for all data structures.
"""

from .auth import (
    AuthToken,
    LoginRequest,
    PasswordChangeRequest,
    RegisterRequest,
    TokenRefreshRequest,
)
from .common import (
    ErrorResponse,
    PaginatedResponse,
    SuccessResponse,
)
from .exchange import (
    ExchangeBalance,
    ExchangeCredentials,
    ExchangeCredentialsCreate,
    ExchangeCredentialsUpdate,
    ExchangeInfo,
    SupportedExchange,
)
from .execution import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionStats,
    ExecutionStatus,
    OrderSide,
    OrderType,
)
from .signal import (
    SignalAction,
    SignalCreate,
    SignalStats,
    TradingSignal,
)
from .signal_bridge import (
    APIKeyValidationResult,
    BridgeEventType,
    BridgeHealthCheck,
    BridgeStatistics,
    BridgeStatus,
    ExecutionAuditEvent,
    InferenceAction,
    InferenceSignalCreate,
    KillSwitchRequest,
    ManualSignalCreate,
    RiskCheckResult,
    RiskDecision,
    RiskReason,
    TradingMode,
    UserRiskLimits,
    UserRiskLimitsCreate,
    UserRiskLimitsUpdate,
)
from .signal_bridge import (
    ExecutionResult as BridgeExecutionResult,
)
from .trading import (
    TradingConfig,
    TradingConfigUpdate,
    TradingStatus,
)
from .user import (
    UserCreate,
    UserProfile,
    UserProfileUpdate,
)

__all__ = [
    # Auth
    "LoginRequest",
    "RegisterRequest",
    "AuthToken",
    "TokenRefreshRequest",
    "PasswordChangeRequest",
    # User
    "UserProfile",
    "UserProfileUpdate",
    "UserCreate",
    # Exchange
    "ExchangeCredentials",
    "ExchangeCredentialsCreate",
    "ExchangeCredentialsUpdate",
    "ExchangeBalance",
    "ExchangeInfo",
    "SupportedExchange",
    # Trading
    "TradingConfig",
    "TradingConfigUpdate",
    "TradingStatus",
    # Signal
    "TradingSignal",
    "SignalCreate",
    "SignalAction",
    "SignalStats",
    # Execution
    "ExecutionRequest",
    "ExecutionResult",
    "ExecutionStatus",
    "ExecutionStats",
    "OrderType",
    "OrderSide",
    # Common
    "PaginatedResponse",
    "SuccessResponse",
    "ErrorResponse",
    # Signal Bridge
    "InferenceSignalCreate",
    "ManualSignalCreate",
    "BridgeExecutionResult",
    "BridgeStatus",
    "BridgeHealthCheck",
    "BridgeStatistics",
    "RiskCheckResult",
    "RiskDecision",
    "RiskReason",
    "TradingMode",
    "KillSwitchRequest",
    "UserRiskLimits",
    "UserRiskLimitsCreate",
    "UserRiskLimitsUpdate",
    "ExecutionAuditEvent",
    "BridgeEventType",
    "InferenceAction",
    "APIKeyValidationResult",
]
