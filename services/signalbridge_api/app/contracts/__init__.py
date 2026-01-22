"""
Pydantic contracts for SignalBridge API.
Contract-First Design - SSOT for all data structures.
"""

from .auth import (
    LoginRequest,
    RegisterRequest,
    AuthToken,
    TokenRefreshRequest,
    PasswordChangeRequest,
)
from .user import (
    UserProfile,
    UserProfileUpdate,
    UserCreate,
)
from .exchange import (
    ExchangeCredentials,
    ExchangeCredentialsCreate,
    ExchangeCredentialsUpdate,
    ExchangeBalance,
    ExchangeInfo,
    SupportedExchange,
)
from .trading import (
    TradingConfig,
    TradingConfigUpdate,
    TradingStatus,
)
from .signal import (
    TradingSignal,
    SignalCreate,
    SignalAction,
    SignalStats,
)
from .execution import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    ExecutionStats,
    OrderType,
    OrderSide,
)
from .common import (
    PaginatedResponse,
    SuccessResponse,
    ErrorResponse,
)
from .signal_bridge import (
    InferenceSignalCreate,
    ManualSignalCreate,
    ExecutionResult as BridgeExecutionResult,
    BridgeStatus,
    BridgeHealthCheck,
    BridgeStatistics,
    RiskCheckResult,
    RiskDecision,
    RiskReason,
    TradingMode,
    KillSwitchRequest,
    UserRiskLimits,
    UserRiskLimitsCreate,
    UserRiskLimitsUpdate,
    ExecutionAuditEvent,
    BridgeEventType,
    InferenceAction,
    APIKeyValidationResult,
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
