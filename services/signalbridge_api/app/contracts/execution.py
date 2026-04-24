"""
Execution contracts.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from .common import BaseContract
from .exchange import SupportedExchange


class OrderType(str, Enum):
    """Order types."""

    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LOSS_LIMIT = "stop_loss_limit"
    TAKE_PROFIT = "take_profit"
    TAKE_PROFIT_LIMIT = "take_profit_limit"


class OrderSide(str, Enum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


class ExecutionStatus(str, Enum):
    """Execution status as defined in spec."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"


class ExecutionRequest(BaseContract):
    """Execution request as defined in spec."""

    id: UUID
    user_id: UUID
    signal_id: UUID | None = None
    exchange: SupportedExchange
    credential_id: UUID
    symbol: str
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    quantity: float = Field(gt=0)
    price: float | None = Field(None, ge=0)
    stop_loss: float | None = Field(None, ge=0)
    take_profit: float | None = Field(None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class ExecutionResult(BaseContract):
    """Execution result as defined in spec."""

    id: UUID
    request_id: UUID
    exchange_order_id: str | None = None
    status: ExecutionStatus
    filled_quantity: float = Field(ge=0, default=0)
    average_price: float = Field(ge=0, default=0)
    commission: float = Field(ge=0, default=0)
    commission_asset: str | None = None
    executed_at: datetime | None = None
    error_message: str | None = None
    raw_response: dict[str, Any] | None = None


class ExecutionCreate(BaseModel):
    """Create execution request contract."""

    signal_id: UUID | None = None
    exchange: SupportedExchange
    credential_id: UUID
    symbol: str = Field(min_length=2, max_length=20)
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    quantity: float = Field(gt=0)
    price: float | None = Field(None, ge=0)
    stop_loss: float | None = Field(None, ge=0)
    take_profit: float | None = Field(None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecutionSummary(BaseContract):
    """Execution summary for list views."""

    id: UUID
    exchange: SupportedExchange
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    status: ExecutionStatus
    filled_quantity: float = 0
    average_price: float = 0
    created_at: datetime
    executed_at: datetime | None = None


class ExecutionStats(BaseContract):
    """Execution statistics."""

    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    pending_executions: int = 0
    total_volume: float = 0.0
    total_commission: float = 0.0
    win_rate: float = 0.0
    period_days: int = 7


class ExecutionFilter(BaseModel):
    """Execution filtering parameters."""

    exchange: SupportedExchange | None = None
    symbol: str | None = None
    side: OrderSide | None = None
    status: ExecutionStatus | None = None
    since: datetime | None = None
    until: datetime | None = None


class TodayStats(BaseContract):
    """Today's trading statistics."""

    total_trades: int = 0
    successful_trades: int = 0
    failed_trades: int = 0
    total_volume: float = 0.0
    total_pnl: float = 0.0
    win_rate: float = 0.0
