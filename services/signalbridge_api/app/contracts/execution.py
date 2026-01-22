"""
Execution contracts.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from uuid import UUID

from .common import BaseContract
from .exchange import SupportedExchange
from .signal import SignalAction


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
    signal_id: Optional[UUID] = None
    exchange: SupportedExchange
    credential_id: UUID
    symbol: str
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    quantity: float = Field(gt=0)
    price: Optional[float] = Field(None, ge=0)
    stop_loss: Optional[float] = Field(None, ge=0)
    take_profit: Optional[float] = Field(None, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class ExecutionResult(BaseContract):
    """Execution result as defined in spec."""

    id: UUID
    request_id: UUID
    exchange_order_id: Optional[str] = None
    status: ExecutionStatus
    filled_quantity: float = Field(ge=0, default=0)
    average_price: float = Field(ge=0, default=0)
    commission: float = Field(ge=0, default=0)
    commission_asset: Optional[str] = None
    executed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None


class ExecutionCreate(BaseModel):
    """Create execution request contract."""

    signal_id: Optional[UUID] = None
    exchange: SupportedExchange
    credential_id: UUID
    symbol: str = Field(min_length=2, max_length=20)
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    quantity: float = Field(gt=0)
    price: Optional[float] = Field(None, ge=0)
    stop_loss: Optional[float] = Field(None, ge=0)
    take_profit: Optional[float] = Field(None, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


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
    executed_at: Optional[datetime] = None


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

    exchange: Optional[SupportedExchange] = None
    symbol: Optional[str] = None
    side: Optional[OrderSide] = None
    status: Optional[ExecutionStatus] = None
    since: Optional[datetime] = None
    until: Optional[datetime] = None


class TodayStats(BaseContract):
    """Today's trading statistics."""

    total_trades: int = 0
    successful_trades: int = 0
    failed_trades: int = 0
    total_volume: float = 0.0
    total_pnl: float = 0.0
    win_rate: float = 0.0
