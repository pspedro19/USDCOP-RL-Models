"""
Trading signal contracts.
"""

from datetime import datetime
from enum import IntEnum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from uuid import UUID

from .common import BaseContract


class SignalAction(IntEnum):
    """Signal action types as defined in spec."""

    BUY = 1
    SELL = 2
    CLOSE = 3
    HOLD = 0


class SignalSource(str):
    """Signal source identifiers."""

    TRADINGVIEW = "tradingview"
    CUSTOM = "custom"
    API = "api"


class TradingSignal(BaseContract):
    """Trading signal as defined in spec."""

    id: UUID
    user_id: UUID
    symbol: str = Field(description="Trading pair (e.g., 'BTCUSDT')")
    action: SignalAction
    price: Optional[float] = Field(None, ge=0)
    quantity: Optional[float] = Field(None, ge=0)
    stop_loss: Optional[float] = Field(None, ge=0)
    take_profit: Optional[float] = Field(None, ge=0)
    source: str = SignalSource.API
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    processed_at: Optional[datetime] = None
    is_processed: bool = False
    execution_id: Optional[UUID] = None


class SignalCreate(BaseModel):
    """Create signal contract."""

    symbol: str = Field(min_length=2, max_length=20)
    action: SignalAction
    price: Optional[float] = Field(None, ge=0)
    quantity: Optional[float] = Field(None, ge=0)
    stop_loss: Optional[float] = Field(None, ge=0)
    take_profit: Optional[float] = Field(None, ge=0)
    source: str = SignalSource.API
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SignalWebhook(BaseModel):
    """TradingView webhook signal format."""

    symbol: str
    action: str  # "buy", "sell", "close"
    price: Optional[float] = None
    quantity: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    passphrase: Optional[str] = None  # For webhook authentication
    comment: Optional[str] = None

    def to_signal_action(self) -> SignalAction:
        """Convert string action to SignalAction enum."""
        action_map = {
            "buy": SignalAction.BUY,
            "sell": SignalAction.SELL,
            "close": SignalAction.CLOSE,
            "hold": SignalAction.HOLD,
        }
        return action_map.get(self.action.lower(), SignalAction.HOLD)


class SignalStats(BaseContract):
    """Signal statistics."""

    total_signals: int = 0
    buy_signals: int = 0
    sell_signals: int = 0
    close_signals: int = 0
    processed_signals: int = 0
    pending_signals: int = 0
    success_rate: float = 0.0
    period_days: int = 7


class SignalFilter(BaseModel):
    """Signal filtering parameters."""

    action: Optional[SignalAction] = None
    symbol: Optional[str] = None
    source: Optional[str] = None
    is_processed: Optional[bool] = None
    since: Optional[datetime] = None
    until: Optional[datetime] = None
