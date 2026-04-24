"""
Trading signal contracts.
"""

from datetime import datetime
from enum import IntEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

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
    price: float | None = Field(None, ge=0)
    quantity: float | None = Field(None, ge=0)
    stop_loss: float | None = Field(None, ge=0)
    take_profit: float | None = Field(None, ge=0)
    source: str = SignalSource.API
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    processed_at: datetime | None = None
    is_processed: bool = False
    execution_id: UUID | None = None


class SignalCreate(BaseModel):
    """Create signal contract."""

    symbol: str = Field(min_length=2, max_length=20)
    action: SignalAction
    price: float | None = Field(None, ge=0)
    quantity: float | None = Field(None, ge=0)
    stop_loss: float | None = Field(None, ge=0)
    take_profit: float | None = Field(None, ge=0)
    source: str = SignalSource.API
    metadata: dict[str, Any] = Field(default_factory=dict)


class SignalWebhook(BaseModel):
    """TradingView webhook signal format."""

    symbol: str
    action: str  # "buy", "sell", "close"
    price: float | None = None
    quantity: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    passphrase: str | None = None  # For webhook authentication
    comment: str | None = None

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

    action: SignalAction | None = None
    symbol: str | None = None
    source: str | None = None
    is_processed: bool | None = None
    since: datetime | None = None
    until: datetime | None = None
