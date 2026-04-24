"""
Trading configuration contracts.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from .common import BaseContract
from .exchange import SupportedExchange


class TradingConfigBase(BaseModel):
    """Base trading configuration."""

    trading_enabled: bool = False
    default_exchange: SupportedExchange | None = None
    max_position_size: float = Field(ge=0, le=1, default=0.1)  # 10% max
    stop_loss_percent: float = Field(ge=0, le=100, default=5.0)
    take_profit_percent: float = Field(ge=0, le=100, default=10.0)
    use_trailing_stop: bool = False
    trailing_stop_percent: float = Field(ge=0, le=100, default=2.0)
    allowed_symbols: list[str] = Field(default_factory=list)
    blocked_symbols: list[str] = Field(default_factory=list)
    max_daily_trades: int = Field(ge=0, default=50)
    max_concurrent_positions: int = Field(ge=0, default=5)


class TradingConfig(BaseContract, TradingConfigBase):
    """Trading configuration response."""

    id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: datetime | None = None


class TradingConfigUpdate(BaseModel):
    """Trading configuration update contract."""

    trading_enabled: bool | None = None
    default_exchange: SupportedExchange | None = None
    max_position_size: float | None = Field(None, ge=0, le=1)
    stop_loss_percent: float | None = Field(None, ge=0, le=100)
    take_profit_percent: float | None = Field(None, ge=0, le=100)
    use_trailing_stop: bool | None = None
    trailing_stop_percent: float | None = Field(None, ge=0, le=100)
    allowed_symbols: list[str] | None = None
    blocked_symbols: list[str] | None = None
    max_daily_trades: int | None = Field(None, ge=0)
    max_concurrent_positions: int | None = Field(None, ge=0)


class TradingStatus(BaseContract):
    """Trading status response."""

    trading_enabled: bool
    active_positions: int = 0
    daily_trades_count: int = 0
    daily_trades_limit: int = 50
    last_trade_at: datetime | None = None
    exchange_connections: dict = Field(default_factory=dict)


class TradingLimits(BaseModel):
    """Trading limits for a user."""

    max_position_size_usdt: float = Field(ge=0)
    remaining_daily_trades: int = Field(ge=0)
    can_open_position: bool = True
    reason: str | None = None
