"""
Broker Adapter — Abstract interface + PaperBroker implementation.
================================================================

Provides a sync broker interface for order placement and position tracking.
PaperBroker simulates immediate fills with configurable slippage.
Future MexcBroker will inherit the same ABC.

The broker is NOT the source of truth for position state — that lives in
the forecast_executions DB table. PaperBroker holds minimal in-memory
state only for fill simulation within a single Airflow task execution.

@version 1.0.0
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class OrderResult:
    """Result of an order placement attempt."""
    order_id: str
    status: OrderStatus
    side: OrderSide
    requested_price: float
    fill_price: float
    quantity: float
    slippage_bps: float
    timestamp: datetime


@dataclass
class PositionInfo:
    """Current position state from the broker's perspective."""
    is_open: bool
    side: Optional[OrderSide] = None
    entry_price: float = 0.0
    quantity: float = 0.0
    unrealized_pnl: float = 0.0


class BrokerAdapter(ABC):
    """Abstract broker interface. All methods are synchronous."""

    @abstractmethod
    def place_order(
        self, side: OrderSide, price: float, quantity: float = 1.0
    ) -> OrderResult:
        """Place a market order at the given price."""

    @abstractmethod
    def close_position(self, price: float) -> OrderResult:
        """Close the current position at the given price."""

    @abstractmethod
    def get_position(self) -> PositionInfo:
        """Return current position state."""

    @abstractmethod
    def cancel_all(self) -> int:
        """Cancel all open orders. Returns count of cancelled orders."""


class PaperBroker(BrokerAdapter):
    """
    Paper broker with immediate fills and configurable slippage.

    Simulates a single position at a time. Slippage is applied adversely:
    BUY fills at price * (1 + slippage_bps/10000), SELL fills at
    price * (1 - slippage_bps/10000).

    Default slippage = 1.0 bps (matches MEXC assumption from pipeline_ssot).
    """

    def __init__(self, slippage_bps: float = 1.0):
        self.slippage_bps = slippage_bps
        self._position: Optional[PositionInfo] = None

    def place_order(
        self, side: OrderSide, price: float, quantity: float = 1.0
    ) -> OrderResult:
        slippage_mult = self.slippage_bps / 10_000
        if side == OrderSide.BUY:
            fill_price = price * (1 + slippage_mult)
        else:
            fill_price = price * (1 - slippage_mult)

        self._position = PositionInfo(
            is_open=True,
            side=side,
            entry_price=fill_price,
            quantity=quantity,
        )

        return OrderResult(
            order_id=uuid.uuid4().hex[:12],
            status=OrderStatus.FILLED,
            side=side,
            requested_price=price,
            fill_price=round(fill_price, 6),
            quantity=quantity,
            slippage_bps=self.slippage_bps,
            timestamp=datetime.now(timezone.utc),
        )

    def close_position(self, price: float) -> OrderResult:
        if self._position is None or not self._position.is_open:
            return OrderResult(
                order_id=uuid.uuid4().hex[:12],
                status=OrderStatus.REJECTED,
                side=OrderSide.SELL,
                requested_price=price,
                fill_price=0.0,
                quantity=0.0,
                slippage_bps=self.slippage_bps,
                timestamp=datetime.now(timezone.utc),
            )

        close_side = (
            OrderSide.SELL if self._position.side == OrderSide.BUY else OrderSide.BUY
        )
        slippage_mult = self.slippage_bps / 10_000
        if close_side == OrderSide.BUY:
            fill_price = price * (1 + slippage_mult)
        else:
            fill_price = price * (1 - slippage_mult)

        qty = self._position.quantity
        self._position = None

        return OrderResult(
            order_id=uuid.uuid4().hex[:12],
            status=OrderStatus.FILLED,
            side=close_side,
            requested_price=price,
            fill_price=round(fill_price, 6),
            quantity=qty,
            slippage_bps=self.slippage_bps,
            timestamp=datetime.now(timezone.utc),
        )

    def get_position(self) -> PositionInfo:
        if self._position is None:
            return PositionInfo(is_open=False)
        return self._position

    def cancel_all(self) -> int:
        return 0  # Paper broker has no pending orders
