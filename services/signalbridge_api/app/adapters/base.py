"""
Base exchange adapter abstract class.
Implements the Strategy pattern for exchange integration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

from app.contracts.exchange import SupportedExchange
from app.contracts.execution import OrderType, OrderSide, ExecutionStatus


@dataclass
class BalanceInfo:
    """Balance information for an asset."""

    asset: str
    free: float
    locked: float
    total: float

    @property
    def available(self) -> float:
        return self.free


@dataclass
class OrderResult:
    """Result of an order operation."""

    success: bool
    order_id: Optional[str] = None
    status: ExecutionStatus = ExecutionStatus.PENDING
    filled_quantity: float = 0.0
    average_price: float = 0.0
    commission: float = 0.0
    commission_asset: Optional[str] = None
    executed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None


@dataclass
class TickerInfo:
    """Current ticker information."""

    symbol: str
    bid: float
    ask: float
    last: float
    volume: float
    timestamp: datetime


@dataclass
class SymbolInfo:
    """Trading pair information."""

    symbol: str
    base_asset: str
    quote_asset: str
    min_quantity: float
    max_quantity: float
    step_size: float
    min_notional: float
    price_precision: int
    quantity_precision: int
    is_active: bool = True


class ExchangeAdapter(ABC):
    """
    Abstract base class for exchange adapters.
    Follows the spec's ExchangeAdapter interface.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        passphrase: Optional[str] = None,
        testnet: bool = False,
    ):
        """
        Initialize exchange adapter.

        Args:
            api_key: Exchange API key
            api_secret: Exchange API secret
            passphrase: Optional passphrase (for exchanges that require it)
            testnet: Whether to use testnet/sandbox
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.testnet = testnet
        self._exchange = None

    @property
    @abstractmethod
    def exchange_name(self) -> SupportedExchange:
        """Return the exchange identifier."""
        pass

    @property
    @abstractmethod
    def base_url(self) -> str:
        """Return the base API URL."""
        pass

    @abstractmethod
    async def validate_credentials(self) -> bool:
        """
        Validate API credentials.
        Returns True if credentials are valid.
        """
        pass

    @abstractmethod
    async def get_balance(self, asset: Optional[str] = None) -> List[BalanceInfo]:
        """
        Get account balances.

        Args:
            asset: Optional specific asset to query

        Returns:
            List of balance information
        """
        pass

    @abstractmethod
    async def get_ticker(self, symbol: str) -> TickerInfo:
        """
        Get current ticker for a symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            Current ticker information
        """
        pass

    @abstractmethod
    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        """
        Get trading pair information.

        Args:
            symbol: Trading pair symbol

        Returns:
            Symbol information including limits
        """
        pass

    @abstractmethod
    async def place_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
    ) -> OrderResult:
        """
        Place a market order.

        Args:
            symbol: Trading pair symbol
            side: Order side (buy/sell)
            quantity: Order quantity

        Returns:
            Order result
        """
        pass

    @abstractmethod
    async def place_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
    ) -> OrderResult:
        """
        Place a limit order.

        Args:
            symbol: Trading pair symbol
            side: Order side (buy/sell)
            quantity: Order quantity
            price: Limit price

        Returns:
            Order result
        """
        pass

    @abstractmethod
    async def cancel_order(
        self,
        symbol: str,
        order_id: str,
    ) -> OrderResult:
        """
        Cancel an existing order.

        Args:
            symbol: Trading pair symbol
            order_id: Order ID to cancel

        Returns:
            Cancellation result
        """
        pass

    @abstractmethod
    async def get_order_status(
        self,
        symbol: str,
        order_id: str,
    ) -> OrderResult:
        """
        Get status of an existing order.

        Args:
            symbol: Trading pair symbol
            order_id: Order ID to check

        Returns:
            Order status
        """
        pass

    async def get_open_orders(
        self,
        symbol: Optional[str] = None,
    ) -> List[OrderResult]:
        """
        Get all open orders.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of open orders
        """
        raise NotImplementedError("get_open_orders not implemented")

    def normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol format for this exchange.
        Override in subclasses if needed.
        """
        return symbol.upper().replace("/", "").replace("-", "")

    def calculate_quantity(
        self,
        symbol_info: SymbolInfo,
        quantity: float,
    ) -> float:
        """
        Round quantity to valid step size.

        Args:
            symbol_info: Symbol trading rules
            quantity: Desired quantity

        Returns:
            Rounded quantity
        """
        step = symbol_info.step_size
        precision = symbol_info.quantity_precision
        return round(quantity - (quantity % step), precision)

    async def close(self) -> None:
        """Close any connections."""
        if self._exchange:
            await self._exchange.close()
