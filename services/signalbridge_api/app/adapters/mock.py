"""
Mock Exchange Adapter
=====================

Simulates exchange behavior for development and testing.
Useful for:
- Unit tests without real exchange connections
- Development without API credentials
- Paper trading simulation
- Integration testing

Author: Trading Team
Version: 1.0.0
Date: 2026-01-22
"""

import asyncio
import random
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import uuid4

from app.adapters.base import (
    ExchangeAdapter,
    BalanceInfo,
    OrderResult,
    TickerInfo,
    SymbolInfo,
)
from app.contracts.exchange import SupportedExchange
from app.contracts.execution import OrderSide, ExecutionStatus


class MockExchangeAdapter(ExchangeAdapter):
    """
    Mock exchange adapter for testing and development.

    Features:
    - Simulates balance tracking
    - Simulates order execution with configurable latency
    - Supports configurable failure rates for testing error handling
    - Tracks order history

    Usage:
        adapter = MockExchangeAdapter(
            api_key="test_key",
            api_secret="test_secret",
            initial_balances={"USDT": 10000, "BTC": 0.5}
        )

        # Place an order
        result = await adapter.place_market_order("BTCUSDT", OrderSide.BUY, 0.01)
    """

    # Default trading pairs with realistic specifications
    DEFAULT_SYMBOLS = {
        "BTCUSDT": SymbolInfo(
            symbol="BTCUSDT",
            base_asset="BTC",
            quote_asset="USDT",
            min_quantity=0.00001,
            max_quantity=1000,
            step_size=0.00001,
            min_notional=10.0,
            price_precision=2,
            quantity_precision=5,
            is_active=True,
        ),
        "ETHUSDT": SymbolInfo(
            symbol="ETHUSDT",
            base_asset="ETH",
            quote_asset="USDT",
            min_quantity=0.0001,
            max_quantity=10000,
            step_size=0.0001,
            min_notional=10.0,
            price_precision=2,
            quantity_precision=4,
            is_active=True,
        ),
        "USDTCOP": SymbolInfo(
            symbol="USDTCOP",
            base_asset="USDT",
            quote_asset="COP",
            min_quantity=1.0,
            max_quantity=100000,
            step_size=1.0,
            min_notional=10000.0,
            price_precision=0,
            quantity_precision=2,
            is_active=True,
        ),
    }

    # Default prices for simulation
    DEFAULT_PRICES = {
        "BTCUSDT": 45000.0,
        "ETHUSDT": 2500.0,
        "USDTCOP": 4200.0,
    }

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        passphrase: Optional[str] = None,
        testnet: bool = True,
        initial_balances: Optional[Dict[str, float]] = None,
        failure_rate: float = 0.0,
        latency_ms: int = 50,
        slippage_bps: int = 5,
    ):
        """
        Initialize mock adapter.

        Args:
            api_key: Simulated API key
            api_secret: Simulated API secret
            passphrase: Optional passphrase (ignored)
            testnet: Always True for mock
            initial_balances: Starting balances (default: 10000 USDT)
            failure_rate: Probability of simulated failures (0.0-1.0)
            latency_ms: Simulated latency in milliseconds
            slippage_bps: Slippage in basis points
        """
        super().__init__(api_key, api_secret, passphrase, testnet=True)

        self.failure_rate = failure_rate
        self.latency_ms = latency_ms
        self.slippage_bps = slippage_bps

        # Initialize balances
        self._balances: Dict[str, float] = initial_balances or {"USDT": 10000.0}

        # Track orders
        self._orders: Dict[str, OrderResult] = {}
        self._order_count = 0

        # Simulated prices (can be updated)
        self._prices: Dict[str, float] = self.DEFAULT_PRICES.copy()

    @property
    def exchange_name(self) -> SupportedExchange:
        """Return mock as exchange identifier."""
        # Return MEXC as placeholder since SupportedExchange doesn't have MOCK
        # In production, you might add MOCK to SupportedExchange
        return SupportedExchange.MEXC

    @property
    def base_url(self) -> str:
        """Return mock URL."""
        return "http://mock-exchange.local"

    async def _simulate_latency(self) -> None:
        """Simulate network latency."""
        if self.latency_ms > 0:
            jitter = random.uniform(0.8, 1.2)
            await asyncio.sleep((self.latency_ms * jitter) / 1000)

    def _should_fail(self) -> bool:
        """Determine if this operation should fail."""
        return random.random() < self.failure_rate

    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        """Apply simulated slippage to price."""
        slippage_pct = self.slippage_bps / 10000
        if side == OrderSide.BUY:
            return price * (1 + slippage_pct)
        return price * (1 - slippage_pct)

    def set_price(self, symbol: str, price: float) -> None:
        """Set the simulated price for a symbol."""
        self._prices[symbol] = price

    def set_balance(self, asset: str, amount: float) -> None:
        """Set the balance for an asset."""
        self._balances[asset] = amount

    async def validate_credentials(self) -> bool:
        """Validate credentials (always True for mock)."""
        await self._simulate_latency()
        return not self._should_fail()

    async def get_balance(self, asset: Optional[str] = None) -> List[BalanceInfo]:
        """Get simulated balances."""
        await self._simulate_latency()

        if self._should_fail():
            raise Exception("Simulated balance fetch failure")

        if asset:
            balance = self._balances.get(asset, 0.0)
            return [BalanceInfo(asset=asset, free=balance, locked=0.0, total=balance)]

        return [
            BalanceInfo(asset=a, free=b, locked=0.0, total=b)
            for a, b in self._balances.items()
        ]

    async def get_ticker(self, symbol: str) -> TickerInfo:
        """Get simulated ticker."""
        await self._simulate_latency()

        if self._should_fail():
            raise Exception(f"Simulated ticker fetch failure for {symbol}")

        price = self._prices.get(symbol, 100.0)
        spread = price * 0.001  # 0.1% spread

        return TickerInfo(
            symbol=symbol,
            bid=price - spread / 2,
            ask=price + spread / 2,
            last=price,
            volume=random.uniform(100, 10000),
            timestamp=datetime.utcnow(),
        )

    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        """Get symbol trading information."""
        await self._simulate_latency()

        if symbol in self.DEFAULT_SYMBOLS:
            return self.DEFAULT_SYMBOLS[symbol]

        # Return generic symbol info for unknown symbols
        base, quote = symbol[:3], symbol[3:]
        return SymbolInfo(
            symbol=symbol,
            base_asset=base,
            quote_asset=quote,
            min_quantity=0.001,
            max_quantity=100000,
            step_size=0.001,
            min_notional=10.0,
            price_precision=2,
            quantity_precision=4,
            is_active=True,
        )

    async def place_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
    ) -> OrderResult:
        """Place a simulated market order."""
        await self._simulate_latency()

        if self._should_fail():
            return OrderResult(
                success=False,
                status=ExecutionStatus.FAILED,
                error_message="Simulated order failure",
            )

        # Get price with slippage
        base_price = self._prices.get(symbol, 100.0)
        fill_price = self._apply_slippage(base_price, side)

        # Generate order ID
        self._order_count += 1
        order_id = f"MOCK_{self._order_count}_{uuid4().hex[:8]}"

        # Calculate commission (0.1%)
        notional = quantity * fill_price
        commission = notional * 0.001

        # Update balances
        symbol_info = await self.get_symbol_info(symbol)
        base_asset = symbol_info.base_asset
        quote_asset = symbol_info.quote_asset

        if side == OrderSide.BUY:
            # Deduct quote, add base
            self._balances[quote_asset] = self._balances.get(quote_asset, 0) - notional - commission
            self._balances[base_asset] = self._balances.get(base_asset, 0) + quantity
        else:
            # Deduct base, add quote
            self._balances[base_asset] = self._balances.get(base_asset, 0) - quantity
            self._balances[quote_asset] = self._balances.get(quote_asset, 0) + notional - commission

        result = OrderResult(
            success=True,
            order_id=order_id,
            status=ExecutionStatus.FILLED,
            filled_quantity=quantity,
            average_price=fill_price,
            commission=commission,
            commission_asset=quote_asset,
            executed_at=datetime.utcnow(),
            raw_response={
                "mock": True,
                "simulated_slippage_bps": self.slippage_bps,
            },
        )

        self._orders[order_id] = result
        return result

    async def place_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
    ) -> OrderResult:
        """Place a simulated limit order."""
        await self._simulate_latency()

        if self._should_fail():
            return OrderResult(
                success=False,
                status=ExecutionStatus.FAILED,
                error_message="Simulated limit order failure",
            )

        # Generate order ID
        self._order_count += 1
        order_id = f"MOCK_{self._order_count}_{uuid4().hex[:8]}"

        # For simulation, immediately fill if price is favorable
        current_price = self._prices.get(symbol, 100.0)
        should_fill = (
            (side == OrderSide.BUY and price >= current_price)
            or (side == OrderSide.SELL and price <= current_price)
        )

        if should_fill:
            # Simulate immediate fill
            notional = quantity * price
            commission = notional * 0.001

            result = OrderResult(
                success=True,
                order_id=order_id,
                status=ExecutionStatus.FILLED,
                filled_quantity=quantity,
                average_price=price,
                commission=commission,
                executed_at=datetime.utcnow(),
            )
        else:
            # Order pending
            result = OrderResult(
                success=True,
                order_id=order_id,
                status=ExecutionStatus.PENDING,
                filled_quantity=0,
                average_price=0,
            )

        self._orders[order_id] = result
        return result

    async def cancel_order(
        self,
        symbol: str,
        order_id: str,
    ) -> OrderResult:
        """Cancel a simulated order."""
        await self._simulate_latency()

        if order_id not in self._orders:
            return OrderResult(
                success=False,
                order_id=order_id,
                status=ExecutionStatus.FAILED,
                error_message=f"Order {order_id} not found",
            )

        order = self._orders[order_id]

        if order.status in (ExecutionStatus.FILLED, ExecutionStatus.CANCELLED):
            return OrderResult(
                success=False,
                order_id=order_id,
                status=order.status,
                error_message=f"Cannot cancel order with status {order.status}",
            )

        # Update to cancelled
        result = OrderResult(
            success=True,
            order_id=order_id,
            status=ExecutionStatus.CANCELLED,
            filled_quantity=order.filled_quantity,
            average_price=order.average_price,
        )
        self._orders[order_id] = result
        return result

    async def get_order_status(
        self,
        symbol: str,
        order_id: str,
    ) -> OrderResult:
        """Get status of a simulated order."""
        await self._simulate_latency()

        if order_id not in self._orders:
            return OrderResult(
                success=False,
                order_id=order_id,
                status=ExecutionStatus.FAILED,
                error_message=f"Order {order_id} not found",
            )

        return self._orders[order_id]

    async def get_open_orders(
        self,
        symbol: Optional[str] = None,
    ) -> List[OrderResult]:
        """Get all open orders."""
        await self._simulate_latency()

        open_orders = [
            order for order in self._orders.values()
            if order.status == ExecutionStatus.PENDING
        ]
        return open_orders

    def get_order_history(self) -> List[OrderResult]:
        """Get all orders (mock-specific method for testing)."""
        return list(self._orders.values())

    def reset(self, initial_balances: Optional[Dict[str, float]] = None) -> None:
        """Reset mock state (useful for testing)."""
        self._balances = initial_balances or {"USDT": 10000.0}
        self._orders.clear()
        self._order_count = 0
        self._prices = self.DEFAULT_PRICES.copy()
