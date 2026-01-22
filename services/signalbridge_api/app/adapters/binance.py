"""
Binance Exchange adapter implementation.
Uses CCXT for unified exchange access.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
import ccxt.async_support as ccxt

from app.contracts.exchange import SupportedExchange
from app.contracts.execution import OrderSide, ExecutionStatus
from app.core.exceptions import ExchangeError, ErrorCode
from .base import (
    ExchangeAdapter,
    BalanceInfo,
    OrderResult,
    TickerInfo,
    SymbolInfo,
)


class BinanceAdapter(ExchangeAdapter):
    """Binance Exchange adapter using CCXT."""

    @property
    def exchange_name(self) -> SupportedExchange:
        return SupportedExchange.BINANCE

    @property
    def base_url(self) -> str:
        if self.testnet:
            return "https://testnet.binance.vision"
        return "https://api.binance.com"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        passphrase: Optional[str] = None,
        testnet: bool = False,
    ):
        super().__init__(api_key, api_secret, passphrase, testnet)
        self._init_exchange()

    def _init_exchange(self) -> None:
        """Initialize CCXT exchange instance."""
        options = {
            "defaultType": "spot",
            "adjustForTimeDifference": True,
            "recvWindow": 60000,
        }

        if self.testnet:
            options["test"] = True

        self._exchange = ccxt.binance({
            "apiKey": self.api_key,
            "secret": self.api_secret,
            "sandbox": self.testnet,
            "enableRateLimit": True,
            "options": options,
        })

        if self.testnet:
            self._exchange.set_sandbox_mode(True)

    async def validate_credentials(self) -> bool:
        """Validate API credentials by fetching account info."""
        try:
            await self._exchange.fetch_balance()
            return True
        except ccxt.AuthenticationError:
            return False
        except Exception as e:
            raise ExchangeError(
                message=f"Failed to validate Binance credentials: {str(e)}",
                error_code=ErrorCode.EXCHANGE_AUTHENTICATION_FAILED,
            )

    async def get_balance(self, asset: Optional[str] = None) -> List[BalanceInfo]:
        """Get account balances."""
        try:
            balance = await self._exchange.fetch_balance()
            result = []

            for currency, data in balance.get("total", {}).items():
                if data > 0 or (asset and currency == asset.upper()):
                    free = balance.get("free", {}).get(currency, 0) or 0
                    locked = balance.get("used", {}).get(currency, 0) or 0
                    total = data or 0

                    if asset and currency != asset.upper():
                        continue

                    result.append(BalanceInfo(
                        asset=currency,
                        free=float(free),
                        locked=float(locked),
                        total=float(total),
                    ))

            return result

        except ccxt.AuthenticationError as e:
            raise ExchangeError(
                message=f"Binance authentication failed: {str(e)}",
                error_code=ErrorCode.EXCHANGE_AUTHENTICATION_FAILED,
            )
        except Exception as e:
            raise ExchangeError(
                message=f"Failed to get Binance balance: {str(e)}",
                error_code=ErrorCode.EXCHANGE_CONNECTION_FAILED,
            )

    async def get_ticker(self, symbol: str) -> TickerInfo:
        """Get current ticker for a symbol."""
        try:
            normalized = self.normalize_symbol(symbol)
            ticker = await self._exchange.fetch_ticker(normalized)

            return TickerInfo(
                symbol=normalized,
                bid=float(ticker.get("bid", 0) or 0),
                ask=float(ticker.get("ask", 0) or 0),
                last=float(ticker.get("last", 0) or 0),
                volume=float(ticker.get("baseVolume", 0) or 0),
                timestamp=datetime.fromtimestamp(ticker["timestamp"] / 1000)
                if ticker.get("timestamp")
                else datetime.utcnow(),
            )

        except Exception as e:
            raise ExchangeError(
                message=f"Failed to get Binance ticker: {str(e)}",
                error_code=ErrorCode.EXCHANGE_CONNECTION_FAILED,
            )

    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        """Get trading pair information."""
        try:
            await self._exchange.load_markets()
            normalized = self.normalize_symbol(symbol)

            if normalized not in self._exchange.markets:
                raise ExchangeError(
                    message=f"Symbol {normalized} not found on Binance",
                    error_code=ErrorCode.INVALID_SYMBOL,
                )

            market = self._exchange.markets[normalized]
            limits = market.get("limits", {})
            precision = market.get("precision", {})

            return SymbolInfo(
                symbol=normalized,
                base_asset=market.get("base", ""),
                quote_asset=market.get("quote", ""),
                min_quantity=float(limits.get("amount", {}).get("min", 0) or 0),
                max_quantity=float(limits.get("amount", {}).get("max", 0) or float("inf")),
                step_size=float(10 ** -precision.get("amount", 8)),
                min_notional=float(limits.get("cost", {}).get("min", 0) or 0),
                price_precision=int(precision.get("price", 8)),
                quantity_precision=int(precision.get("amount", 8)),
                is_active=market.get("active", True),
            )

        except ExchangeError:
            raise
        except Exception as e:
            raise ExchangeError(
                message=f"Failed to get Binance symbol info: {str(e)}",
                error_code=ErrorCode.EXCHANGE_CONNECTION_FAILED,
            )

    async def place_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
    ) -> OrderResult:
        """Place a market order."""
        try:
            normalized = self.normalize_symbol(symbol)
            order = await self._exchange.create_order(
                symbol=normalized,
                type="market",
                side=side.value,
                amount=quantity,
            )

            return self._parse_order_result(order)

        except ccxt.InsufficientFunds as e:
            return OrderResult(
                success=False,
                status=ExecutionStatus.REJECTED,
                error_message=f"Insufficient balance: {str(e)}",
            )
        except ccxt.InvalidOrder as e:
            return OrderResult(
                success=False,
                status=ExecutionStatus.REJECTED,
                error_message=f"Invalid order: {str(e)}",
            )
        except Exception as e:
            return OrderResult(
                success=False,
                status=ExecutionStatus.FAILED,
                error_message=f"Order failed: {str(e)}",
            )

    async def place_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
    ) -> OrderResult:
        """Place a limit order."""
        try:
            normalized = self.normalize_symbol(symbol)
            order = await self._exchange.create_order(
                symbol=normalized,
                type="limit",
                side=side.value,
                amount=quantity,
                price=price,
            )

            return self._parse_order_result(order)

        except ccxt.InsufficientFunds as e:
            return OrderResult(
                success=False,
                status=ExecutionStatus.REJECTED,
                error_message=f"Insufficient balance: {str(e)}",
            )
        except ccxt.InvalidOrder as e:
            return OrderResult(
                success=False,
                status=ExecutionStatus.REJECTED,
                error_message=f"Invalid order: {str(e)}",
            )
        except Exception as e:
            return OrderResult(
                success=False,
                status=ExecutionStatus.FAILED,
                error_message=f"Order failed: {str(e)}",
            )

    async def cancel_order(
        self,
        symbol: str,
        order_id: str,
    ) -> OrderResult:
        """Cancel an existing order."""
        try:
            normalized = self.normalize_symbol(symbol)
            result = await self._exchange.cancel_order(order_id, normalized)

            return OrderResult(
                success=True,
                order_id=order_id,
                status=ExecutionStatus.CANCELLED,
                raw_response=result,
            )

        except ccxt.OrderNotFound:
            return OrderResult(
                success=False,
                order_id=order_id,
                status=ExecutionStatus.FAILED,
                error_message="Order not found",
            )
        except Exception as e:
            return OrderResult(
                success=False,
                order_id=order_id,
                status=ExecutionStatus.FAILED,
                error_message=f"Cancel failed: {str(e)}",
            )

    async def get_order_status(
        self,
        symbol: str,
        order_id: str,
    ) -> OrderResult:
        """Get status of an existing order."""
        try:
            normalized = self.normalize_symbol(symbol)
            order = await self._exchange.fetch_order(order_id, normalized)

            return self._parse_order_result(order)

        except ccxt.OrderNotFound:
            return OrderResult(
                success=False,
                order_id=order_id,
                status=ExecutionStatus.FAILED,
                error_message="Order not found",
            )
        except Exception as e:
            return OrderResult(
                success=False,
                order_id=order_id,
                status=ExecutionStatus.FAILED,
                error_message=f"Failed to get order status: {str(e)}",
            )

    async def get_open_orders(
        self,
        symbol: Optional[str] = None,
    ) -> List[OrderResult]:
        """Get all open orders."""
        try:
            normalized = self.normalize_symbol(symbol) if symbol else None
            orders = await self._exchange.fetch_open_orders(normalized)

            return [self._parse_order_result(order) for order in orders]

        except Exception as e:
            raise ExchangeError(
                message=f"Failed to get open orders: {str(e)}",
                error_code=ErrorCode.EXCHANGE_CONNECTION_FAILED,
            )

    def _parse_order_result(self, order: Dict[str, Any]) -> OrderResult:
        """Parse CCXT order response to OrderResult."""
        status_map = {
            "open": ExecutionStatus.SUBMITTED,
            "closed": ExecutionStatus.FILLED,
            "canceled": ExecutionStatus.CANCELLED,
            "cancelled": ExecutionStatus.CANCELLED,
            "expired": ExecutionStatus.CANCELLED,
            "rejected": ExecutionStatus.REJECTED,
        }

        ccxt_status = order.get("status", "").lower()
        filled = float(order.get("filled", 0) or 0)
        amount = float(order.get("amount", 0) or 0)

        if ccxt_status == "closed" and filled < amount:
            status = ExecutionStatus.PARTIAL
        else:
            status = status_map.get(ccxt_status, ExecutionStatus.PENDING)

        # Parse fee
        fee = order.get("fee", {}) or {}
        commission = float(fee.get("cost", 0) or 0)
        commission_asset = fee.get("currency")

        return OrderResult(
            success=status in [ExecutionStatus.FILLED, ExecutionStatus.SUBMITTED, ExecutionStatus.PARTIAL],
            order_id=str(order.get("id", "")),
            status=status,
            filled_quantity=filled,
            average_price=float(order.get("average", 0) or order.get("price", 0) or 0),
            commission=commission,
            commission_asset=commission_asset,
            executed_at=datetime.fromtimestamp(order["timestamp"] / 1000)
            if order.get("timestamp")
            else None,
            raw_response=order,
        )

    async def close(self) -> None:
        """Close exchange connection."""
        if self._exchange:
            await self._exchange.close()
