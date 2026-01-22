"""
Exchange adapters module.
Implements Factory and Strategy patterns for exchange integration.
"""

from .base import ExchangeAdapter, OrderResult, BalanceInfo, TickerInfo, SymbolInfo
from .factory import ExchangeAdapterFactory, get_exchange_adapter
from .mexc import MEXCAdapter
from .binance import BinanceAdapter
from .mock import MockExchangeAdapter

__all__ = [
    "ExchangeAdapter",
    "OrderResult",
    "BalanceInfo",
    "TickerInfo",
    "SymbolInfo",
    "ExchangeAdapterFactory",
    "get_exchange_adapter",
    "MEXCAdapter",
    "BinanceAdapter",
    "MockExchangeAdapter",
]
