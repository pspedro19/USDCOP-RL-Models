"""
Exchange adapters module.
Implements Factory and Strategy patterns for exchange integration.
"""

from .base import BalanceInfo, ExchangeAdapter, OrderResult, SymbolInfo, TickerInfo
from .binance import BinanceAdapter
from .factory import ExchangeAdapterFactory, get_exchange_adapter
from .mexc import MEXCAdapter
from .mock import MockExchangeAdapter

__all__ = [
    "BalanceInfo",
    "BinanceAdapter",
    "ExchangeAdapter",
    "ExchangeAdapterFactory",
    "MEXCAdapter",
    "MockExchangeAdapter",
    "OrderResult",
    "SymbolInfo",
    "TickerInfo",
    "get_exchange_adapter",
]
