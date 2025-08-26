"""
Unified Service Contracts for USDCOP Trading System
Defines standard interfaces and data structures
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, Dict, List, Optional, Any
from datetime import datetime
import pandas as pd

# Data Structures
@dataclass(frozen=True)
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    spread: Optional[float] = None
    tick_volume: Optional[int] = None
    real_volume: Optional[float] = None

@dataclass(frozen=True)
class HealthSnapshot:
    service: str
    status: str
    uptime: float
    metrics: Dict[str, Any]
    active_mode: Optional[str] = None
    circuit_state: Optional[str] = None
    connected: Optional[bool] = None
    uptime_pct: Optional[float] = None
    avg_latency_ms: Optional[float] = None

@dataclass(frozen=True)
class Timeframe:
    name: str
    minutes: int
    pandas_freq: str

# Service Protocols
class DataSource(Protocol):
    @abstractmethod
    def get_historical_rates(self, symbol: str, timeframe: str, 
                           start: datetime, end: datetime) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def get_latest_rates(self, symbol: str, timeframe: str, count: int = 1) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def get_rates_count(self, symbol: str, timeframe: str) -> int:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass
    
    @abstractmethod
    def health_check(self) -> HealthSnapshot:
        pass

# Compatibility Shims
class DataSourceAdapter:
    """Adapter for legacy data sources"""
    
    def __init__(self, legacy_source):
        self.source = legacy_source
    
    def get_historical_rates(self, symbol: str, timeframe: str, 
                           start: datetime, end: datetime) -> pd.DataFrame:
        # Map legacy method to new interface
        if hasattr(self.source, 'fetch_historical'):
            return self.source.fetch_historical(symbol, timeframe, start, end)
        elif hasattr(self.source, 'get_historical_rates'):
            return self.source.get_historical_rates(symbol, timeframe, start, end)
        else:
            raise NotImplementedError(f"Source {type(self.source)} has no historical data method")
    
    def get_rates_count(self, symbol: str, timeframe: str) -> int:
        if hasattr(self.source, 'get_rates_count'):
            return self.source.get_rates_count(symbol, timeframe)
        return 0  # Default fallback
    
    def is_in_fallback_mode(self) -> bool:
        if hasattr(self.source, 'is_in_fallback_mode'):
            return self.source.is_in_fallback_mode()
        return False

# Trading Contracts
@dataclass(frozen=True)
class TradeSignal:
    symbol: str
    side: str  # 'BUY', 'SELL', 'CLOSE'
    confidence: float
    timestamp: datetime
    price: Optional[float] = None
    volume: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

@dataclass(frozen=True)
class TradeResult:
    success: bool
    order_id: Optional[str] = None
    price: Optional[float] = None
    volume: Optional[float] = None
    timestamp: Optional[datetime] = None
    error: Optional[str] = None

# Risk Management Contracts
@dataclass(frozen=True)
class RiskLimits:
    max_position_size: float
    max_daily_loss: float
    max_drawdown: float
    max_correlation: float

class RiskManager(Protocol):
    @abstractmethod
    def validate_signal(self, signal: TradeSignal) -> bool:
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: TradeSignal, account_balance: float) -> float:
        pass
    
    @abstractmethod
    def check_risk_limits(self, current_positions: List[Any]) -> bool:
        pass
