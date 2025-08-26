"""
fallback_manager.py
==================
Unified production-ready fallback manager for 24/7 data acquisition.
Combines the best features from multiple implementations to provide
a robust, scalable solution for continuous market data access.

Key Features:
- Multiple data sources with configurable priority (MT5, CCXT, Simulator)
- Circuit breaker pattern with states (CLOSED, OPEN, HALF_OPEN)
- Comprehensive health monitoring and metrics
- Stream management with automatic restart on source switch
- Staleness detection for realtime streams
- Event callbacks for external monitoring
- Thread-safe operations with proper locking
- Manual override capabilities
- Exponential backoff and cooldown periods
- Detailed logging and error tracking

Usage:
    from src.core.connectors.fallback_manager import FallbackManager, DataSourceConfig
    
    # Configure sources
    config = FallbackManager.create_default_config()
    
    # Initialize manager
    manager = FallbackManager(config)
    await manager.initialize()
    
    # Register callbacks
    manager.add_event_callback(lambda event, data: print(f"Event: {event}"))
    
    # Fetch data with automatic fallback
    df = await manager.get_historical_data('USDCOP', 'M5', start, end)
    
    # Start streaming with auto-recovery
    stop_handle = await manager.start_stream('USDCOP', 'M1', on_bar_callback)
    
    # Monitor health
    health = manager.get_health_status()
    
    # Shutdown gracefully
    await manager.shutdown()
"""

import os
import asyncio
import enum
import logging
import threading
import time
import json
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Callable, Any, Tuple, Union

import pandas as pd
import numpy as np

# Event bus integration
try:
    from src.core.events.bus import event_bus, Event, EventType
    from src.utils.logger import get_correlation_id
    EVENT_BUS_AVAILABLE = True
except ImportError:
    EVENT_BUS_AVAILABLE = False
    event_bus = None
    Event = None
    EventType = None
    get_correlation_id = lambda: None

# Try importing optional dependencies
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    ccxt = None

# Import required connectors
from .mt5_connector import (
    RobustMT5Connector, 
    ConnectorConfig,
    BaseConnector,
    ConnectorMode
)
from src.core.data.data_simulator import DataSimulator, SimulatorConfig

# Configure logging
logger = logging.getLogger(__name__)


# ======================================================================
# Enums and Types
# ======================================================================

class DataSourceType(str, enum.Enum):
    """Available data source types"""
    MT5 = "MT5"
    CCXT = "CCXT"
    SIMULATOR = "SIMULATOR"
    CUSTOM = "CUSTOM"


class SourceStatus(str, enum.Enum):
    """Data source operational status"""
    ACTIVE = "ACTIVE"          # Currently serving data
    STANDBY = "STANDBY"        # Available but not active
    FAILED = "FAILED"          # Failed and in cooldown
    DISABLED = "DISABLED"      # Administratively disabled
    INITIALIZING = "INITIALIZING"  # Starting up


class CircuitState(str, enum.Enum):
    """Circuit breaker states for failure management"""
    CLOSED = "CLOSED"          # Normal operation (confusingly, this means "working")
    OPEN = "OPEN"              # Circuit open, failures detected (not working)
    HALF_OPEN = "HALF_OPEN"    # Testing recovery


# ======================================================================
# Configuration Classes
# ======================================================================

@dataclass
class DataSourceConfig:
    """Configuration for a single data source"""
    # Priority (1 = highest)
    priority: int = 1
    enabled: bool = True
    
    # Failure handling
    max_failures: int = 3
    max_consecutive_errors: int = 3
    cooldown_minutes: int = 5
    
    # Health monitoring
    health_check_interval: int = 30  # seconds
    
    # Staleness detection (for streams)
    max_staleness_seconds: int = 120
    
    # Circuit breaker
    circuit_cooldown_seconds: int = 60
    half_open_probe_interval: int = 30
    
    # Type-specific configurations
    mt5_config: Optional[Dict[str, Any]] = None
    ccxt_exchange: str = "binance"
    ccxt_config: Optional[Dict[str, Any]] = None
    simulator_config: Optional[Dict[str, Any]] = None
    
    # Special flags
    always_available: bool = False  # For simulator
    require_auth: bool = True
    
    # Retry configuration
    retry_count: int = 3
    retry_delay_base: float = 1.0  # Exponential backoff base


@dataclass
class SourceMetrics:
    """Performance and health metrics for a data source"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    consecutive_failures: int = 0
    consecutive_errors: int = 0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    last_error: Optional[str] = None
    average_latency_ms: float = 0.0
    uptime_percent: float = 100.0
    
    # Stream metrics
    stream_count: int = 0
    last_stream_update: Optional[datetime] = None
    
    def record_success(self, latency_ms: float):
        """Record successful operation"""
        self.total_requests += 1
        self.successful_requests += 1
        self.consecutive_failures = 0
        self.consecutive_errors = 0
        self.last_success = datetime.now(timezone.utc)
        
        # Update average latency (exponential moving average)
        if self.average_latency_ms == 0:
            self.average_latency_ms = latency_ms
        else:
            self.average_latency_ms = (self.average_latency_ms * 0.9 + latency_ms * 0.1)
        
        self._update_uptime()
    
    def record_failure(self, error: str):
        """Record failed operation"""
        self.total_requests += 1
        self.failed_requests += 1
        self.consecutive_failures += 1
        self.consecutive_errors += 1
        self.last_failure = datetime.now(timezone.utc)
        self.last_error = error
        self._update_uptime()
    
    def record_stream_update(self):
        """Record stream data update"""
        self.last_stream_update = datetime.now(timezone.utc)
    
    def _update_uptime(self):
        """Calculate uptime percentage"""
        if self.total_requests > 0:
            self.uptime_percent = (self.successful_requests / self.total_requests) * 100


# ======================================================================
# Stream Management
# ======================================================================

@dataclass
class StreamSession:
    """Represents an active data stream"""
    symbol: str
    timeframe: str
    callback: Callable[[Dict[str, Any]], None]
    poll_interval: float = 1.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_update: Optional[datetime] = None
    update_count: int = 0
    _stop_event: Optional[threading.Event] = None
    _thread: Optional[threading.Thread] = None
    
    def update_timestamp(self, bar_time: Optional[datetime] = None):
        """Update last activity timestamp"""
        self.last_update = datetime.now(timezone.utc)
        self.update_count += 1
    
    def get_staleness_seconds(self) -> float:
        """Get seconds since last update"""
        if self.last_update is None:
            return float('inf')
        return (datetime.now(timezone.utc) - self.last_update).total_seconds()
    
    def stop(self):
        """Stop the stream"""
        if self._stop_event:
            self._stop_event.set()


class StreamHandle:
    """Handle for controlling a stream"""
    def __init__(self, session: StreamSession, manager: 'FallbackManager'):
        self.session = session
        self.manager = manager
        
    def stop(self):
        """Stop this stream"""
        self.manager.stop_stream(self.session)
        
    def is_active(self) -> bool:
        """Check if stream is still active"""
        return self.session in self.manager._streams


# ======================================================================
# Base Data Source Interface
# ======================================================================

class DataSource(ABC):
    """Abstract base class for all data sources"""
    
    def __init__(self, source_type: DataSourceType, config: DataSourceConfig):
        self.source_type = source_type
        self.config = config
        self.status = SourceStatus.DISABLED
        self.metrics = SourceMetrics()
        self._lock = threading.RLock()
        self._logger = logging.getLogger(f"{__name__}.{source_type.value}")
        
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the data source"""
        pass
    
    @abstractmethod
    async def check_health(self) -> Tuple[bool, Optional[str]]:
        """Check source health. Returns (is_healthy, error_message)"""
        pass
    
    @abstractmethod
    async def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start: datetime, 
        end: datetime
    ) -> pd.DataFrame:
        """Fetch historical data"""
        pass
    
    @abstractmethod
    async def get_realtime_data(
        self, 
        symbol: str, 
        timeframe: str,
        count: int = 1
    ) -> pd.DataFrame:
        """Fetch latest data bars"""
        pass
    
    @abstractmethod
    async def shutdown(self):
        """Clean shutdown"""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get source information and metrics"""
        with self._lock:
            return {
                'type': self.source_type.value,
                'status': self.status.value,
                'priority': self.config.priority,
                'metrics': {
                    'uptime': round(self.metrics.uptime_percent, 2),
                    'requests': self.metrics.total_requests,
                    'failures': self.metrics.failed_requests,
                    'consecutive_errors': self.metrics.consecutive_errors,
                    'avg_latency_ms': round(self.metrics.average_latency_ms, 2),
                    'last_success': self.metrics.last_success.isoformat() if self.metrics.last_success else None,
                    'last_error': self.metrics.last_error
                }
            }


# ======================================================================
# MT5 Data Source Implementation
# ======================================================================

class MT5DataSource(DataSource):
    """MetaTrader 5 data source"""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(DataSourceType.MT5, config)
        mt5_cfg = ConnectorConfig(**(config.mt5_config or {}))
        self.connector = RobustMT5Connector(mt5_cfg)
        
    async def initialize(self) -> bool:
        try:
            self.status = SourceStatus.INITIALIZING
            # Run MT5 initialization in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.connector.initialize)
            
            if self.connector.is_connected():
                self.status = SourceStatus.STANDBY
                self._logger.info("MT5 initialized successfully")
                return True
            else:
                self.status = SourceStatus.FAILED
                self._logger.error("MT5 initialization failed - not connected")
                return False
                
        except Exception as e:
            self._logger.error(f"MT5 initialization error: {e}")
            self.status = SourceStatus.FAILED
            self.metrics.record_failure(str(e))
            return False
    
    async def check_health(self) -> Tuple[bool, Optional[str]]:
        try:
            health = self.connector.health()
            is_healthy = health.get('connected', False)
            error = health.get('last_error')
            return is_healthy, error
        except Exception as e:
            return False, str(e)
    
    async def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start: datetime, 
        end: datetime
    ) -> pd.DataFrame:
        start_time = time.time()
        try:
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None, 
                self.connector.get_rates_range, 
                symbol, 
                timeframe, 
                start, 
                end
            )
            
            latency = (time.time() - start_time) * 1000
            self.metrics.record_success(latency)
            return df
            
        except Exception as e:
            self.metrics.record_failure(str(e))
            raise
    
    async def get_realtime_data(
        self, 
        symbol: str, 
        timeframe: str,
        count: int = 1
    ) -> pd.DataFrame:
        start_time = time.time()
        try:
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None,
                self.connector.get_rates_count,
                symbol,
                timeframe,
                count
            )
            
            latency = (time.time() - start_time) * 1000
            self.metrics.record_success(latency)
            return df
            
        except Exception as e:
            self.metrics.record_failure(str(e))
            raise
    
    async def shutdown(self):
        try:
            self.connector.shutdown()
            self.status = SourceStatus.DISABLED
        except Exception as e:
            self._logger.error(f"Shutdown error: {e}")


# ======================================================================
# CCXT Data Source Implementation
# ======================================================================

class CCXTDataSource(DataSource):
    """Cryptocurrency exchange data via CCXT"""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(DataSourceType.CCXT, config)
        self.exchange_name = config.ccxt_exchange
        self.exchange = None
        self.symbol_mapping = {
            'USDCOP': 'USD/COP',
            'EURUSD': 'EUR/USD',
            'GBPUSD': 'GBP/USD',
            'USDJPY': 'USD/JPY',
            'AUDUSD': 'AUD/USD'
        }
    
    async def initialize(self) -> bool:
        if not CCXT_AVAILABLE:
            self._logger.error("CCXT module not available")
            self.status = SourceStatus.DISABLED
            return False
            
        try:
            self.status = SourceStatus.INITIALIZING
            
            # Create exchange instance
            exchange_class = getattr(ccxt, self.exchange_name)
            self.exchange = exchange_class(self.config.ccxt_config or {
                'enableRateLimit': True,
                'timeout': 30000
            })
            
            # Check capabilities
            if not self.exchange.has['fetchOHLCV']:
                raise ValueError(f"{self.exchange_name} doesn't support OHLCV data")
            
            # Load markets
            await self.exchange.load_markets()
            
            self.status = SourceStatus.STANDBY
            self._logger.info(f"CCXT {self.exchange_name} initialized")
            return True
            
        except Exception as e:
            self._logger.error(f"CCXT initialization failed: {e}")
            self.status = SourceStatus.FAILED
            self.metrics.record_failure(str(e))
            return False
    
    async def check_health(self) -> Tuple[bool, Optional[str]]:
        try:
            # Simple health check - fetch ticker
            await self.exchange.fetch_ticker('BTC/USDT')
            return True, None
        except Exception as e:
            return False, str(e)
    
    def _convert_symbol(self, symbol: str) -> str:
        """Convert MT5 symbol to CCXT format"""
        return self.symbol_mapping.get(symbol, symbol)
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert timeframe to CCXT format"""
        mapping = {
            'M1': '1m', 'M5': '5m', 'M15': '15m', 'M30': '30m',
            'H1': '1h', 'H4': '4h', 'D1': '1d', 'W1': '1w'
        }
        return mapping.get(timeframe.upper(), timeframe.lower())
    
    async def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start: datetime, 
        end: datetime
    ) -> pd.DataFrame:
        start_time = time.time()
        try:
            ccxt_symbol = self._convert_symbol(symbol)
            ccxt_timeframe = self._convert_timeframe(timeframe)
            
            # CCXT uses millisecond timestamps
            since = int(start.timestamp() * 1000)
            end_ms = int(end.timestamp() * 1000)
            limit = 1000  # Most exchanges limit
            
            all_ohlcv = []
            
            while since < end_ms:
                ohlcv = await self.exchange.fetch_ohlcv(
                    ccxt_symbol, 
                    ccxt_timeframe, 
                    since, 
                    limit
                )
                
                if not ohlcv:
                    break
                    
                all_ohlcv.extend(ohlcv)
                
                # Move to next batch
                since = ohlcv[-1][0] + 1
                
                # Rate limit respect
                await asyncio.sleep(self.exchange.rateLimit / 1000)
            
            # Convert to DataFrame
            df = pd.DataFrame(
                all_ohlcv,
                columns=['time', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert time to datetime
            df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
            
            # Filter to requested range
            df = df[(df['time'] >= start) & (df['time'] <= end)]
            
            # Add MT5-compatible columns
            df['tick_volume'] = df['volume']
            df['spread'] = 0
            df['real_volume'] = df['volume']
            
            latency = (time.time() - start_time) * 1000
            self.metrics.record_success(latency)
            
            return df
            
        except Exception as e:
            self.metrics.record_failure(str(e))
            raise
    
    async def get_realtime_data(
        self, 
        symbol: str, 
        timeframe: str,
        count: int = 1
    ) -> pd.DataFrame:
        # Calculate time range
        end = datetime.now(timezone.utc)
        tf_minutes = {
            'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30, 
            'H1': 60, 'H4': 240, 'D1': 1440
        }
        minutes = tf_minutes.get(timeframe.upper(), 5) * (count + 1)
        start = end - timedelta(minutes=minutes)
        
        df = await self.get_historical_data(symbol, timeframe, start, end)
        return df.tail(count)
    
    async def shutdown(self):
        try:
            if self.exchange:
                await self.exchange.close()
            self.status = SourceStatus.DISABLED
        except Exception as e:
            self._logger.error(f"Shutdown error: {e}")


# ======================================================================
# Simulator Data Source Implementation
# ======================================================================

class SimulatorDataSource(DataSource):
    """Simulated data source for testing and fallback"""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(DataSourceType.SIMULATOR, config)
        sim_cfg = SimulatorConfig(**(config.simulator_config or {}))
        self.simulator = DataSimulator(sim_cfg)
        self.config.always_available = True
        
    async def initialize(self) -> bool:
        self.status = SourceStatus.STANDBY
        self._logger.info("Simulator initialized (always available)")
        return True
    
    async def check_health(self) -> Tuple[bool, Optional[str]]:
        return True, None  # Always healthy
    
    async def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start: datetime, 
        end: datetime
    ) -> pd.DataFrame:
        start_time = time.time()
        try:
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None,
                self.simulator.generate_historical,
                symbol,
                timeframe,
                start,
                end
            )
            
            latency = (time.time() - start_time) * 1000
            self.metrics.record_success(latency)
            return df
            
        except Exception as e:
            self.metrics.record_failure(str(e))
            raise
    
    async def get_realtime_data(
        self, 
        symbol: str, 
        timeframe: str,
        count: int = 1
    ) -> pd.DataFrame:
        end = datetime.now(timezone.utc)
        tf_minutes = {
            'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
            'H1': 60, 'H4': 240, 'D1': 1440
        }
        minutes = tf_minutes.get(timeframe.upper(), 5) * (count + 1)
        start = end - timedelta(minutes=minutes)
        
        df = await self.get_historical_data(symbol, timeframe, start, end)
        return df.tail(count)
    
    async def shutdown(self):
        self.status = SourceStatus.DISABLED


# ======================================================================
# Main Fallback Manager
# ======================================================================

class FallbackManager:
    """
    Production-ready fallback manager for 24/7 data acquisition.
    
    Features:
    - Multiple data sources with priority-based selection
    - Circuit breaker pattern for failure handling
    - Automatic stream recovery on source switch
    - Comprehensive health monitoring
    - Event callbacks for external monitoring
    - Thread-safe operations
    """
    
    def __init__(self, config: Dict[str, DataSourceConfig]):
        """
        Initialize fallback manager.
        
        Args:
            config: Dictionary mapping source names to DataSourceConfig
        """
        self.config = config
        self.sources: Dict[str, DataSource] = {}
        self.active_source: Optional[DataSource] = None
        
        # State management
        self._lock = threading.RLock()
        self._running = False
        self._circuit_state = CircuitState.CLOSED
        self._last_circuit_change = datetime.now(timezone.utc)
        self._mode_override: Optional[DataSourceType] = None
        
        # Monitoring
        self._health_task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None
        
        # Streams
        self._streams: List[StreamSession] = []
        
        # Events and callbacks
        self.event_history = deque(maxlen=1000)
        self._event_callbacks: List[Callable] = []
        
        # Event bus integration
        self._event_bus = event_bus if EVENT_BUS_AVAILABLE else None
        
        # Metrics
        self._start_time = datetime.now(timezone.utc)
        
        self.logger = logger
    
    # ==================== Initialization ====================
    
    async def initialize(self):
        """Initialize all configured data sources"""
        self._running = True
        self.logger.info("Initializing FallbackManager")
        
        # Create source instances
        for name, cfg in self.config.items():
            if not cfg.enabled:
                continue
                
            try:
                if name == 'mt5':
                    self.sources['mt5'] = MT5DataSource(cfg)
                elif name == 'ccxt':
                    self.sources['ccxt'] = CCXTDataSource(cfg)
                elif name == 'simulator':
                    self.sources['simulator'] = SimulatorDataSource(cfg)
                else:
                    self.logger.warning(f"Unknown source type: {name}")
            except Exception as e:
                self.logger.error(f"Failed to create source {name}: {e}")
        
        # Initialize sources in parallel
        init_tasks = []
        for name, source in self.sources.items():
            init_tasks.append(self._init_source(name, source))
        
        results = await asyncio.gather(*init_tasks, return_exceptions=True)
        
        # Select initial active source
        await self._select_active_source()
        
        # Start monitoring tasks
        self._health_task = asyncio.create_task(self._health_monitor())
        self._monitor_task = asyncio.create_task(self._stream_monitor())
        
        self.logger.info(
            f"FallbackManager initialized with {len(self.sources)} sources. "
            f"Active: {self.active_source.source_type.value if self.active_source else 'None'}"
        )
    
    async def _init_source(self, name: str, source: DataSource) -> bool:
        """Initialize individual source"""
        try:
            success = await source.initialize()
            
            if success:
                self._record_event('source_initialized', {
                    'source': name,
                    'type': source.source_type.value
                })
                
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to initialize {name}: {e}")
            self._record_event('source_init_failed', {
                'source': name,
                'error': str(e)
            })
            return False
    
    # ==================== Source Selection ====================
    
    async def _select_active_source(self):
        """Select active source based on priority and availability"""
        with self._lock:
            # Check manual override
            if self._mode_override:
                for source in self.sources.values():
                    if source.source_type == self._mode_override:
                        if source.status in [SourceStatus.ACTIVE, SourceStatus.STANDBY]:
                            await self._activate_source(source, "manual_override")
                            return
            
            # Get available sources sorted by priority
            available = [
                s for s in self.sources.values()
                if s.status in [SourceStatus.ACTIVE, SourceStatus.STANDBY]
            ]
            available.sort(key=lambda x: x.config.priority)
            
            if available:
                await self._activate_source(available[0], "priority_selection")
            else:
                self.logger.error("No available data sources!")
                self.active_source = None
    
    async def _activate_source(self, source: DataSource, reason: str):
        """Activate a specific source"""
        old_source = self.active_source
        
        if old_source == source:
            return  # Already active
            
        # Deactivate old source
        if old_source and old_source.status == SourceStatus.ACTIVE:
            old_source.status = SourceStatus.STANDBY
        
        # Activate new source
        self.active_source = source
        source.status = SourceStatus.ACTIVE
        
        # Record event
        self._record_event('source_switched', {
            'from': old_source.source_type.value if old_source else None,
            'to': source.source_type.value,
            'reason': reason
        })
        
        # Restart streams if needed
        if old_source and self._streams:
            await self._restart_all_streams()
        
        # Notify callbacks
        await self._notify_callbacks('source_changed', {
            'old': old_source.get_info() if old_source else None,
            'new': source.get_info()
        })
    
    # ==================== Circuit Breaker ====================
    
    def _update_circuit_state(self, new_state: CircuitState):
        """Update circuit breaker state"""
        if self._circuit_state != new_state:
            self._circuit_state = new_state
            self._last_circuit_change = datetime.now(timezone.utc)
            
            self._record_event('circuit_state_changed', {
                'from': self._circuit_state.value,
                'to': new_state.value
            })
    
    async def _handle_source_failure(self, source: DataSource, error: str):
        """Handle source failure with circuit breaker logic"""
        source.metrics.record_failure(error)
        
        # Check if we should mark source as failed
        if source.metrics.consecutive_failures >= source.config.max_failures:
            source.status = SourceStatus.FAILED
            self._update_circuit_state(CircuitState.OPEN)
            
            self._record_event('source_failed', {
                'source': source.source_type.value,
                'consecutive_failures': source.metrics.consecutive_failures,
                'error': error
            })
            
            # Switch to next available source
            if source == self.active_source:
                await self._select_active_source()
    
    # ==================== Health Monitoring ====================
    
    async def _health_monitor(self):
        """Monitor health of all sources"""
        while self._running:
            try:
                check_tasks = []
                
                for name, source in self.sources.items():
                    if source.config.enabled:
                        check_tasks.append(self._check_source_health(name, source))
                
                await asyncio.gather(*check_tasks, return_exceptions=True)
                
                # Handle circuit breaker state transitions
                await self._update_circuit_breaker()
                
                # Re-evaluate active source
                await self._select_active_source()
                
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
            
            # Wait for next check
            await asyncio.sleep(30)
    
    async def _check_source_health(self, name: str, source: DataSource):
        """Check individual source health"""
        try:
            # Skip if in cooldown
            if source.status == SourceStatus.FAILED:
                if source.metrics.last_failure:
                    cooldown_end = source.metrics.last_failure + timedelta(
                        minutes=source.config.cooldown_minutes
                    )
                    if datetime.now(timezone.utc) < cooldown_end:
                        return
            
            # Perform health check
            is_healthy, error = await source.check_health()
            
            if is_healthy:
                # Handle recovery
                if source.status == SourceStatus.FAILED:
                    source.status = SourceStatus.STANDBY
                    source.metrics.consecutive_failures = 0
                    
                    self._record_event('source_recovered', {
                        'source': name,
                        'type': source.source_type.value
                    })
            else:
                # Handle failure
                await self._handle_source_failure(source, error or "Health check failed")
                
        except Exception as e:
            self.logger.error(f"Health check error for {name}: {e}")
    
    async def _update_circuit_breaker(self):
        """Update circuit breaker state based on conditions"""
        now = datetime.now(timezone.utc)
        time_since_change = (now - self._last_circuit_change).total_seconds()
        
        if self._circuit_state == CircuitState.OPEN:
            # Check if cooldown has passed
            cooldown = self.config.get('mt5', DataSourceConfig()).circuit_cooldown_seconds
            if time_since_change >= cooldown:
                self._update_circuit_state(CircuitState.HALF_OPEN)
                
        elif self._circuit_state == CircuitState.HALF_OPEN:
            # Try to recover primary source
            primary_source = self._get_primary_source()
            if primary_source and primary_source.status == SourceStatus.STANDBY:
                # Test with probe
                success = await self._probe_source(primary_source)
                if success:
                    self._update_circuit_state(CircuitState.CLOSED)
                    await self._activate_source(primary_source, "circuit_closed")
                else:
                    self._update_circuit_state(CircuitState.OPEN)
    
    async def _probe_source(self, source: DataSource) -> bool:
        """Test if source is working with minimal request"""
        try:
            # Try to get just one recent bar
            df = await source.get_realtime_data('EURUSD', 'M1', count=1)
            return df is not None and not df.empty
        except Exception:
            return False
    
    def _get_primary_source(self) -> Optional[DataSource]:
        """Get highest priority source"""
        sources = list(self.sources.values())
        sources.sort(key=lambda x: x.config.priority)
        return sources[0] if sources else None
    
    # ==================== Stream Management ====================
    
    async def _stream_monitor(self):
        """Monitor stream staleness"""
        while self._running:
            try:
                now = datetime.now(timezone.utc)
                
                for stream in list(self._streams):
                    staleness = stream.get_staleness_seconds()
                    
                    if staleness > self.active_source.config.max_staleness_seconds:
                        self.logger.warning(
                            f"Stream stale: {stream.symbol} {stream.timeframe} "
                            f"({staleness:.1f}s)"
                        )
                        
                        # Record staleness event
                        self._record_event('stream_stale', {
                            'symbol': stream.symbol,
                            'timeframe': stream.timeframe,
                            'staleness_seconds': staleness
                        })
                        
                        # If active source has stale streams, consider switching
                        if self.active_source and len(self._streams) > 0:
                            stale_count = sum(
                                1 for s in self._streams 
                                if s.get_staleness_seconds() > 
                                self.active_source.config.max_staleness_seconds
                            )
                            
                            if stale_count >= len(self._streams) * 0.5:
                                await self._handle_source_failure(
                                    self.active_source, 
                                    f"Too many stale streams ({stale_count}/{len(self._streams)})"
                                )
                
            except Exception as e:
                self.logger.error(f"Stream monitor error: {e}")
            
            await asyncio.sleep(10)  # Check every 10 seconds
    
    async def start_stream(
        self,
        symbol: str,
        timeframe: str,
        callback: Callable[[Dict[str, Any]], None],
        poll_interval: float = 1.0
    ) -> StreamHandle:
        """
        Start a data stream with automatic recovery.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            callback: Function to call with new data
            poll_interval: Seconds between polls
            
        Returns:
            StreamHandle for controlling the stream
        """
        session = StreamSession(
            symbol=symbol,
            timeframe=timeframe,
            callback=callback,
            poll_interval=poll_interval
        )
        
        with self._lock:
            self._streams.append(session)
            
        # Start the stream
        await self._start_stream_session(session)
        
        # Record event
        self._record_event('stream_started', {
            'symbol': symbol,
            'timeframe': timeframe,
            'source': self.active_source.source_type.value if self.active_source else None
        })
        
        return StreamHandle(session, self)
    
    async def _start_stream_session(self, session: StreamSession):
        """Start streaming for a session"""
        if not self.active_source:
            self.logger.error("No active source for streaming")
            return
            
        stop_event = threading.Event()
        session._stop_event = stop_event
        
        # Create wrapped callback that updates metrics
        def wrapped_callback(bar: Dict[str, Any]):
            try:
                session.update_timestamp()
                self.active_source.metrics.record_stream_update()
                session.callback(bar)
            except Exception as e:
                self.logger.error(f"Stream callback error: {e}")
        
        # Start stream thread
        async def stream_loop():
            while not stop_event.is_set() and self._running:
                try:
                    # Get latest data
                    df = await self.active_source.get_realtime_data(
                        session.symbol,
                        session.timeframe,
                        count=1
                    )
                    
                    if not df.empty:
                        # Convert to dict and call callback
                        bar = df.iloc[-1].to_dict()
                        wrapped_callback(bar)
                    
                except Exception as e:
                    self.logger.error(f"Stream error for {session.symbol}: {e}")
                    
                # Wait for next poll
                await asyncio.sleep(session.poll_interval)
        
        # Start async task
        session._thread = asyncio.create_task(stream_loop())
    
    def stop_stream(self, session: StreamSession):
        """Stop a stream session"""
        with self._lock:
            if session in self._streams:
                self._streams.remove(session)
                
        if session._stop_event:
            session._stop_event.set()
            
        if session._thread:
            session._thread.cancel()
        
        # Record event
        self._record_event('stream_stopped', {
            'symbol': session.symbol,
            'timeframe': session.timeframe,
            'updates': session.update_count
        })
    
    async def _restart_all_streams(self):
        """Restart all active streams (after source switch)"""
        self.logger.info(f"Restarting {len(self._streams)} streams")
        
        for session in list(self._streams):
            try:
                # Stop old stream
                if session._stop_event:
                    session._stop_event.set()
                if session._thread:
                    session._thread.cancel()
                
                # Start new stream
                await self._start_stream_session(session)
                
            except Exception as e:
                self.logger.error(f"Failed to restart stream {session.symbol}: {e}")
    
    # ==================== Data Access API ====================
    
    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        retry_count: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get historical data with automatic fallback.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe (M1, M5, H1, etc.)
            start: Start datetime (UTC)
            end: End datetime (UTC)
            retry_count: Override retry count
            
        Returns:
            DataFrame with OHLCV data
            
        Raises:
            RuntimeError: If all sources fail
        """
        if retry_count is None:
            retry_count = self.config.get('mt5', DataSourceConfig()).retry_count
            
        attempted_sources = []
        last_error = None
        
        # Try each available source
        for _ in range(len(self.sources)):
            with self._lock:
                if not self.active_source:
                    await self._select_active_source()
                    if not self.active_source:
                        raise RuntimeError("No available data sources")
                
                current_source = self.active_source
            
            # Skip if already tried
            if current_source in attempted_sources:
                break
                
            attempted_sources.append(current_source)
            
            # Attempt with retries
            for retry in range(retry_count):
                try:
                    df = await current_source.get_historical_data(
                        symbol, timeframe, start, end
                    )
                    
                    # Validate data
                    if df is not None and not df.empty:
                        return df
                    else:
                        raise ValueError("Empty data returned")
                        
                except Exception as e:
                    last_error = e
                    self.logger.warning(
                        f"Data fetch failed for {current_source.source_type.value} "
                        f"(attempt {retry+1}/{retry_count}): {e}"
                    )
                    
                    if retry < retry_count - 1:
                        # Exponential backoff
                        delay = current_source.config.retry_delay_base * (2 ** retry)
                        await asyncio.sleep(delay)
            
            # Source failed all retries
            await self._handle_source_failure(current_source, str(last_error))
        
        # All sources failed
        raise RuntimeError(
            f"All data sources failed. Last error: {last_error}"
        )
    
    async def get_realtime_data(
        self,
        symbol: str,
        timeframe: str,
        count: int = 1
    ) -> pd.DataFrame:
        """
        Get realtime/recent data with automatic fallback.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            count: Number of recent bars
            
        Returns:
            DataFrame with recent OHLCV data
        """
        return await self._get_data_with_fallback(
            lambda source: source.get_realtime_data(symbol, timeframe, count),
            "realtime_data"
        )
    
    def fetch_with_fallback(self, *args, **kwargs) -> Optional[pd.DataFrame]:
        """Sync wrapper for async data fetching with fallback (test compatibility)"""
        import asyncio
        
        # Handle the case where no sources are configured (for testing)
        if not self.sources and not self.config:
            # Use simulator as default for testing
            from src.core.data.data_simulator import DataSimulator
            sim = DataSimulator()
            # Use generate_data method which accepts parameters
            return sim.generate_data(*args, **kwargs)
        
        loop = asyncio.new_event_loop()
        try:
            # Initialize if not already initialized
            if not self.sources:
                loop.run_until_complete(self.initialize())
            
            return loop.run_until_complete(self._get_data_with_fallback(
                lambda source: source.get_historical_data(*args, **kwargs),
                "fetch_data"
            ))
        finally:
            loop.close()
    
    async def _get_data_with_fallback(
        self,
        fetch_func: Callable[[DataSource], Any],
        operation_name: str
    ) -> Any:
        """Generic method for operations with fallback"""
        attempted_sources = []
        last_error = None
        
        for _ in range(len(self.sources)):
            with self._lock:
                if not self.active_source:
                    await self._select_active_source()
                    if not self.active_source:
                        raise RuntimeError("No available data sources")
                
                current_source = self.active_source
            
            if current_source in attempted_sources:
                break
                
            attempted_sources.append(current_source)
            
            try:
                result = await fetch_func(current_source)
                if result is not None:
                    return result
                    
            except Exception as e:
                last_error = e
                self.logger.warning(
                    f"{operation_name} failed for {current_source.source_type.value}: {e}"
                )
                await self._handle_source_failure(current_source, str(e))
        
        raise RuntimeError(
            f"All sources failed for {operation_name}. Last error: {last_error}"
        )
    
    # ==================== Control API ====================
    
    def set_mode(self, mode: Optional[DataSourceType] = None):
        """
        Force specific mode or return to automatic.
        
        Args:
            mode: DataSourceType to force, or None for automatic
        """
        with self._lock:
            self._mode_override = mode
            
            self._record_event('mode_changed', {
                'mode': mode.value if mode else 'AUTO',
                'previous': self._mode_override.value if self._mode_override else 'AUTO'
            })
        
        # Trigger source re-selection
        asyncio.create_task(self._select_active_source())
    
    def force_simulator(self, enable: bool = True):
        """
        Force simulator mode on/off.
        
        Args:
            enable: True to force simulator, False to return to auto
        """
        self.set_mode(DataSourceType.SIMULATOR if enable else None)
    
    # ==================== Monitoring API ====================
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        with self._lock:
            uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()
            
            status = {
                'active_source': self.active_source.source_type.value if self.active_source else None,
                'mode': self._mode_override.value if self._mode_override else 'AUTO',
                'circuit_state': self._circuit_state.value,
                'uptime_seconds': uptime,
                'sources': {},
                'streams': {
                    'active': len(self._streams),
                    'details': [
                        {
                            'symbol': s.symbol,
                            'timeframe': s.timeframe,
                            'staleness': s.get_staleness_seconds(),
                            'updates': s.update_count
                        }
                        for s in self._streams
                    ]
                },
                'events': {
                    'total': len(self.event_history),
                    'recent': list(self.event_history)[-10:]
                }
            }
            
            # Add source details
            for name, source in self.sources.items():
                status['sources'][name] = source.get_info()
            
            # Calculate overall availability
            total_requests = sum(
                s.metrics.total_requests for s in self.sources.values()
            )
            successful_requests = sum(
                s.metrics.successful_requests for s in self.sources.values()
            )
            
            status['overall_availability'] = (
                (successful_requests / total_requests * 100) if total_requests > 0 else 0
            )
            
            return status
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export metrics for external monitoring (Prometheus, etc.)"""
        metrics = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'fallback_manager_info': {
                'active_source': self.active_source.source_type.value if self.active_source else None,
                'circuit_state': self._circuit_state.value,
                'mode': self._mode_override.value if self._mode_override else 'AUTO',
                'stream_count': len(self._streams)
            },
            'sources': {}
        }
        
        # Add per-source metrics
        for name, source in self.sources.items():
            metrics['sources'][name] = {
                'status': source.status.value,
                'total_requests': source.metrics.total_requests,
                'success_rate': (
                    source.metrics.successful_requests / source.metrics.total_requests * 100
                    if source.metrics.total_requests > 0 else 0
                ),
                'avg_latency_ms': source.metrics.average_latency_ms,
                'consecutive_failures': source.metrics.consecutive_failures,
                'uptime_percent': source.metrics.uptime_percent
            }
        
        return metrics
    
    # ==================== Event Management ====================
    
    def add_event_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """
        Add callback for system events.
        
        Args:
            callback: Function(event_type, event_data)
        """
        self._event_callbacks.append(callback)
    
    def remove_event_callback(self, callback: Callable):
        """Remove event callback"""
        if callback in self._event_callbacks:
            self._event_callbacks.remove(callback)
    
    async def _notify_callbacks(self, event_type: str, data: Dict[str, Any]):
        """Notify all callbacks of an event"""
        for callback in list(self._event_callbacks):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_type, data)
                else:
                    callback(event_type, data)
            except Exception as e:
                self.logger.error(f"Event callback error: {e}")
    
    def _record_event(self, event_type: str, data: Dict[str, Any]):
        """Record event in history"""
        event = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'type': event_type,
            'data': data
        }
        
        self.event_history.append(event)
        self.logger.info(f"Event: {event_type} - {data}")
        
        # Publish standardized event to event bus
        if self._event_bus and EVENT_BUS_AVAILABLE:
            try:
                event_obj = Event(
                    event=self._map_event_name(event_type),
                    source="fallback_manager",
                    ts=datetime.now(timezone.utc).isoformat(),
                    correlation_id=get_correlation_id() or "",
                    payload=data,
                    metadata={
                        "internal_type": event_type,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
                
                self._event_bus.publish(event_obj)
                
            except Exception as e:
                self.logger.warning(f"Event bus publishing failed: {e}")
    
    def _map_event_name(self, internal: str) -> str:
        """Map internal event names to standardized names"""
        mapping = {
            "source_activated": EventType.SOURCE_CHANGED.value,
            "source_recovered": EventType.SOURCE_CHANGED.value,
            "circuit_open": EventType.CIRCUIT_OPEN.value,
            "circuit_half_open": EventType.CIRCUIT_HALF_OPEN.value,
            "circuit_closed": EventType.CIRCUIT_CLOSED.value,
            "stream_stale": EventType.STREAM_STALE.value
        }
        return mapping.get(internal, internal)
    
    def get_event_history(self, 
                         event_type: Optional[str] = None,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get event history.
        
        Args:
            event_type: Filter by event type
            limit: Maximum events to return
            
        Returns:
            List of events
        """
        events = list(self.event_history)
        
        if event_type:
            events = [e for e in events if e['type'] == event_type]
            
        return events[-limit:]
    
    # ==================== Lifecycle ====================
    
    async def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Shutting down FallbackManager")
        self._running = False
        
        # Stop all streams
        for stream in list(self._streams):
            self.stop_stream(stream)
        
        # Cancel monitoring tasks
        for task in [self._health_task, self._monitor_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Shutdown all sources
        shutdown_tasks = []
        for source in self.sources.values():
            shutdown_tasks.append(source.shutdown())
        
        await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        
        # Final event
        self._record_event('manager_shutdown', {
            'total_events': len(self.event_history),
            'runtime_seconds': (datetime.now(timezone.utc) - self._start_time).total_seconds()
        })
        
        self.logger.info("FallbackManager shutdown complete")
    
    # ==================== Factory Methods ====================
    
    @staticmethod
    def create_default_config() -> Dict[str, DataSourceConfig]:
        """Create default configuration for quick setup"""
        return {
            'mt5': DataSourceConfig(
                priority=1,
                enabled=True,
                max_failures=3,
                cooldown_minutes=5,
                mt5_config={
                    'server': os.getenv('MT5_SERVER'),
                    'login': int(os.getenv('MT5_LOGIN', '0')) or None,
                    'password': os.getenv('MT5_PASSWORD'),
                    'path': os.getenv('MT5_PATH'),
                    'timeout': 30000
                }
            ),
            'ccxt': DataSourceConfig(
                priority=2,
                enabled=CCXT_AVAILABLE,
                max_failures=3,
                cooldown_minutes=5,
                ccxt_exchange=os.getenv('CCXT_EXCHANGE', 'binance'),
                ccxt_config={
                    'apiKey': os.getenv('CCXT_API_KEY'),
                    'secret': os.getenv('CCXT_SECRET'),
                    'enableRateLimit': True
                }
            ),
            'simulator': DataSourceConfig(
                priority=3,
                enabled=True,
                always_available=True,
                simulator_config={
                    'initial_price': 4000.0,  # USDCOP
                    'volatility': 0.1,
                    'trend': 0.00001,
                    'seed': 42
                }
            )
        }


# ======================================================================
# Compatibility Shims for Dashboard
# ======================================================================

def get_rates_count(self, symbol: str, timeframe: str) -> int:
    """Get count of available rates - compatibility wrapper for dashboard"""
    if self.current_source:
        return self.current_source.get_rates_count(symbol, timeframe)
    return 0

@property
def is_in_fallback_mode(self) -> bool:
    """Check if currently using fallback source - compatibility wrapper for dashboard"""
    return self.current_source != self.primary_source

def health(self) -> dict:
    """Compatibility wrapper for dashboard"""
    return self.get_health_status()

# ======================================================================
# Example Usage
# ======================================================================

async def main():
    """Example usage of the FallbackManager"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create configuration
    config = FallbackManager.create_default_config()
    
    # Create manager
    manager = FallbackManager(config)
    
    # Add event listener
    def on_event(event_type: str, data: Dict[str, Any]):
        print(f"\n📢 EVENT: {event_type}")
        print(f"   Data: {json.dumps(data, indent=2)}")
    
    manager.add_event_callback(on_event)
    
    try:
        # Initialize
        await manager.initialize()
        
        # Check initial health
        health = manager.get_health_status()
        print("\n📊 Initial Health Status:")
        print(json.dumps(health, indent=2))
        
        # Test historical data fetch
        print("\n📈 Fetching historical data...")
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=7)
        
        df = await manager.get_historical_data('USDCOP', 'M5', start, end)
        print(f"Retrieved {len(df)} historical records")
        print(df.head())
        
        # Test realtime data
        print("\n⚡ Fetching realtime data...")
        realtime_df = await manager.get_realtime_data('USDCOP', 'M5', count=5)
        print(f"Retrieved {len(realtime_df)} realtime records")
        print(realtime_df)
        
        # Test streaming
        print("\n📡 Starting data stream...")
        
        def on_bar(bar: Dict[str, Any]):
            print(f"New bar: {bar['time']} - O:{bar['open']:.2f} H:{bar['high']:.2f} "
                  f"L:{bar['low']:.2f} C:{bar['close']:.2f}")
        
        stream_handle = await manager.start_stream('EURUSD', 'M1', on_bar)
        
        # Let it run for a bit
        await asyncio.sleep(10)
        
        # Test mode switching
        print("\n🔄 Switching to simulator mode...")
        manager.force_simulator(True)
        await asyncio.sleep(5)
        
        # Back to auto
        print("\n🔄 Switching back to auto mode...")
        manager.force_simulator(False)
        await asyncio.sleep(5)
        
        # Stop stream
        stream_handle.stop()
        
        # Export metrics
        metrics = manager.export_metrics()
        print("\n📊 Metrics Export:")
        print(json.dumps(metrics, indent=2))
        
        # Event history
        events = manager.get_event_history(limit=20)
        print(f"\n📜 Recent Events ({len(events)}):")
        for event in events[-5:]:
            print(f"  {event['timestamp']}: {event['type']}")
        
    finally:
        # Cleanup
        await manager.shutdown()
        print("\n✅ Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())