"""
dashboard.py
============
Production-ready unified trading dashboard with comprehensive health monitoring.

This implementation combines the best features from all versions:
- Advanced health monitoring with circuit breaker states
- Professional dark theme with responsive design
- Real-time data updates with WebSocket support
- Comprehensive metrics and performance monitoring
- Multi-tab interface with trading, metrics, system, and backtest views
- Export functionality and alert system
- Force simulator mode for 24/7 operation
- Async operation with proper error handling

Dependencies:
- dash>=2.14.0
- dash-bootstrap-components>=1.5.0
- plotly>=5.18.0
- pandas>=2.0.0
- numpy>=1.24.0
- asyncio
- threading
- websocket-client (optional for real-time)

Usage:
    python -m src.visualization.dashboard --host 0.0.0.0 --port 8050
"""

from __future__ import annotations

import os
import sys
import json
import logging
import asyncio
import threading
import time
import sqlite3
import traceback
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from pathlib import Path
from contextlib import contextmanager
from functools import wraps, lru_cache

import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback_context, ALL, MATCH
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

# Setup logger early to avoid NameError
logger = logging.getLogger(__name__)

# Import system components with better error handling
try:
    from src.core.connectors.mt5_connector import MT5ConnectorConfig, MT5Connector
    from src.core.connectors.fallback_manager import FallbackManager, FallbackPolicy, CircuitState
    from src.markets.usdcop.metrics import USDCOPMetrics, TradingMetrics, calculate_metrics
    from src.markets.usdcop.feature_engine import FeatureEngine
    from src.visualization.components.health_monitor import HealthMonitor, HealthMonitorConfig
    from src.core.config.unified_config import UnifiedConfig
    from src.core.database.database_manager import DatabaseManager
    from src.core.observability.metrics.prometheus_registry import PrometheusRegistry
    from src.core.lifecycle.shutdown_manager import ShutdownManager
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    # Fallback imports for standalone testing
    logger.warning(f"Could not import all components: {e}")
    MT5ConnectorConfig = None
    FallbackManager = None
    USDCOPMetrics = None
    FeatureEngine = None
    HealthMonitor = None
    HealthMonitorConfig = None
    UnifiedConfig = None
    DatabaseManager = None
    PrometheusRegistry = None
    ShutdownManager = None
    COMPONENTS_AVAILABLE = False

# ------------------------------------------------------------
# Logging Configuration Enhancement
# ------------------------------------------------------------
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ------------------------------------------------------------
# Configuration and Constants
# ------------------------------------------------------------
@dataclass
class DashboardConfig:
    """Enhanced dashboard configuration"""
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8050
    debug: bool = False
    
    # Trading settings
    symbol: str = "USDCOP"
    timeframe: str = "M5"
    bars: int = 500
    refresh_sec: int = 5
    
    # Features
    enable_websocket: bool = False
    enable_health_monitor: bool = True
    enable_persistence: bool = True
    enable_real_time: bool = True
    enable_export: bool = True
    
    # Performance
    cache_timeout: int = 300
    max_data_points: int = 2000
    update_batch_size: int = 100
    
    # Database
    db_path: str = "./data/dashboard.db"
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file: str = "logs/dashboard.log"
    
    @classmethod
    def from_env(cls) -> "DashboardConfig":
        """Load configuration from environment variables"""
        return cls(
            host=os.getenv("DASH_HOST", "0.0.0.0"),
            port=int(os.getenv("DASH_PORT", "8050")),
            debug=os.getenv("DASH_DEBUG", "false").lower() == "true",
            symbol=os.getenv("DASH_SYMBOL", "USDCOP"),
            timeframe=os.getenv("DASH_TIMEFRAME", "M5"),
            bars=int(os.getenv("DASH_BARS", "500")),
            refresh_sec=int(os.getenv("DASH_REFRESH_SEC", "5")),
            enable_websocket=os.getenv("ENABLE_WEBSOCKET", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "DashboardConfig":
        """Load configuration from YAML file"""
        try:
            import yaml
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
            dashboard_config = data.get('dashboard', {})
            
            return cls(
                host=dashboard_config.get('server', {}).get('host', '0.0.0.0'),
                port=dashboard_config.get('server', {}).get('port', 8050),
                debug=dashboard_config.get('server', {}).get('debug', False),
                refresh_sec=dashboard_config.get('data_refresh', {}).get('refresh_interval', 5),
                enable_real_time=dashboard_config.get('data_refresh', {}).get('real_time', True),
                max_data_points=dashboard_config.get('data_refresh', {}).get('max_data_points', 1000)
            )
        except Exception as e:
            logger.warning(f"Could not load YAML config: {e}, using defaults")
            return cls.from_env()

# Theme configuration
THEME = {
    'dark': {
        'bg_primary': '#0d1117',
        'bg_secondary': '#161b22',
        'bg_card': '#1c2128',
        'text_primary': '#c9d1d9',
        'text_secondary': '#8b949e',
        'border': '#30363d',
        'success': '#2ea043',
        'danger': '#f85149',
        'warning': '#d29922',
        'info': '#58a6ff',
        'purple': '#a371f7',
        'chart_bg': '#0d1117',
        'grid_color': '#30363d'
    }
}

# Custom CSS for professional appearance
CUSTOM_CSS = """
/* Base styles */
.dashboard-container {
    background-color: #0d1117;
    color: #c9d1d9;
    min-height: 100vh;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
}

/* Cards and panels */
.health-card, .metric-card, .control-panel {
    background-color: #1c2128;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 16px;
    transition: all 0.3s ease;
}

.health-card:hover, .metric-card:hover {
    border-color: #58a6ff;
    box-shadow: 0 4px 12px rgba(88, 166, 255, 0.15);
}

/* Metric values */
.metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    margin: 10px 0;
    font-variant-numeric: tabular-nums;
}

.metric-label {
    font-size: 0.875rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Status badges */
.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
}

.status-active { background-color: #2ea043; color: white; }
.status-standby { background-color: #d29922; color: white; }
.status-failed { background-color: #f85149; color: white; }
.status-real { background-color: #58a6ff; color: white; }
.status-sim { background-color: #a371f7; color: white; }

/* Buttons */
.btn-custom {
    background-color: #21262d;
    color: #c9d1d9;
    border: 1px solid #30363d;
    padding: 6px 16px;
    border-radius: 6px;
    font-weight: 500;
    transition: all 0.2s;
}

.btn-custom:hover {
    background-color: #30363d;
    border-color: #58a6ff;
}

/* Logs */
.log-container {
    background-color: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px;
    height: 400px;
    overflow-y: auto;
    font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
    font-size: 13px;
    line-height: 1.5;
}

.log-entry {
    padding: 2px 0;
    border-bottom: 1px solid #21262d;
}

.log-entry:last-child {
    border-bottom: none;
}

/* Alerts */
.alert-toast {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 9999;
    max-width: 400px;
    animation: slideIn 0.3s ease;
}

@keyframes slideIn {
    from { transform: translateX(100%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}

/* Charts */
.chart-container {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 16px;
}

/* Tabs */
.nav-tabs {
    border-bottom: 1px solid #30363d;
}

.nav-tabs .nav-link {
    color: #8b949e;
    border: none;
    padding: 12px 20px;
}

.nav-tabs .nav-link:hover {
    color: #c9d1d9;
}

.nav-tabs .nav-link.active {
    color: #58a6ff;
    background-color: transparent;
    border-bottom: 2px solid #58a6ff;
}

/* Inputs and controls */
.form-control, .form-select {
    background-color: #0d1117;
    border: 1px solid #30363d;
    color: #c9d1d9;
}

.form-control:focus, .form-select:focus {
    background-color: #0d1117;
    border-color: #58a6ff;
    color: #c9d1d9;
    box-shadow: 0 0 0 1px #58a6ff;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

::-webkit-scrollbar-track {
    background: #161b22;
}

::-webkit-scrollbar-thumb {
    background: #30363d;
    border-radius: 5px;
}

::-webkit-scrollbar-thumb:hover {
    background: #484f58;
}

/* Responsive */
@media (max-width: 768px) {
    .metric-value {
        font-size: 1.8rem;
    }
    
    .health-card, .metric-card {
        margin-bottom: 12px;
    }
}
"""

# ------------------------------------------------------------
# Data Management Classes
# ------------------------------------------------------------
@dataclass
class CacheEntry:
    """Cache entry with TTL"""
    data: Any
    timestamp: datetime
    ttl_seconds: int = 300
    
    @property
    def is_expired(self) -> bool:
        return (datetime.now() - self.timestamp).total_seconds() > self.ttl_seconds

class DataCache:
    """Thread-safe data cache with TTL"""
    
    def __init__(self, default_ttl: int = 300):
        self.cache: Dict[str, CacheEntry] = {}
        self.default_ttl = default_ttl
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            entry = self.cache.get(key)
            if entry and not entry.is_expired:
                return entry.data
            elif entry:
                del self.cache[key]
            return None
    
    def set(self, key: str, data: Any, ttl: Optional[int] = None) -> None:
        with self.lock:
            self.cache[key] = CacheEntry(
                data=data,
                timestamp=datetime.now(),
                ttl_seconds=ttl or self.default_ttl
            )
    
    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count removed"""
        with self.lock:
            expired_keys = [k for k, v in self.cache.items() if v.is_expired]
            for k in expired_keys:
                del self.cache[k]
            return len(expired_keys)

class DatabaseConnection:
    """Thread-safe database connection manager"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.local = threading.local()
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize database schema"""
        with self.get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    component TEXT,
                    metadata TEXT
                );
                
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL,
                    exit_price REAL,
                    quantity REAL,
                    pnl REAL,
                    metadata TEXT
                );
                
                CREATE TABLE IF NOT EXISTS system_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    event_type TEXT NOT NULL,
                    severity TEXT,
                    message TEXT,
                    metadata TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp);
                CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
                CREATE INDEX IF NOT EXISTS idx_events_timestamp ON system_events(timestamp);
            """)
    
    @contextmanager
    def get_connection(self):
        """Get thread-local database connection"""
        if not hasattr(self.local, 'conn'):
            self.local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.local.conn.row_factory = sqlite3.Row
        
        try:
            yield self.local.conn
        except Exception:
            self.local.conn.rollback()
            raise
        else:
            self.local.conn.commit()

# ------------------------------------------------------------
# Error Handling and Decorators
# ------------------------------------------------------------
def error_handler(func):
    """Decorator for comprehensive error handling"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            # Return appropriate fallback based on expected return type
            if hasattr(func, '__annotations__'):
                return_type = func.__annotations__.get('return')
                if return_type == go.Figure:
                    return go.Figure()
                elif return_type == list:
                    return []
                elif return_type == dict:
                    return {}
            return None
    return wrapper

def cache_result(ttl: int = 300):
    """Decorator to cache function results"""
    def decorator(func):
        cache = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Check cache
            if cache_key in cache:
                entry_time, result = cache[cache_key]
                if (time.time() - entry_time) < ttl:
                    return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache[cache_key] = (time.time(), result)
            
            # Cleanup old cache entries periodically
            if len(cache) > 100:
                current_time = time.time()
                cache = {k: v for k, v in cache.items() 
                        if (current_time - v[0]) < ttl}
            
            return result
        
        wrapper.clear_cache = lambda: cache.clear()
        return wrapper
    return decorator

# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------
@error_handler
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for the dataframe"""
    if df.empty or len(df) < 20:
        return df
    
    # Price-based indicators
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    # ATR
    df['tr'] = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()
    
    return df

def format_health_status(health: Dict[str, Any]) -> Dict[str, str]:
    """Format health data for display"""
    try:
        mode = health.get("active_mode", "UNKNOWN")
        circuit = health.get("circuit_state", "UNKNOWN")
        forced = health.get("forced_sim", False)
        conn_health = health.get("connector_health", {})
        connected = conn_health.get("connected", False)
        last_switch = health.get("last_switch_ts", None)
        sessions = len(health.get("sessions", []))
        
        # Format last switch time
        if last_switch:
            try:
                dt = datetime.fromisoformat(last_switch.replace('Z', '+00:00'))
                last_switch_str = dt.strftime("%H:%M:%S")
            except:
                last_switch_str = last_switch
        else:
            last_switch_str = "—"
        
        return {
            "mode": str(mode).upper(),
            "circuit": str(circuit).upper(),
            "forced": "YES" if forced else "NO",
            "connected": "YES" if connected else "NO",
            "last_switch": last_switch_str,
            "sessions": str(sessions),
            "latency": f"{conn_health.get('avg_latency_ms', 0):.0f}ms" if connected else "—",
            "uptime": f"{conn_health.get('uptime_pct', 0):.1f}%" if connected else "0%"
        }
    except Exception as e:
        logger.error(f"Error formatting health status: {e}")
        return {
            "mode": "ERROR", "circuit": "ERROR", "forced": "—",
            "connected": "—", "last_switch": "—", "sessions": "0",
            "latency": "—", "uptime": "—"
        }

# ------------------------------------------------------------
# Dashboard Components
# ------------------------------------------------------------
def create_metric_card(title: str, value_id: str, color: str = "text-primary") -> dbc.Col:
    """Create a metric card component"""
    return dbc.Col([
        html.Div([
            html.Div(title, className="metric-label"),
            html.H3(id=value_id, className=f"metric-value {color}")
        ], className="metric-card h-100 text-center")
    ], xs=6, sm=4, md=2, className="mb-3")

def create_health_card(title: str, value_id: str, status_id: str = None) -> dbc.Col:
    """Create a health status card"""
    return dbc.Col([
        html.Div([
            html.Div(title, className="metric-label mb-2"),
            html.Div(id=value_id, className="h5 mb-0"),
            html.Div(id=status_id, className="mt-2") if status_id else None
        ], className="health-card h-100")
    ], xs=12, sm=6, md=4, lg=2, className="mb-3")

# ------------------------------------------------------------
# Main Dashboard Class
# ------------------------------------------------------------
class UnifiedTradingDashboard:
    """Production-ready unified trading dashboard with enhanced features"""
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        """Initialize the dashboard with configuration"""
        self.config = config or DashboardConfig.from_env()
        
        # Initialize enhanced logging
        self._setup_logging()
        
        # Initialize Dash app with production settings
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[
                dbc.themes.DARKLY,
                "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
            ],
            suppress_callback_exceptions=True,
            update_title=None,
            title="USDCOP Trading Dashboard - Production",
            meta_tags=[
                {"name": "viewport", "content": "width=device-width, initial-scale=1"}
            ]
        )
        
        # System components with error handling
        self.fallback_manager = None
        self.mt5_connector = None
        self.health_monitor = None
        self.metrics_calculator = None
        self.feature_engine = None
        self.db_manager = None
        self.shutdown_manager = None
        
        # Enhanced data management
        self.data_cache = DataCache(self.config.cache_timeout)
        self.db_connection = DatabaseConnection(self.config.db_path) if self.config.enable_persistence else None
        
        # Data stores with thread safety
        self.current_data = pd.DataFrame()
        self.health_data = {}
        self.system_logs = deque(maxlen=1000)
        self.alerts_queue = deque(maxlen=50)
        self.predictions = deque(maxlen=100)
        self.performance_metrics = {}
        self._data_lock = threading.RLock()
        
        # State management
        self.selected_symbol = self.config.symbol
        self.selected_timeframe = self.config.timeframe
        self.real_time_mode = self.config.enable_real_time
        self.async_thread = None
        self.cleanup_thread = None
        self.running = False
        self.last_update = datetime.now()
        
        # Performance tracking
        self.update_counts = {'success': 0, 'errors': 0}
        self.response_times = deque(maxlen=100)
        
        # Setup with error handling
        try:
            self._initialize_system()
            self._setup_layout()
            self._setup_callbacks()
            self._start_background_tasks()
            self.logger.info("Dashboard initialized successfully")
        except Exception as e:
            self.logger.error(f"Dashboard initialization failed: {e}")
            raise
        
    def _setup_logging(self):
        """Setup enhanced logging configuration"""
        # Configure dashboard logger
        self.logger = logging.getLogger(f"{__name__}.dashboard")
        self.logger.setLevel(getattr(logging, self.config.log_level))
        
        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '[%(asctime)s] %(levelname)-8s %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
            
            # File handler if enabled
            if self.config.log_to_file:
                try:
                    os.makedirs(os.path.dirname(self.config.log_file), exist_ok=True)
                    file_handler = logging.FileHandler(self.config.log_file)
                    file_handler.setFormatter(console_formatter)
                    self.logger.addHandler(file_handler)
                except Exception as e:
                    self.logger.warning(f"Could not setup file logging: {e}")
        
        # Set global logger reference
        global logger
        logger = self.logger
    
    def _initialize_system(self):
        """Initialize system components with comprehensive error handling"""
        try:
            # Initialize unified config if available
            if UnifiedConfig and COMPONENTS_AVAILABLE:
                try:
                    unified_config = UnifiedConfig.load_config()
                    self.logger.info("Loaded unified configuration")
                except Exception as e:
                    self.logger.warning(f"Could not load unified config: {e}")
                    unified_config = None
            else:
                unified_config = None
            
            # Initialize health monitor
            if HealthMonitor and self.config.enable_health_monitor:
                try:
                    health_config = HealthMonitorConfig.from_yaml() if COMPONENTS_AVAILABLE else None
                    self.health_monitor = HealthMonitor(health_config)
                    self.health_monitor.start()
                    self.logger.info("Health monitor initialized")
                except Exception as e:
                    self.logger.error(f"Health monitor initialization failed: {e}")
            
            # Initialize MT5 components with fallback
            if MT5ConnectorConfig and FallbackManager and COMPONENTS_AVAILABLE:
                try:
                    self._initialize_mt5_components()
                except Exception as e:
                    self.logger.error(f"MT5 initialization failed: {e}")
            
            # Initialize metrics calculator
            if USDCOPMetrics:
                try:
                    self.metrics_calculator = USDCOPMetrics()
                    self.logger.info("Metrics calculator initialized")
                except Exception as e:
                    self.logger.error(f"Metrics calculator initialization failed: {e}")
            
            # Initialize feature engine
            if FeatureEngine:
                try:
                    self.feature_engine = FeatureEngine()
                    self.logger.info("Feature engine initialized")
                except Exception as e:
                    self.logger.error(f"Feature engine initialization failed: {e}")
            
            # Initialize shutdown manager
            if ShutdownManager and COMPONENTS_AVAILABLE:
                try:
                    self.shutdown_manager = ShutdownManager()
                    self.logger.info("Shutdown manager initialized")
                except Exception as e:
                    self.logger.warning(f"Shutdown manager initialization failed: {e}")
            
            # Demo mode fallback
            if not any([self.fallback_manager, self.health_monitor, self.metrics_calculator]):
                self.logger.warning("Running in demo mode - limited functionality available")
            
            self._log_system_event("INFO", "System initialization completed", {
                'components_available': COMPONENTS_AVAILABLE,
                'health_monitor': self.health_monitor is not None,
                'fallback_manager': self.fallback_manager is not None,
                'metrics_calculator': self.metrics_calculator is not None
            })
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            self.logger.debug(f"Initialization traceback: {traceback.format_exc()}")
            raise
    
    def _initialize_mt5_components(self):
        """Initialize MT5-related components"""
        # Get MT5 configuration from environment or config
        mt5_server = os.getenv("MT5_SERVER")
        mt5_login = os.getenv("MT5_LOGIN")
        mt5_password = os.getenv("MT5_PASSWORD")
        mt5_path = os.getenv("MT5_PATH")
        
        if mt5_login and mt5_password:
            mt5_cfg = MT5ConnectorConfig(
                server=mt5_server,
                login=int(mt5_login),
                password=mt5_password,
                path=mt5_path,
                max_retries=5,
                base_backoff_sec=2.0,
                connect_timeout_sec=10.0,
                sim_initial_price=4000.0,
                sim_annual_vol=0.10,
                enforce_ohlc_sanity=True,
            )
            
            # Fallback Policy
            policy = FallbackPolicy(
                max_staleness_sec=120,
                max_consecutive_errors=3,
                monitor_interval_sec=5,
                circuit_cooldown_sec=60,
                half_open_probe_interval_sec=30,
                probe_symbol=self.selected_symbol,
                probe_timeframe=self.selected_timeframe,
                prefer_real=True,
                allow_sim_on_init_failure=True,
            )
            
            # Initialize fallback manager
            self.fallback_manager = FallbackManager(mt5_cfg, policy)
            self.fallback_manager.start()
            self.logger.info("MT5 fallback manager initialized")
        else:
            self.logger.warning("MT5 credentials not provided - running in simulation mode")
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        if self.config.enable_persistence:
            self.cleanup_thread = threading.Thread(
                target=self._cleanup_loop,
                daemon=True,
                name="DashboardCleanup"
            )
            self.cleanup_thread.start()
            self.logger.info("Background cleanup task started")
    
    def _cleanup_loop(self):
        """Background cleanup of cache and database"""
        while self.running:
            try:
                # Clean expired cache entries
                cleaned = self.data_cache.cleanup_expired()
                if cleaned > 0:
                    self.logger.debug(f"Cleaned {cleaned} expired cache entries")
                
                # Database maintenance (if enabled)
                if self.db_connection:
                    self._cleanup_database()
                
                # Sleep for cleanup interval
                time.sleep(300)  # 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                time.sleep(60)  # Wait before retry
    
    def _cleanup_database(self):
        """Clean old database records"""
        try:
            cutoff_date = datetime.now() - timedelta(days=7)
            with self.db_connection.get_connection() as conn:
                # Clean old metrics
                cursor = conn.execute(
                    "DELETE FROM metrics WHERE timestamp < ?",
                    (cutoff_date,)
                )
                metrics_deleted = cursor.rowcount
                
                # Clean old events
                cursor = conn.execute(
                    "DELETE FROM system_events WHERE timestamp < ?",
                    (cutoff_date,)
                )
                events_deleted = cursor.rowcount
                
                if metrics_deleted > 0 or events_deleted > 0:
                    self.logger.info(
                        f"Database cleanup: {metrics_deleted} metrics, {events_deleted} events deleted"
                    )
                
        except Exception as e:
            self.logger.error(f"Database cleanup failed: {e}")
            
    def _log(self, level: str, message: str):
        """Add entry to system logs (legacy method for compatibility)"""
        self._log_system_event(level, message)
        
    def _log_system_event(self, level: str, message: str, metadata: Optional[Dict] = None):
        """Enhanced system event logging with metadata"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Add to in-memory logs
        with self._data_lock:
            self.system_logs.append({
                'timestamp': timestamp,
                'level': level,
                'message': message,
                'metadata': metadata or {}
            })
        
        # Log to Python logger
        getattr(self.logger, level.lower(), self.logger.info)(message)
        
        # Persist to database if enabled
        if self.db_connection and self.config.enable_persistence:
            try:
                with self.db_connection.get_connection() as conn:
                    conn.execute("""
                        INSERT INTO system_events (event_type, severity, message, metadata)
                        VALUES (?, ?, ?, ?)
                    """, (
                        "dashboard",
                        level,
                        message,
                        json.dumps(metadata) if metadata else None
                    ))
            except Exception as e:
                self.logger.error(f"Failed to persist system event: {e}")
    
    def _persist_metrics(self, metrics_data: Dict[str, Any], component: str = "dashboard"):
        """Persist metrics to database"""
        if not self.db_connection or not self.config.enable_persistence:
            return
        
        try:
            with self.db_connection.get_connection() as conn:
                for metric_name, metric_value in metrics_data.items():
                    if isinstance(metric_value, (int, float)):
                        conn.execute("""
                            INSERT INTO metrics (metric_name, metric_value, component)
                            VALUES (?, ?, ?)
                        """, (metric_name, float(metric_value), component))
        except Exception as e:
            self.logger.error(f"Failed to persist metrics: {e}")
    
    def _get_cached_data(self, key: str, fetch_func: Callable, ttl: int = None) -> Any:
        """Get data from cache or fetch if not available/expired"""
        cached_data = self.data_cache.get(key)
        if cached_data is not None:
            return cached_data
        
        try:
            data = fetch_func()
            self.data_cache.set(key, data, ttl)
            return data
        except Exception as e:
            self.logger.error(f"Failed to fetch data for key {key}: {e}")
            return None
    
    @error_handler
    def _update_performance_metrics(self):
        """Update performance metrics for monitoring"""
        try:
            current_time = time.time()
            
            # Calculate average response time
            if self.response_times:
                avg_response_time = sum(self.response_times) / len(self.response_times)
            else:
                avg_response_time = 0
            
            # Update metrics
            self.performance_metrics.update({
                'total_updates': self.update_counts['success'] + self.update_counts['errors'],
                'success_rate': (self.update_counts['success'] / 
                               max(1, self.update_counts['success'] + self.update_counts['errors'])) * 100,
                'avg_response_time_ms': avg_response_time * 1000,
                'cache_hit_ratio': self._calculate_cache_hit_ratio(),
                'memory_usage_mb': self._get_memory_usage(),
                'uptime_hours': (current_time - self.last_update.timestamp()) / 3600 if self.last_update else 0
            })
            
            # Persist metrics
            self._persist_metrics(self.performance_metrics, "dashboard_performance")
            
        except Exception as e:
            self.logger.error(f"Failed to update performance metrics: {e}")
    
    def _calculate_cache_hit_ratio(self) -> float:
        """Calculate cache hit ratio (simplified)"""
        # This is a simplified calculation - in production you'd track actual hits/misses
        return 75.0  # Placeholder
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
        
    def _setup_layout(self):
        """Setup the dashboard layout"""
        self.app.layout = dbc.Container([
            # Custom CSS
            html.Style(CUSTOM_CSS),
            
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1([
                        html.Span("🚀 ", style={'fontSize': '1.2em'}),
                        "Trading Dashboard",
                        html.Span(f" - {self.selected_symbol}", 
                                className="text-info", 
                                style={'fontSize': '0.8em'})
                    ], className="mb-0"),
                    html.P(id="last-update", className="text-muted mb-0")
                ], width=8),
                dbc.Col([
                    html.Div(id="alert-container", className="alert-toast")
                ], width=4)
            ], className="mb-4"),
            
            # Health Status Row
            html.Div(id="health-panel", className="mb-4"),
            
            # Control Panel
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Symbol", className="text-muted"),
                            dbc.Input(
                                id="symbol-input",
                                value=self.selected_symbol,
                                className="form-control"
                            )
                        ], xs=6, md=2),
                        dbc.Col([
                            dbc.Label("Timeframe", className="text-muted"),
                            dbc.Select(
                                id="timeframe-select",
                                options=[
                                    {"label": tf, "value": tf}
                                    for tf in ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]
                                ],
                                value=self.selected_timeframe,
                                className="form-select"
                            )
                        ], xs=6, md=2),
                        dbc.Col([
                            dbc.Label("Bars", className="text-muted"),
                            dcc.Slider(
                                id="bars-slider",
                                min=100, max=2000, step=100,
                                value=self.config["DASH_BARS"],
                                marks={i: str(i) for i in range(100, 2001, 500)},
                                tooltip={"placement": "bottom", "always_visible": False}
                            )
                        ], xs=12, md=3),
                        dbc.Col([
                            dbc.Label("Actions", className="text-muted"),
                            dbc.ButtonGroup([
                                dbc.Button(
                                    "RT",
                                    id="realtime-toggle",
                                    color="success",
                                    size="sm",
                                    outline=True,
                                    title="Toggle Real-time"
                                ),
                                dbc.Button(
                                    "Force Sim",
                                    id="force-sim-btn",
                                    color="warning",
                                    size="sm",
                                    outline=True,
                                    title="Force Simulator"
                                ),
                                dbc.Button(
                                    "Export",
                                    id="export-btn",
                                    color="info",
                                    size="sm",
                                    outline=True
                                ),
                                dbc.Button(
                                    "Clear",
                                    id="clear-logs-btn",
                                    color="danger",
                                    size="sm",
                                    outline=True,
                                    title="Clear Logs"
                                )
                            ])
                        ], xs=12, md=5, className="text-end")
                    ])
                ])
            ], className="control-panel mb-4"),
            
            # Main Content Tabs
            dbc.Tabs([
                dbc.Tab(label="📈 Trading", tab_id="trading"),
                dbc.Tab(label="📊 Metrics", tab_id="metrics"),
                dbc.Tab(label="🔍 Analysis", tab_id="analysis"),
                dbc.Tab(label="⚙️ System", tab_id="system"),
            ], id="main-tabs", active_tab="trading"),
            
            html.Div(id="tab-content", className="mt-4"),
            
            # Hidden components
            dcc.Interval(id="interval-fast", interval=1000, n_intervals=0),
            dcc.Interval(id="interval-slow", interval=5000, n_intervals=0),
            dcc.Store(id="data-store", storage_type="memory"),
            dcc.Store(id="health-store", storage_type="memory"),
            dcc.Store(id="predictions-store", storage_type="memory"),
            dcc.Download(id="download-data"),
            
        ], fluid=True, className="dashboard-container py-4")
        
    def _create_health_panel(self) -> html.Div:
        """Create the health monitoring panel"""
        return dbc.Row([
            create_health_card("Mode", "health-mode", "mode-badge"),
            create_health_card("Circuit", "health-circuit", "circuit-badge"),
            create_health_card("Connection", "health-connection", "connection-badge"),
            create_health_card("Forced Sim", "health-forced"),
            create_health_card("Last Switch", "health-switch"),
            create_health_card("Latency", "health-latency"),
        ])
        
    def _create_trading_tab(self) -> html.Div:
        """Create the main trading view tab"""
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dcc.Graph(id="main-chart", className="mb-0")
                    ], className="chart-container")
                ], width=12, lg=9),
                dbc.Col([
                    html.Div([
                        html.H5("Predictions", className="mb-3"),
                        html.Div(id="predictions-panel")
                    ], className="chart-container h-100")
                ], width=12, lg=3)
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dcc.Graph(id="indicator-chart", className="mb-0")
                    ], className="chart-container")
                ], width=12)
            ])
        ])
        
    def _create_metrics_tab(self) -> html.Div:
        """Create the metrics view tab"""
        return html.Div([
            dbc.Row([
                create_metric_card("Total Return", "metric-return", "text-success"),
                create_metric_card("Sharpe Ratio", "metric-sharpe", "text-info"),
                create_metric_card("Sortino Ratio", "metric-sortino", "text-info"),
                create_metric_card("Max Drawdown", "metric-drawdown", "text-danger"),
                create_metric_card("Win Rate", "metric-winrate", "text-success"),
                create_metric_card("Profit Factor", "metric-profit", "text-warning"),
            ], className="mb-4"),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dcc.Graph(id="equity-chart")
                    ], className="chart-container")
                ], width=12, lg=6),
                dbc.Col([
                    html.Div([
                        dcc.Graph(id="drawdown-chart")
                    ], className="chart-container")
                ], width=12, lg=6)
            ])
        ])
        
    def _create_analysis_tab(self) -> html.Div:
        """Create the analysis view tab"""
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H5("Market Structure", className="mb-3"),
                        dcc.Graph(id="structure-chart")
                    ], className="chart-container")
                ], width=12, lg=6),
                dbc.Col([
                    html.Div([
                        html.H5("Volatility Analysis", className="mb-3"),
                        dcc.Graph(id="volatility-chart")
                    ], className="chart-container")
                ], width=12, lg=6)
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H5("Correlation Matrix", className="mb-3"),
                        dcc.Graph(id="correlation-chart")
                    ], className="chart-container")
                ], width=12)
            ])
        ])
        
    def _create_system_tab(self) -> html.Div:
        """Create the system monitoring tab"""
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H5("System Performance", className="mb-3"),
                        dcc.Graph(id="performance-chart")
                    ], className="chart-container")
                ], width=12, lg=6),
                dbc.Col([
                    html.Div([
                        html.H5("Resource Usage", className="mb-3"),
                        dcc.Graph(id="resource-chart")
                    ], className="chart-container")
                ], width=12, lg=6)
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H5("System Logs", className="mb-3 d-flex justify-content-between"),
                        html.Div(id="logs-container", className="log-container")
                    ], className="health-card")
                ], width=12)
            ])
        ])
        
    def _setup_callbacks(self):
        """Setup all dashboard callbacks"""
        
        # Tab content renderer
        @self.app.callback(
            Output("tab-content", "children"),
            Input("main-tabs", "active_tab")
        )
        def render_tab_content(active_tab):
            if active_tab == "trading":
                return self._create_trading_tab()
            elif active_tab == "metrics":
                return self._create_metrics_tab()
            elif active_tab == "analysis":
                return self._create_analysis_tab()
            elif active_tab == "system":
                return self._create_system_tab()
            return html.Div()
        
        # Health panel update
        @self.app.callback(
            Output("health-panel", "children"),
            Output("health-mode", "children"),
            Output("mode-badge", "children"),
            Output("health-circuit", "children"),
            Output("circuit-badge", "children"),
            Output("health-connection", "children"),
            Output("connection-badge", "children"),
            Output("health-forced", "children"),
            Output("health-switch", "children"),
            Output("health-latency", "children"),
            Output("last-update", "children"),
            Input("interval-fast", "n_intervals"),
            State("force-sim-btn", "n_clicks")
        )
        def update_health_panel(n_intervals, force_clicks):
            # Get health data
            if self.fallback_manager:
                try:
                    health = self.fallback_manager.health()
                    self.health_data = health
                except Exception as e:
                    self._log("ERROR", f"Failed to get health data: {e}")
                    health = {}
            else:
                # Demo data
                health = {
                    "active_mode": "SIMULATOR",
                    "circuit_state": "CLOSED",
                    "forced_sim": bool(force_clicks and force_clicks % 2),
                    "connector_health": {
                        "connected": True,
                        "avg_latency_ms": np.random.uniform(5, 50),
                        "uptime_pct": 99.5
                    }
                }
            
            # Format health data
            h = format_health_status(health)
            
            # Create badges
            mode_badge = html.Span(
                h["mode"], 
                className=f"status-badge status-{'real' if h['mode'] == 'REAL' else 'sim'}"
            )
            
            circuit_color = {
                "CLOSED": "success",
                "OPEN": "danger",
                "HALF_OPEN": "warning"
            }.get(h["circuit"], "secondary")
            circuit_badge = html.Span(
                h["circuit"], 
                className=f"status-badge status-{circuit_color}"
            )
            
            conn_badge = html.Span(
                "CONNECTED" if h["connected"] == "YES" else "DISCONNECTED",
                className=f"status-badge status-{'active' if h['connected'] == 'YES' else 'failed'}"
            )
            
            # Update time
            update_time = f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            return (
                self._create_health_panel(),
                h["mode"], mode_badge,
                h["circuit"], circuit_badge,
                h["uptime"], conn_badge,
                h["forced"],
                h["last_switch"],
                h["latency"],
                update_time
            )
        
        # Enhanced data update callback with performance tracking
        @self.app.callback(
            Output("data-store", "data"),
            Output("alert-container", "children"),
            Input("interval-slow", "n_intervals"),
            State("symbol-input", "value"),
            State("timeframe-select", "value"),
            State("bars-slider", "value")
        )
        def update_data(n_intervals, symbol, timeframe, bars):
            start_time = time.time()
            
            try:
                # Validate inputs
                if not symbol or not timeframe or not bars:
                    self.update_counts['errors'] += 1
                    return dash.no_update, None
                
                symbol = symbol.strip().upper()
                bars = min(int(bars), self.config.max_data_points)
                
                # Check cache first
                cache_key = f"market_data_{symbol}_{timeframe}_{bars}"
                
                def fetch_market_data():
                    if self.fallback_manager:
                        df = self.fallback_manager.get_rates_count(symbol, timeframe, bars)
                        if not df.empty:
                            df = calculate_indicators(df)
                            return df
                    # Generate demo data as fallback
                    return self._generate_demo_data(bars)
                
                # Get data (cached or fresh)
                df = self._get_cached_data(cache_key, fetch_market_data, ttl=self.config.refresh_sec * 2)
                
                if df is not None and not df.empty:
                    with self._data_lock:
                        self.current_data = df
                    
                    # Update performance metrics
                    response_time = time.time() - start_time
                    self.response_times.append(response_time)
                    self.update_counts['success'] += 1
                    self.last_update = datetime.now()
                    
                    # Log successful update
                    self._log_system_event(
                        "INFO", 
                        f"Updated {len(df)} bars for {symbol} {timeframe}",
                        {
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'bars_count': len(df),
                            'response_time_ms': response_time * 1000,
                            'cached': cache_key in [entry.data for entry in self.data_cache.cache.values()]
                        }
                    )
                    
                    # Persist trading metrics if available
                    if self.metrics_calculator and hasattr(df, 'close'):
                        try:
                            self._update_trading_metrics(df)
                        except Exception as e:
                            self.logger.warning(f"Failed to update trading metrics: {e}")
                    
                    return df.to_dict('records'), None
                else:
                    self.update_counts['errors'] += 1
                    self._log_system_event("WARNING", "No market data available")
                    return dash.no_update, None
                    
            except Exception as e:
                self.update_counts['errors'] += 1
                response_time = time.time() - start_time
                self.response_times.append(response_time)
                
                error_msg = f"Data update failed: {str(e)}"
                self._log_system_event("ERROR", error_msg, {
                    'symbol': symbol if 'symbol' in locals() else 'unknown',
                    'error_type': type(e).__name__,
                    'response_time_ms': response_time * 1000
                })
                
                alert = dbc.Alert(
                    error_msg,
                    color="danger",
                    dismissable=True,
                    duration=5000
                )
                return dash.no_update, alert
    
    def _update_trading_metrics(self, df: pd.DataFrame):
        """Update trading performance metrics"""
        try:
            if len(df) < 30:  # Need sufficient data for meaningful metrics
                return
            
            # Calculate basic metrics
            returns = df['close'].pct_change().dropna()
            total_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
            volatility = returns.std() * np.sqrt(252 * 288)  # Annualized volatility
            
            # Sharpe ratio (simplified)
            sharpe_ratio = (returns.mean() * np.sqrt(252 * 288)) / returns.std() if returns.std() > 0 else 0
            
            # Update metrics
            trading_metrics = {
                'total_return_pct': total_return,
                'volatility_annual': volatility,
                'sharpe_ratio': sharpe_ratio,
                'current_price': df['close'].iloc[-1],
                'price_change_24h': (df['close'].iloc[-1] - df['close'].iloc[-min(288, len(df))]) / df['close'].iloc[-min(288, len(df))] * 100
            }
            
            # Persist metrics
            self._persist_metrics(trading_metrics, "trading_performance")
            
        except Exception as e:
            self.logger.error(f"Failed to update trading metrics: {e}")
        
        # Enhanced main chart update with caching
        @self.app.callback(
            Output("main-chart", "figure"),
            Input("data-store", "data"),
            State("symbol-input", "value"),
            State("timeframe-select", "value")
        )
        @error_handler
        def update_main_chart(data, symbol, timeframe):
            if not data:
                return self._create_empty_chart("No data available")
            
            try:
                df = pd.DataFrame(data)
                if df.empty:
                    return self._create_empty_chart("Empty dataset")
                
                # Ensure time column is datetime
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'], errors='coerce')
                
                # Check cache for chart
                cache_key = f"main_chart_{symbol}_{timeframe}_{len(df)}_{hash(str(data[:5]))}"
                
                def create_chart():
                    return self._create_candlestick_chart(df, symbol, timeframe)
                
                # Use caching for better performance
                figure = self._get_cached_data(cache_key, create_chart, ttl=self.config.refresh_sec)
                return figure or self._create_empty_chart("Chart generation failed")
                
            except Exception as e:
                self.logger.error(f"Main chart update failed: {e}")
                return self._create_empty_chart(f"Chart error: {str(e)}")
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create an empty chart with a message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5, 
            showarrow=False,
            font=dict(size=16, color=THEME['dark']['text_secondary'])
        )
        fig.update_layout(
            template='plotly_dark',
            height=600,
            paper_bgcolor=THEME['dark']['bg_primary'],
            plot_bgcolor=THEME['dark']['chart_bg'],
            font=dict(color=THEME['dark']['text_primary']),
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig
    
    def _create_candlestick_chart(self, df: pd.DataFrame, symbol: str, timeframe: str) -> go.Figure:
        """Create enhanced candlestick chart with indicators"""
        try:
            # Create subplots for price and volume
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.7, 0.3],
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(f'{symbol} {timeframe} - Advanced Chart', 'Volume & Momentum')
            )
            
            x_axis = df['time'] if 'time' in df.columns else df.index
            
            # Candlestick with enhanced styling
            fig.add_trace(
                go.Candlestick(
                    x=x_axis,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='OHLC',
                    increasing_line_color=THEME['dark']['success'],
                    decreasing_line_color=THEME['dark']['danger'],
                    increasing_fillcolor=f"{THEME['dark']['success']}40",
                    decreasing_fillcolor=f"{THEME['dark']['danger']}40"
                ),
                row=1, col=1
            )
            
            # Add technical indicators with improved visibility
            self._add_technical_indicators(fig, df, x_axis)
            
            # Volume with color coding
            if 'volume' in df.columns or 'tick_volume' in df.columns:
                volume_col = df.get('volume', df.get('tick_volume', []))
                volume_colors = [
                    THEME['dark']['success'] if close >= open_ else THEME['dark']['danger']
                    for close, open_ in zip(df['close'], df['open'])
                ]
                
                fig.add_trace(
                    go.Bar(
                        x=x_axis,
                        y=volume_col,
                        name='Volume',
                        marker_color=volume_colors,
                        opacity=0.6,
                        yaxis='y2'
                    ),
                    row=2, col=1
                )
            
            # Enhanced layout with better styling
            fig.update_layout(
                template='plotly_dark',
                height=650,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    bgcolor="rgba(0,0,0,0.5)"
                ),
                xaxis_rangeslider_visible=False,
                paper_bgcolor=THEME['dark']['bg_primary'],
                plot_bgcolor=THEME['dark']['chart_bg'],
                font=dict(color=THEME['dark']['text_primary'], size=11),
                title=dict(
                    text=f"📈 {symbol} {timeframe} Trading Chart",
                    x=0.5,
                    font=dict(size=16)
                ),
                hovermode='x unified'
            )
            
            # Enhanced axes styling
            fig.update_xaxes(
                gridcolor=THEME['dark']['grid_color'],
                gridwidth=0.5,
                showgrid=True,
                tickformat='%H:%M\\n%d/%m' if timeframe in ['M1', 'M5', 'M15', 'M30', 'H1'] else '%d/%m/%Y'
            )
            fig.update_yaxes(
                gridcolor=THEME['dark']['grid_color'],
                gridwidth=0.5,
                showgrid=True,
                tickformat='.4f' if symbol == 'USDCOP' else '.5f'
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Candlestick chart creation failed: {e}")
            return self._create_empty_chart(f"Chart error: {str(e)}")
    
    def _add_technical_indicators(self, fig: go.Figure, df: pd.DataFrame, x_axis) -> None:
        """Add technical indicators to the chart"""
        try:
            # Moving averages with better styling
            if 'sma_20' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=x_axis,
                        y=df['sma_20'],
                        name='SMA 20',
                        line=dict(color=THEME['dark']['info'], width=2, dash='solid'),
                        opacity=0.8
                    ),
                    row=1, col=1
                )
                
            if 'ema_20' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=x_axis,
                        y=df['ema_20'],
                        name='EMA 20',
                        line=dict(color=THEME['dark']['warning'], width=2, dash='dot'),
                        opacity=0.8
                    ),
                    row=1, col=1
                )
            
            # Bollinger Bands with fill
            if all(col in df.columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
                # Upper and lower bands
                fig.add_trace(
                    go.Scatter(
                        x=x_axis,
                        y=df['bb_upper'],
                        name='BB Upper',
                        line=dict(color=THEME['dark']['purple'], width=1, dash='dash'),
                        opacity=0.6
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=x_axis,
                        y=df['bb_lower'],
                        name='BB Lower',
                        line=dict(color=THEME['dark']['purple'], width=1, dash='dash'),
                        fill='tonexty',
                        fillcolor=f"{THEME['dark']['purple']}20",
                        opacity=0.6
                    ),
                    row=1, col=1
                )
                
                # Middle band
                fig.add_trace(
                    go.Scatter(
                        x=x_axis,
                        y=df['bb_middle'],
                        name='BB Middle',
                        line=dict(color=THEME['dark']['purple'], width=1),
                        opacity=0.4
                    ),
                    row=1, col=1
                )
            
            # Support and resistance levels (if available)
            self._add_support_resistance(fig, df, x_axis)
            
        except Exception as e:
            self.logger.warning(f"Failed to add technical indicators: {e}")
    
    def _add_support_resistance(self, fig: go.Figure, df: pd.DataFrame, x_axis) -> None:
        """Add support and resistance levels"""
        try:
            # Simple support/resistance based on recent highs and lows
            if len(df) >= 50:
                recent_data = df.tail(50)
                resistance = recent_data['high'].max()
                support = recent_data['low'].min()
                
                # Add horizontal lines for S/R
                fig.add_hline(
                    y=resistance,
                    line_dash="solid",
                    line_color="red",
                    opacity=0.5,
                    annotation_text=f"Resistance: {resistance:.4f}",
                    annotation_position="bottom right"
                )
                fig.add_hline(
                    y=support,
                    line_dash="solid", 
                    line_color="green",
                    opacity=0.5,
                    annotation_text=f"Support: {support:.4f}",
                    annotation_position="top right"
                )
                
        except Exception as e:
            self.logger.debug(f"Support/resistance calculation failed: {e}")
                
            # Create candlestick chart
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.7, 0.3],
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(f'{symbol} {timeframe}', 'Volume')
            )
            
            # Candlestick
            fig.add_trace(
                go.Candlestick(
                    x=df['time'] if 'time' in df else df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='OHLC',
                    increasing_line_color=THEME['dark']['success'],
                    decreasing_line_color=THEME['dark']['danger']
                ),
                row=1, col=1
            )
            
            # Add indicators
            if 'sma_20' in df:
                fig.add_trace(
                    go.Scatter(
                        x=df['time'] if 'time' in df else df.index,
                        y=df['sma_20'],
                        name='SMA 20',
                        line=dict(color=THEME['dark']['info'], width=1)
                    ),
                    row=1, col=1
                )
                
            if 'ema_20' in df:
                fig.add_trace(
                    go.Scatter(
                        x=df['time'] if 'time' in df else df.index,
                        y=df['ema_20'],
                        name='EMA 20',
                        line=dict(color=THEME['dark']['warning'], width=1)
                    ),
                    row=1, col=1
                )
                
            if 'bb_upper' in df and 'bb_lower' in df:
                fig.add_trace(
                    go.Scatter(
                        x=df['time'] if 'time' in df else df.index,
                        y=df['bb_upper'],
                        name='BB Upper',
                        line=dict(color=THEME['dark']['purple'], width=1, dash='dash')
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=df['time'] if 'time' in df else df.index,
                        y=df['bb_lower'],
                        name='BB Lower',
                        line=dict(color=THEME['dark']['purple'], width=1, dash='dash')
                    ),
                    row=1, col=1
                )
            
            # Volume
            volume_colors = [THEME['dark']['success'] if close >= open_ else THEME['dark']['danger']
                           for close, open_ in zip(df['close'], df['open'])]
            
            fig.add_trace(
                go.Bar(
                    x=df['time'] if 'time' in df else df.index,
                    y=df['volume'] if 'volume' in df else df.get('tick_volume', []),
                    name='Volume',
                    marker_color=volume_colors,
                    opacity=0.5
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                template='plotly_dark',
                height=600,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                xaxis_rangeslider_visible=False,
                paper_bgcolor=THEME['dark']['bg_primary'],
                plot_bgcolor=THEME['dark']['chart_bg'],
                font=dict(color=THEME['dark']['text_primary'])
            )
            
            # Update axes
            fig.update_xaxes(gridcolor=THEME['dark']['grid_color'])
            fig.update_yaxes(gridcolor=THEME['dark']['grid_color'])
            
            return fig
        
        # Indicator charts
        @self.app.callback(
            Output("indicator-chart", "figure"),
            Input("data-store", "data")
        )
        def update_indicator_chart(data):
            if not data:
                return go.Figure()
                
            df = pd.DataFrame(data)
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                
            # Create subplots for RSI and MACD
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.5, 0.5],
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=('RSI', 'MACD')
            )
            
            x_axis = df['time'] if 'time' in df else df.index
            
            # RSI
            if 'rsi' in df:
                fig.add_trace(
                    go.Scatter(
                        x=x_axis,
                        y=df['rsi'],
                        name='RSI',
                        line=dict(color=THEME['dark']['purple'], width=2)
                    ),
                    row=1, col=1
                )
                
                # Overbought/Oversold lines
                fig.add_hline(y=70, line_dash="dash", line_color=THEME['dark']['danger'], row=1, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color=THEME['dark']['success'], row=1, col=1)
                fig.add_hline(y=50, line_dash="dot", line_color=THEME['dark']['text_secondary'], row=1, col=1)
            
            # MACD
            if 'macd' in df and 'macd_signal' in df:
                fig.add_trace(
                    go.Scatter(
                        x=x_axis,
                        y=df['macd'],
                        name='MACD',
                        line=dict(color=THEME['dark']['info'], width=2)
                    ),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=x_axis,
                        y=df['macd_signal'],
                        name='Signal',
                        line=dict(color=THEME['dark']['warning'], width=2)
                    ),
                    row=2, col=1
                )
                
                # MACD Histogram
                if 'macd_histogram' in df:
                    colors = [THEME['dark']['success'] if x > 0 else THEME['dark']['danger'] 
                             for x in df['macd_histogram']]
                    fig.add_trace(
                        go.Bar(
                            x=x_axis,
                            y=df['macd_histogram'],
                            name='MACD Hist',
                            marker_color=colors,
                            opacity=0.3
                        ),
                        row=2, col=1
                    )
            
            # Update layout
            fig.update_layout(
                template='plotly_dark',
                height=400,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                paper_bgcolor=THEME['dark']['bg_primary'],
                plot_bgcolor=THEME['dark']['chart_bg'],
                font=dict(color=THEME['dark']['text_primary'])
            )
            
            # Update axes
            fig.update_xaxes(gridcolor=THEME['dark']['grid_color'])
            fig.update_yaxes(gridcolor=THEME['dark']['grid_color'])
            
            # RSI range
            fig.update_yaxes(range=[0, 100], row=1, col=1)
            
            return fig
        
        # Metrics update
        @self.app.callback(
            Output("metric-return", "children"),
            Output("metric-sharpe", "children"),
            Output("metric-sortino", "children"),
            Output("metric-drawdown", "children"),
            Output("metric-winrate", "children"),
            Output("metric-profit", "children"),
            Output("equity-chart", "figure"),
            Output("drawdown-chart", "figure"),
            Input("data-store", "data"),
            Input("main-tabs", "active_tab")
        )
        def update_metrics(data, active_tab):
            if not data or active_tab != "metrics":
                raise PreventUpdate
                
            df = pd.DataFrame(data)
            
            # Calculate returns
            returns = df['close'].pct_change().dropna()
            
            # Calculate metrics
            total_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
            
            # Sharpe ratio (annualized)
            if len(returns) > 0 and returns.std() > 0:
                sharpe = (returns.mean() * np.sqrt(252 * 288)) / returns.std()  # 288 = 5min bars per day
            else:
                sharpe = 0
                
            # Sortino ratio
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0 and downside_returns.std() > 0:
                sortino = (returns.mean() * np.sqrt(252 * 288)) / downside_returns.std()
            else:
                sortino = 0
                
            # Max drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            max_dd = drawdown.min() * 100
            
            # Placeholder values for win rate and profit factor
            win_rate = 52.5  # Demo value
            profit_factor = 1.25  # Demo value
            
            # Create equity curve chart
            equity_fig = go.Figure()
            equity_fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=cumulative,
                    mode='lines',
                    name='Equity',
                    line=dict(color=THEME['dark']['success'], width=2)
                )
            )
            
            equity_fig.update_layout(
                template='plotly_dark',
                height=350,
                title='Equity Curve',
                showlegend=False,
                paper_bgcolor=THEME['dark']['bg_primary'],
                plot_bgcolor=THEME['dark']['chart_bg'],
                font=dict(color=THEME['dark']['text_primary'])
            )
            
            # Create drawdown chart
            dd_fig = go.Figure()
            dd_fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=drawdown * 100,
                    mode='lines',
                    fill='tozeroy',
                    name='Drawdown',
                    line=dict(color=THEME['dark']['danger'], width=1),
                    fillcolor=f"rgba(248, 81, 73, 0.2)"
                )
            )
            
            dd_fig.update_layout(
                template='plotly_dark',
                height=350,
                title='Drawdown %',
                showlegend=False,
                paper_bgcolor=THEME['dark']['bg_primary'],
                plot_bgcolor=THEME['dark']['chart_bg'],
                font=dict(color=THEME['dark']['text_primary'])
            )
            
            return (
                f"{total_return:.2f}%",
                f"{sharpe:.2f}",
                f"{sortino:.2f}",
                f"{max_dd:.2f}%",
                f"{win_rate:.1f}%",
                f"{profit_factor:.2f}",
                equity_fig,
                dd_fig
            )
        
        # System logs update
        @self.app.callback(
            Output("logs-container", "children"),
            Input("interval-slow", "n_intervals"),
            Input("clear-logs-btn", "n_clicks")
        )
        def update_logs(n_intervals, clear_clicks):
            ctx = callback_context
            
            if ctx.triggered:
                if ctx.triggered[0]['prop_id'].split('.')[0] == 'clear-logs-btn':
                    self.system_logs.clear()
                    self._log("INFO", "Logs cleared")
            
            # Get last 50 logs
            logs_html = []
            for log in list(self.system_logs)[-50:]:
                color = {
                    'ERROR': THEME['dark']['danger'],
                    'WARNING': THEME['dark']['warning'],
                    'INFO': THEME['dark']['info'],
                    'DEBUG': THEME['dark']['text_secondary']
                }.get(log['level'], THEME['dark']['text_primary'])
                
                logs_html.append(
                    html.Div([
                        html.Span(f"[{log['timestamp']}] ", 
                                style={'color': THEME['dark']['text_secondary']}),
                        html.Span(f"{log['level']}: ", 
                                style={'color': color, 'fontWeight': 'bold'}),
                        html.Span(log['message'])
                    ], className="log-entry")
                )
            
            if not logs_html:
                logs_html = [html.Div("No logs available", 
                                    className="text-muted text-center")]
            
            return logs_html
        
        # Export functionality
        @self.app.callback(
            Output("download-data", "data"),
            Input("export-btn", "n_clicks"),
            prevent_initial_call=True
        )
        def export_data(n_clicks):
            if self.current_data.empty:
                return None
                
            # Prepare export data
            export_df = self.current_data.copy()
            export_df.reset_index(inplace=True)
            
            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trading_data_{self.selected_symbol}_{timestamp}.csv"
            
            self._log("INFO", f"Exported data to {filename}")
            
            return dcc.send_data_frame(export_df.to_csv, filename, index=False)
        
        # Real-time toggle
        @self.app.callback(
            Output("realtime-toggle", "outline"),
            Output("interval-slow", "disabled"),
            Input("realtime-toggle", "n_clicks"),
            State("realtime-toggle", "outline")
        )
        def toggle_realtime(n_clicks, current_outline):
            if n_clicks:
                new_state = not current_outline
                self.real_time_mode = not new_state
                self._log("INFO", f"Real-time mode: {'ON' if self.real_time_mode else 'OFF'}")
                return new_state, new_state
            return True, False
        
        # Force simulator
        @self.app.callback(
            Output("force-sim-btn", "outline"),
            Input("force-sim-btn", "n_clicks"),
            State("force-sim-btn", "outline")
        )
        def toggle_force_sim(n_clicks, current_outline):
            if n_clicks and self.fallback_manager:
                new_state = not current_outline
                try:
                    self.fallback_manager.force_simulator(not new_state)
                    self._log("WARNING", f"Force simulator: {'ON' if not new_state else 'OFF'}")
                except Exception as e:
                    self._log("ERROR", f"Failed to toggle force simulator: {e}")
                return new_state
            return current_outline
            
    def _generate_demo_data(self, bars: int) -> pd.DataFrame:
        """Generate demo data for testing"""
        # Generate realistic OHLCV data
        dates = pd.date_range(end=datetime.now(), periods=bars, freq='5min')
        
        # Random walk for price
        returns = np.random.normal(0.0001, 0.002, bars)
        price = 4000 * np.exp(np.cumsum(returns))
        
        # Create OHLCV
        df = pd.DataFrame({
            'time': dates,
            'open': price * (1 + np.random.uniform(-0.001, 0.001, bars)),
            'high': price * (1 + np.abs(np.random.normal(0.002, 0.001, bars))),
            'low': price * (1 - np.abs(np.random.normal(0.002, 0.001, bars))),
            'close': price,
            'volume': np.random.randint(100, 10000, bars)
        })
        
        # Ensure OHLC consistency
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        # Add indicators
        df = calculate_indicators(df)
        
        return df
        
    def run(self, host: str = None, port: int = None, debug: bool = None):
        """Run the dashboard server with enhanced error handling and monitoring"""
        host = host or self.config.host
        port = port or self.config.port
        debug = debug if debug is not None else self.config.debug
        
        self.running = True
        self.logger.info(f"Starting enhanced USDCOP Trading Dashboard")
        self.logger.info(f"Server: http://{host}:{port}")
        self.logger.info(f"Debug mode: {debug}")
        self.logger.info(f"Components available: {COMPONENTS_AVAILABLE}")
        
        # Log system capabilities
        capabilities = {
            'health_monitor': self.health_monitor is not None,
            'fallback_manager': self.fallback_manager is not None,
            'metrics_calculator': self.metrics_calculator is not None,
            'caching_enabled': True,
            'persistence_enabled': self.config.enable_persistence,
            'real_time_enabled': self.config.enable_real_time
        }
        
        self.logger.info(f"System capabilities: {capabilities}")
        self._log_system_event("INFO", "Dashboard server starting", capabilities)
        
        try:
            # Pre-startup checks
            self._perform_startup_checks()
            
            # Start background performance monitoring
            if not debug:  # Only in production
                self._start_performance_monitoring()
            
            # Start the server
            self.app.run_server(
                host=host,
                port=port,
                debug=debug,
                use_reloader=False,  # Avoid issues with threading
                dev_tools_hot_reload=debug,
                dev_tools_silence_routes_logging=not debug
            )
            
        except KeyboardInterrupt:
            self.logger.info("Dashboard stopped by user (Ctrl+C)")
        except Exception as e:
            self.logger.error(f"Dashboard server error: {e}")
            self.logger.debug(f"Server error traceback: {traceback.format_exc()}")
            raise
        finally:
            self.stop()
    
    def _perform_startup_checks(self):
        """Perform pre-startup health checks"""
        try:
            # Check database connection if enabled
            if self.db_connection:
                with self.db_connection.get_connection() as conn:
                    conn.execute("SELECT 1").fetchone()
                self.logger.info("Database connection: OK")
            
            # Check health monitor
            if self.health_monitor:
                snapshot = self.health_monitor.get_snapshot()
                self.logger.info(f"Health monitor: {snapshot.status}")
            
            # Check data availability
            demo_data = self._generate_demo_data(100)
            if not demo_data.empty:
                self.logger.info("Demo data generation: OK")
            
            # Update performance metrics
            self._update_performance_metrics()
            
        except Exception as e:
            self.logger.warning(f"Startup checks failed: {e}")
    
    def _start_performance_monitoring(self):
        """Start background performance monitoring"""
        def performance_monitor():
            while self.running:
                try:
                    self._update_performance_metrics()
                    time.sleep(60)  # Update every minute
                except Exception as e:
                    self.logger.error(f"Performance monitoring error: {e}")
                    time.sleep(60)
        
        perf_thread = threading.Thread(
            target=performance_monitor,
            daemon=True,
            name="PerformanceMonitor"
        )
        perf_thread.start()
        self.logger.info("Performance monitoring started")
            
    def stop(self):
        """Stop the dashboard and cleanup resources"""
        self.logger.info("Initiating dashboard shutdown...")
        self.running = False
        
        # Stop background threads
        if hasattr(self, 'cleanup_thread') and self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5)
            self.logger.info("Cleanup thread stopped")
        
        # Stop system components
        if self.fallback_manager:
            try:
                self.fallback_manager.stop()
                self.logger.info("Fallback manager stopped")
            except Exception as e:
                self.logger.error(f"Error stopping fallback manager: {e}")
        
        if self.health_monitor:
            try:
                self.health_monitor.stop()
                self.logger.info("Health monitor stopped")
            except Exception as e:
                self.logger.error(f"Error stopping health monitor: {e}")
        
        if self.shutdown_manager:
            try:
                self.shutdown_manager.shutdown()
                self.logger.info("Shutdown manager executed")
            except Exception as e:
                self.logger.error(f"Error executing shutdown manager: {e}")
        
        # Clear caches
        if hasattr(self, 'data_cache'):
            self.data_cache.clear()
            self.logger.info("Data cache cleared")
        
        # Close database connections
        if hasattr(self, 'db_connection') and self.db_connection:
            try:
                if hasattr(self.db_connection, 'local') and hasattr(self.db_connection.local, 'conn'):
                    self.db_connection.local.conn.close()
                self.logger.info("Database connections closed")
            except Exception as e:
                self.logger.error(f"Error closing database connections: {e}")
        
        # Log final metrics
        try:
            self._log_system_event("INFO", "Dashboard shutdown completed", {
                'total_updates': self.update_counts.get('success', 0) + self.update_counts.get('errors', 0),
                'success_rate': (self.update_counts.get('success', 0) / 
                               max(1, self.update_counts.get('success', 0) + self.update_counts.get('errors', 0))) * 100,
                'uptime_seconds': (datetime.now() - self.last_update).total_seconds() if self.last_update else 0
            })
        except Exception:
            pass  # Don't let logging errors prevent shutdown
        
        self.logger.info("Dashboard shutdown complete")

# ------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------
def main():
    """Enhanced main function with comprehensive configuration and error handling"""
    import argparse
    import signal
    import sys
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='USDCOP Trading Dashboard - Production Ready',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --host 0.0.0.0 --port 8050 --debug
  %(prog)s --config config/dashboard_config.yaml
  %(prog)s --log-level DEBUG --enable-persistence
        """
    )
    
    # Server options
    server_group = parser.add_argument_group('server options')
    server_group.add_argument('--host', default=None, 
                             help='Host to bind (default: from config or 0.0.0.0)')
    server_group.add_argument('--port', type=int, default=None, 
                             help='Port to bind (default: from config or 8050)')
    server_group.add_argument('--debug', action='store_true', 
                             help='Enable debug mode')
    
    # Configuration options
    config_group = parser.add_argument_group('configuration options')
    config_group.add_argument('--config', 
                             help='Path to configuration file (YAML or JSON)')
    config_group.add_argument('--log-level', default='INFO',
                             choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                             help='Logging level (default: INFO)')
    
    # Feature flags
    features_group = parser.add_argument_group('feature options')
    features_group.add_argument('--enable-persistence', action='store_true',
                               help='Enable database persistence')
    features_group.add_argument('--disable-health-monitor', action='store_true',
                               help='Disable health monitoring')
    features_group.add_argument('--enable-websocket', action='store_true',
                               help='Enable WebSocket support')
    features_group.add_argument('--demo-mode', action='store_true',
                               help='Force demo mode (no external dependencies)')
    
    # Performance options
    perf_group = parser.add_argument_group('performance options')
    perf_group.add_argument('--cache-timeout', type=int, default=300,
                           help='Cache timeout in seconds (default: 300)')
    perf_group.add_argument('--max-data-points', type=int, default=2000,
                           help='Maximum data points to load (default: 2000)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            try:
                if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                    config = DashboardConfig.from_yaml(str(config_path))
                else:
                    # Try JSON format
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                    # Convert JSON to DashboardConfig
                    config = DashboardConfig(**{k: v for k, v in config_data.items() 
                                               if k in DashboardConfig.__annotations__})
                print(f"✓ Loaded configuration from {config_path}")
            except Exception as e:
                print(f"✗ Failed to load config from {args.config}: {e}")
                sys.exit(1)
        else:
            print(f"✗ Configuration file not found: {args.config}")
            sys.exit(1)
    else:
        # Try to find default config files
        default_configs = [
            "config/dashboard_config.yaml",
            "configs/dashboard_config.yaml", 
            "dashboard_config.yaml"
        ]
        
        for default_config in default_configs:
            if os.path.exists(default_config):
                try:
                    config = DashboardConfig.from_yaml(default_config)
                    print(f"✓ Loaded default configuration from {default_config}")
                    break
                except Exception as e:
                    print(f"⚠ Failed to load default config {default_config}: {e}")
        
        if not config:
            config = DashboardConfig.from_env()
            print("✓ Using environment/default configuration")
    
    # Override config with command line arguments
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port
    if args.debug:
        config.debug = True
    if args.log_level:
        config.log_level = args.log_level
    if args.enable_persistence:
        config.enable_persistence = True
    if args.disable_health_monitor:
        config.enable_health_monitor = False
    if args.enable_websocket:
        config.enable_websocket = True
    if args.cache_timeout:
        config.cache_timeout = args.cache_timeout
    if args.max_data_points:
        config.max_data_points = args.max_data_points
    
    # Setup global logging
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='[%(asctime)s] %(levelname)-8s %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create logger for main
    main_logger = logging.getLogger('dashboard.main')
    
    # Print startup banner
    print("=" * 60)
    print("🚀 USDCOP Trading Dashboard - Production Ready")
    print("=" * 60)
    print(f"Server: http://{config.host}:{config.port}")
    print(f"Debug: {config.debug}")
    print(f"Log Level: {config.log_level}")
    print(f"Components Available: {COMPONENTS_AVAILABLE}")
    print(f"Features: Cache={config.cache_timeout}s, Persistence={config.enable_persistence}")
    print("=" * 60)
    
    # Create and configure dashboard
    dashboard = None
    
    def signal_handler(sig, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n⚠ Received signal {sig}, shutting down gracefully...")
        if dashboard:
            dashboard.stop()
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Create dashboard instance
        main_logger.info("Initializing dashboard...")
        dashboard = UnifiedTradingDashboard(config)
        
        # Demo mode override
        if args.demo_mode:
            main_logger.info("Demo mode enabled - disabling external connections")
            dashboard.fallback_manager = None
            dashboard.health_monitor = None
        
        # Start the dashboard
        main_logger.info("Starting dashboard server...")
        dashboard.run(
            host=config.host,
            port=config.port,
            debug=config.debug
        )
        
    except KeyboardInterrupt:
        main_logger.info("Shutdown requested by user")
    except Exception as e:
        main_logger.error(f"Dashboard startup failed: {e}")
        main_logger.debug(f"Startup error traceback: {traceback.format_exc()}")
        
        print(f"\n✗ Dashboard failed to start: {e}")
        if config.debug:
            print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
    finally:
        if dashboard:
            try:
                dashboard.stop()
            except Exception as e:
                main_logger.error(f"Error during shutdown: {e}")
        
        print("\n👋 Dashboard stopped. Goodbye!")
        main_logger.info("Dashboard shutdown completed")


if __name__ == "__main__":
    main()