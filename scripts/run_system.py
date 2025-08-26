#!/usr/bin/env python3
"""
USDCOP Trading System - Main Entry Point
========================================
Production-ready orchestrator that combines the best features from all implementations.

This unified system provides:
- Multiple operation modes (live, backtest, train, etc.)
- Robust component initialization with fallbacks
- Advanced health monitoring and metrics
- State persistence and recovery
- Comprehensive error handling
- Multi-threaded architecture
- Real-time monitoring capabilities
- Graceful shutdown handling
- Environment-aware configuration

Usage:
    python scripts/run_system.py live                  # Run 24/7 trading
    python scripts.run_system.py backfill --start 2024-01-01 --end 2024-12-31
    python scripts/run_system.py train --algo ppo --steps 1000000
    python scripts/run_system.py dashboard             # Launch monitoring UI

Author: USDCOP Trading System
Version: 4.0.0
"""

from __future__ import annotations

import os
import sys
import re
import io
import gc
import time
import json
import yaml
import queue
import signal
import shutil
import random
import logging
import argparse
import threading
import subprocess
import traceback
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Handle imports gracefully
try:
    import pandas as pd
    import numpy as np
    import psutil
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: pip install pandas numpy psutil pyyaml")
    sys.exit(1)

# Service Discovery and API imports
try:
    from fastapi import FastAPI
    import uvicorn
    from src.visualization.components.health_monitor import create_health_monitor_router
    from src.core.discovery.registry import ServiceRegistry, ServiceMeta
    from src.core.events.bus import event_bus
    print("✅ Service discovery and API features enabled")
except ImportError as e:
    print(f"Warning: Service discovery and API features disabled: {e}")
    print("Install with: pip install fastapi uvicorn")
    FastAPI = None
    uvicorn = None
    create_health_monitor_router = None
    ServiceRegistry = None
    ServiceMeta = None
    event_bus = None

# Suppress warnings in production
warnings.filterwarnings('ignore')

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Also add current directory for direct script execution
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT_DIR))
    sys.path.insert(0, ".")

# -----------------------------------------------------------------------------
# Optional Module Imports (with graceful fallback)
# -----------------------------------------------------------------------------

# Try to import local modules
MODULES = {
    'pipeline': None,
    'agent': None,
    'metrics': None,
    'dashboard': None,
    'mt5_connector': None,
    'risk_manager': None,
    'portfolio': None,
    'backtester': None
}

try:
    from src.markets.usdcop import pipeline
    MODULES['pipeline'] = pipeline
except ImportError:
    pass

try:
    from src.markets.usdcop import agent
    MODULES['agent'] = agent
except ImportError:
    pass

try:
    from src.markets.usdcop import metrics
    MODULES['metrics'] = metrics
except ImportError:
    pass

try:
    from src.visualization import dashboard
    MODULES['dashboard'] = dashboard
except ImportError:
    pass

try:
    from src.core.connectors.mt5_connector import MT5Connector
    MODULES['mt5_connector'] = MT5Connector
except ImportError:
    pass

try:
    from src.core.risk_manager import RiskManager
    MODULES['risk_manager'] = RiskManager
except ImportError:
    pass

try:
    from src.core.portfolio_manager import PortfolioManager
    MODULES['portfolio'] = PortfolioManager
except ImportError:
    pass

try:
    from src.backtesting.advanced_backtester import AdvancedBacktester
    MODULES['backtester'] = AdvancedBacktester
except ImportError:
    pass

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------

def setup_logging(level: str = "INFO", log_file: Optional[str] = None, 
                  console: bool = True) -> logging.Logger:
    """Setup comprehensive logging configuration"""
    logger = logging.getLogger("trading_system")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Format
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)-8s %(name)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        try:
            from logging.handlers import RotatingFileHandler
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=50 * 1024 * 1024,  # 50MB
                backupCount=5
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Failed to setup file logging: {e}")
    
    return logger

# Initialize logger
logger = setup_logging()

# -----------------------------------------------------------------------------
# Configuration Management
# -----------------------------------------------------------------------------

def expand_env_vars(value: Any) -> Any:
    """Expand environment variables in configuration values"""
    if not isinstance(value, str):
        return value
    
    # Pattern for ${VAR} or ${VAR:-default}
    pattern = re.compile(r'\$\{([^}:]+)(?::-([^}]*))?\}')
    
    def replacer(match):
        var_name = match.group(1)
        default_value = match.group(2) if match.group(2) is not None else ""
        return os.environ.get(var_name, default_value)
    
    return pattern.sub(replacer, value)

def deep_expand_config(obj: Any) -> Any:
    """Recursively expand environment variables in config"""
    if isinstance(obj, dict):
        return {k: deep_expand_config(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deep_expand_config(x) for x in obj]
    else:
        return expand_env_vars(obj)

def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two configuration dictionaries"""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result

def load_config(path: str, profile: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration with profile support"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Expand environment variables
    config = deep_expand_config(config)
    
    # Handle profiles
    active_profile = profile or os.getenv('APP_PROFILE') or config.get('active_profile', 'dev')
    
    # Merge defaults with profile
    defaults = config.get('defaults', {})
    profiles = config.get('profiles', {})
    
    if active_profile in profiles:
        final_config = merge_configs(defaults, profiles[active_profile])
    else:
        logger.warning(f"Profile '{active_profile}' not found, using defaults")
        final_config = defaults
    
    # Add metadata
    final_config['_profile'] = active_profile
    final_config['_config_path'] = path
    final_config['_loaded_at'] = datetime.now().isoformat()
    
    return final_config

# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------

@dataclass
class SystemState:
    """Complete system state tracking"""
    status: str = "initializing"  # initializing, ready, running, paused, stopping, stopped, error
    mode: str = "development"     # development, production, backtest
    profile: str = "dev"
    start_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    last_health_check: datetime = field(default_factory=datetime.now)
    
    # Trading stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    active_positions: int = 0
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    peak_balance: float = 0.0
    current_drawdown: float = 0.0
    
    # System health
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    active_threads: int = 0
    
    # Error tracking
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    component_status: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'status': self.status,
            'mode': self.mode,
            'profile': self.profile,
            'start_time': self.start_time.isoformat(),
            'last_update': self.last_update.isoformat(),
            'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
            'trading': {
                'total_trades': self.total_trades,
                'win_rate': self.winning_trades / max(self.total_trades, 1),
                'active_positions': self.active_positions,
                'total_pnl': self.total_pnl,
                'daily_pnl': self.daily_pnl,
                'current_drawdown': self.current_drawdown
            },
            'health': {
                'cpu_percent': self.cpu_percent,
                'memory_percent': self.memory_percent,
                'disk_percent': self.disk_percent,
                'active_threads': self.active_threads
            },
            'errors': len(self.errors),
            'warnings': len(self.warnings),
            'components': self.component_status
        }
    
    def add_error(self, error: str, component: str = "system"):
        """Add error with timestamp"""
        self.errors.append({
            'timestamp': datetime.now().isoformat(),
            'component': component,
            'error': error
        })
        # Keep only last 1000 errors
        if len(self.errors) > 1000:
            self.errors = self.errors[-1000:]
    
    def add_warning(self, warning: str, component: str = "system"):
        """Add warning with timestamp"""
        self.warnings.append({
            'timestamp': datetime.now().isoformat(),
            'component': component,
            'warning': warning
        })
        # Keep only last 1000 warnings
        if len(self.warnings) > 1000:
            self.warnings = self.warnings[-1000:]

@dataclass
class SystemConfig:
    """System runtime configuration"""
    # Operation mode
    mode: str = "production"
    dry_run: bool = False
    use_simulator: bool = False
    
    # Features
    enable_dashboard: bool = True
    enable_notifications: bool = True
    enable_ml_updates: bool = False
    enable_health_checks: bool = True
    enable_state_persistence: bool = True
    
    # Intervals (seconds)
    health_check_interval: int = 60
    metrics_update_interval: int = 30
    state_save_interval: int = 300
    data_update_interval: int = 5
    position_check_interval: int = 10
    model_update_interval: int = 86400  # 24 hours
    
    # System limits
    max_threads: int = 8
    max_memory_percent: float = 80.0
    max_cpu_percent: float = 90.0
    max_errors_per_hour: int = 100
    
    # Paths
    log_dir: str = "logs"
    data_dir: str = "data"
    model_dir: str = "models"
    report_dir: str = "reports"
    state_file: str = "data/system_state.json"
    metrics_file: str = "data/metrics/system_health.prom"

# -----------------------------------------------------------------------------
# Trading System Orchestrator
# -----------------------------------------------------------------------------

class TradingSystemOrchestrator:
    """Main trading system orchestrator combining all components"""
    
    def __init__(self, config_path: str, system_config: SystemConfig):
        """Initialize the orchestrator"""
        self.config_path = config_path
        self.system_config = system_config
        self.state = SystemState(mode=system_config.mode)
        
        # Configuration
        self.config = None
        
        # Core components
        self.components = {
            'mt5_connector': None,
            'data_pipeline': None,
            'data_simulator': None,
            'rl_agent': None,
            'strategy': None,
            'risk_manager': None,
            'position_manager': None,
            'order_executor': None,
            'portfolio_manager': None,
            'metrics_calculator': None,
            'dashboard': None,
            'notification_manager': None
        }
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=system_config.max_threads)
        self.running = False
        self.workers = {}
        self.locks = {
            'state': threading.Lock(),
            'positions': threading.Lock(),
            'orders': threading.Lock(),
            'data': threading.Lock()
        }
        
        # Queues for inter-thread communication
        self.queues = {
            'signals': queue.Queue(maxsize=100),
            'orders': queue.Queue(maxsize=100),
            'metrics': queue.Queue(maxsize=1000)
        }
        
        # API and Service Discovery
        self.api_thread = None
        self.api_port = int(os.getenv("APP_API_PORT", "8000"))
        self.service_registry = None
        
        # Install signal handlers
        self._install_signal_handlers()
        
        logger.info(f"Trading System Orchestrator initialized in {system_config.mode} mode")
    
    def initialize(self) -> bool:
        """Initialize all system components"""
        try:
            logger.info("="*60)
            logger.info("SYSTEM INITIALIZATION STARTING")
            logger.info("="*60)
            
            # Load configuration
            self.config = load_config(self.config_path)
            self.state.profile = self.config.get('_profile', 'unknown')
            self._update_component_status('config', 'loaded')
            
            # Initialize components in order
            initialization_steps = [
                ('logging', self._setup_logging),
                ('directories', self._create_directories),
                ('notifications', self._initialize_notifications),
                ('data_source', self._initialize_data_source),
                ('data_pipeline', self._initialize_data_pipeline),
                ('risk_management', self._initialize_risk_management),
                ('trading_model', self._initialize_trading_model),
                ('strategy', self._initialize_strategy),
                ('portfolio', self._initialize_portfolio),
                ('metrics', self._initialize_metrics),
                ('dashboard', self._initialize_dashboard)
            ]
            
            for step_name, step_func in initialization_steps:
                logger.info(f"Initializing {step_name}...")
                try:
                    if not step_func():
                        logger.error(f"Failed to initialize {step_name}")
                        if self.system_config.mode == "production":
                            return False
                        else:
                            logger.warning(f"Continuing in {self.system_config.mode} mode despite failure")
                    else:
                        self._update_component_status(step_name, 'ready')
                except Exception as e:
                    logger.error(f"Error initializing {step_name}: {e}")
                    self.state.add_error(str(e), step_name)
                    if self.system_config.mode == "production":
                        return False
            
            # Load previous state if exists
            if self.system_config.enable_state_persistence:
                self._load_state()
            
            # Validate system
            if not self._validate_system():
                logger.error("System validation failed")
                return False
            
            self.state.status = "ready"
            logger.info("="*60)
            logger.info("SYSTEM INITIALIZATION COMPLETE")
            logger.info("="*60)
            
            # Start API and register service
            if self.system_config.enable_dashboard:
                self._start_api_and_register()
            
            # Send startup notification
            self._send_notification(
                "System Started",
                f"Trading system initialized successfully\nMode: {self.system_config.mode}\nProfile: {self.state.profile}"
            )
            
            return True
            
        except Exception as e:
            logger.critical(f"Critical initialization error: {e}")
            logger.critical(traceback.format_exc())
            self.state.status = "error"
            self.state.add_error(f"Init failed: {str(e)}")
            return False
    
    def start(self):
        """Start the trading system"""
        if self.state.status != "ready":
            logger.error(f"Cannot start system in status: {self.state.status}")
            return
        
        logger.info("Starting trading system...")
        self.running = True
        self.state.status = "running"
        self.state.start_time = datetime.now()
        
        try:
            # Start worker threads
            self._start_workers()
            
            # Main event loop
            self._main_loop()
            
        except Exception as e:
            logger.error(f"System error: {e}")
            logger.error(traceback.format_exc())
            self.state.add_error(f"Runtime error: {str(e)}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the system gracefully"""
        logger.info("Stopping trading system...")
        self.running = False
        self.state.status = "stopping"
        
        try:
            # Stop workers
            self._stop_workers()
            
            # Close positions if needed
            if self.system_config.mode == "production" and not self.system_config.dry_run:
                self._close_all_positions()
            
            # Save final state
            if self.system_config.enable_state_persistence:
                self._save_state()
            
            # Disconnect from data sources
            if self.components['mt5_connector']:
                self.components['mt5_connector'].disconnect()
            
            # Stop dashboard
            if self.components['dashboard']:
                self.components['dashboard'].stop()
            
            # Generate final report
            self._generate_final_report()
            
            # Deregister from Consul
            if self.service_registry:
                try:
                    self.service_registry.deregister()
                    logger.info("✅ Service deregistered from Consul")
                except Exception as e:
                    logger.warning(f"⚠️ Service deregistration failed: {e}")
            
            # Send shutdown notification
            self._send_notification(
                "System Stopped",
                f"Trading system shut down\nTotal P&L: ${self.state.total_pnl:,.2f}\nTotal trades: {self.state.total_trades}"
            )
            
            self.state.status = "stopped"
            logger.info("System stopped successfully")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            self.state.add_error(f"Shutdown error: {str(e)}")
    
    # -------------------------------------------------------------------------
    # Component Initialization
    # -------------------------------------------------------------------------
    
    def _setup_logging(self) -> bool:
        """Setup logging configuration"""
        try:
            log_config = self.config.get('logging', {})
            log_file = None
            
            if log_config.get('file', {}).get('enabled', True):
                log_dir = Path(self.system_config.log_dir)
                log_dir.mkdir(exist_ok=True)
                log_file = log_dir / f"trading_{self.state.mode}_{datetime.now():%Y%m%d}.log"
            
            global logger
            logger = setup_logging(
                level=log_config.get('level', 'INFO'),
                log_file=str(log_file) if log_file else None,
                console=log_config.get('console', {}).get('enabled', True)
            )
            
            return True
        except Exception as e:
            print(f"Logging setup failed: {e}")
            return False
    
    def _create_directories(self) -> bool:
        """Create required directories"""
        try:
            dirs = [
                self.system_config.log_dir,
                self.system_config.data_dir,
                self.system_config.model_dir,
                self.system_config.report_dir,
                Path(self.system_config.state_file).parent,
                Path(self.system_config.metrics_file).parent
            ]
            
            for dir_path in dirs:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
            
            return True
        except Exception as e:
            logger.error(f"Failed to create directories: {e}")
            return False
    
    def _initialize_notifications(self) -> bool:
        """Initialize notification system"""
        if not self.system_config.enable_notifications:
            logger.info("Notifications disabled")
            return True
        
        try:
            # Import notification manager
            from src.utils.notifications import NotificationManager
            
            notif_config = self.config.get('notifications', {})
            self.components['notification_manager'] = NotificationManager()
            
            # Setup channels
            channels_setup = 0
            
            if notif_config.get('email', {}).get('enabled', False):
                self.components['notification_manager'].setup_email(notif_config['email'])
                channels_setup += 1
            
            if notif_config.get('telegram', {}).get('enabled', False):
                self.components['notification_manager'].setup_telegram(notif_config['telegram'])
                channels_setup += 1
            
            if notif_config.get('discord', {}).get('enabled', False):
                self.components['notification_manager'].setup_discord(notif_config['discord'])
                channels_setup += 1
            
            logger.info(f"Notifications initialized with {channels_setup} channels")
            return True
            
        except ImportError:
            logger.warning("Notification module not available")
            self.system_config.enable_notifications = False
            return True
        except Exception as e:
            logger.error(f"Failed to initialize notifications: {e}")
            return False
    
    def _initialize_data_source(self) -> bool:
        """Initialize data source (MT5 or simulator)"""
        try:
            if self.system_config.use_simulator:
                return self._initialize_simulator()
            else:
                # Try MT5 first
                if self._initialize_mt5():
                    return True
                
                # Fallback to simulator if MT5 fails
                logger.warning("MT5 connection failed, falling back to simulator")
                self.system_config.use_simulator = True
                return self._initialize_simulator()
                
        except Exception as e:
            logger.error(f"Failed to initialize data source: {e}")
            return False
    
    def _initialize_mt5(self) -> bool:
        """Initialize MT5 connection"""
        try:
            if not MODULES['mt5_connector']:
                logger.warning("MT5 connector module not available")
                return False
            
            mt5_config = self.config.get('mt5', {})
            
            self.components['mt5_connector'] = MODULES['mt5_connector'](
                login=mt5_config.get('login'),
                password=mt5_config.get('password'),
                server=mt5_config.get('server'),
                path=mt5_config.get('path'),
                timeout=mt5_config.get('timeout', 60000)
            )
            
            if self.components['mt5_connector'].connect():
                account_info = self.components['mt5_connector'].get_account_info()
                logger.info(f"Connected to MT5 - Account: {account_info.login}, "
                          f"Balance: ${account_info.balance:,.2f}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"MT5 initialization failed: {e}")
            return False
    
    def _initialize_simulator(self) -> bool:
        """Initialize market data simulator"""
        try:
            from src.core.data.data_simulator import MarketDataSimulator, SimulationConfig
            
            sim_config = SimulationConfig(
                symbol=self.config.get('acquisition', {}).get('symbol', 'USDCOP'),
                start_price=self.config.get('simulator', {}).get('base_price', 4000.0),
                volatility=self.config.get('simulator', {}).get('volatility', 0.1),
                use_garch=True,
                use_jumps=True,
                use_regime_switching=True
            )
            
            self.components['data_simulator'] = MarketDataSimulator(sim_config)
            logger.info("Market data simulator initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize simulator: {e}")
            return False
    
    def _initialize_data_pipeline(self) -> bool:
        """Initialize data pipeline"""
        try:
            # Try direct import first
            if MODULES['pipeline']:
                from src.markets.usdcop.pipeline import USDCOPPipeline
                self.components['data_pipeline'] = USDCOPPipeline(self.config)
                logger.info("Data pipeline initialized (direct)")
                return True
            
            # Fallback to subprocess validation
            result = subprocess.run(
                [sys.executable, "-m", "src.markets.usdcop.pipeline", "--help"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("Data pipeline available via CLI")
                return True
            
            logger.error("Data pipeline not available")
            return False
            
        except Exception as e:
            logger.error(f"Failed to initialize data pipeline: {e}")
            return False
    
    def _initialize_risk_management(self) -> bool:
        """Initialize risk management components"""
        try:
            risk_config = self.config.get('risk_management', {})
            
            # Risk Manager
            if MODULES['risk_manager']:
                self.components['risk_manager'] = MODULES['risk_manager'](
                    max_risk_per_trade=risk_config.get('position_sizing', {}).get('risk_percent', 1.0),
                    max_daily_loss=risk_config.get('max_exposure', {}).get('max_drawdown_percent', 10.0),
                    max_positions=risk_config.get('max_exposure', {}).get('max_trades', 3)
                )
                logger.info("Risk manager initialized")
            
            # Position Manager
            from src.core.position_manager import PositionManager
            self.components['position_manager'] = PositionManager(
                mt5_connector=self.components['mt5_connector'],
                risk_manager=self.components['risk_manager']
            )
            
            # Order Executor
            from src.core.order_executor import OrderExecutor
            self.components['order_executor'] = OrderExecutor(
                mt5_connector=self.components['mt5_connector'],
                position_manager=self.components['position_manager'],
                dry_run=self.system_config.dry_run
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize risk management: {e}")
            return False
    
    def _initialize_trading_model(self) -> bool:
        """Initialize RL trading model"""
        try:
            model_config = self.config.get('ml_config', {}).get('model', {})
            model_path = Path(self.system_config.model_dir) / model_config.get('name', 'rl_model.zip')
            
            # Try to load existing model
            if model_path.exists() and self.system_config.mode != "development":
                if MODULES['agent']:
                    from src.markets.usdcop.agent import RLTradingAgent
                    self.components['rl_agent'] = RLTradingAgent.load(str(model_path))
                    logger.info(f"Loaded existing model from {model_path}")
                    return True
            
            # Create new model for development/backtest
            if self.system_config.mode in ["development", "backtest"]:
                logger.info("Creating new RL model for development/backtest")
                # This would normally train a new model
                return True
            
            logger.error(f"No model found at {model_path}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to initialize trading model: {e}")
            return False
    
    def _initialize_strategy(self) -> bool:
        """Initialize trading strategy"""
        try:
            from src.strategies.rl_strategy import RLStrategy
            
            self.components['strategy'] = RLStrategy(
                rl_agent=self.components['rl_agent'],
                symbol=self.config.get('acquisition', {}).get('symbol', 'USDCOP'),
                timeframe=self.config.get('acquisition', {}).get('timeframe', 'M5'),
                risk_manager=self.components['risk_manager']
            )
            
            # Configure strategy
            strategy_config = self.config.get('trading', {})
            self.components['strategy'].set_parameters(
                confidence_threshold=strategy_config.get('confidence_threshold', 0.6),
                max_positions=strategy_config.get('max_positions', 3)
            )
            
            logger.info("Trading strategy initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize strategy: {e}")
            return False
    
    def _initialize_portfolio(self) -> bool:
        """Initialize portfolio manager"""
        try:
            if MODULES['portfolio']:
                self.components['portfolio_manager'] = MODULES['portfolio'](
                    initial_balance=self.config.get('backtesting', {}).get('initial_balance', 10000),
                    risk_manager=self.components['risk_manager']
                )
                logger.info("Portfolio manager initialized")
                return True
            
            logger.warning("Portfolio manager not available")
            return True  # Not critical
            
        except Exception as e:
            logger.error(f"Failed to initialize portfolio: {e}")
            return False
    
    def _initialize_metrics(self) -> bool:
        """Initialize metrics calculator"""
        try:
            if MODULES['metrics']:
                from src.markets.usdcop.metrics import TradingMetrics, MetricsConfig
                
                metrics_config = MetricsConfig(
                    risk_free_rate=self.config.get('metrics', {}).get('risk_free_rate', 0.02),
                    trading_days_year=252
                )
                
                self.components['metrics_calculator'] = TradingMetrics(metrics_config)
                logger.info("Metrics calculator initialized")
                return True
            
            logger.warning("Metrics module not available")
            return True  # Not critical
            
        except Exception as e:
            logger.error(f"Failed to initialize metrics: {e}")
            return False
    
    def _initialize_dashboard(self) -> bool:
        """Initialize web dashboard"""
        if not self.system_config.enable_dashboard:
            logger.info("Dashboard disabled")
            return True
        
        try:
            dashboard_config = self.config.get('monitoring', {}).get('dashboard', {})
            
            if not dashboard_config.get('enabled', True):
                return True
            
            if MODULES['dashboard']:
                self.components['dashboard'] = MODULES['dashboard'](
                    port=dashboard_config.get('port', 8000),
                    update_interval=dashboard_config.get('update_interval', 60)
                )
                
                # Start in separate thread
                dashboard_thread = threading.Thread(
                    target=self.components['dashboard'].run,
                    daemon=True
                )
                dashboard_thread.start()
                
                logger.info(f"Dashboard started at http://localhost:{dashboard_config.get('port', 8000)}")
                return True
            
            logger.warning("Dashboard module not available")
            return True  # Not critical
            
        except Exception as e:
            logger.error(f"Failed to initialize dashboard: {e}")
            return False
    
    # -------------------------------------------------------------------------
    # Worker Threads
    # -------------------------------------------------------------------------
    
    def _start_workers(self):
        """Start all worker threads"""
        logger.info("Starting worker threads...")
        
        worker_configs = [
            ('health_monitor', self._health_monitor_worker, self.system_config.health_check_interval),
            ('data_updater', self._data_update_worker, self.system_config.data_update_interval),
            ('position_monitor', self._position_monitor_worker, self.system_config.position_check_interval),
            ('strategy_executor', self._strategy_executor_worker, 1),
            ('metrics_calculator', self._metrics_worker, self.system_config.metrics_update_interval),
            ('state_saver', self._state_saver_worker, self.system_config.state_save_interval)
        ]
        
        if self.system_config.enable_ml_updates:
            worker_configs.append(
                ('model_updater', self._model_updater_worker, self.system_config.model_update_interval)
            )
        
        for name, worker_func, interval in worker_configs:
            self.workers[name] = self.executor.submit(self._worker_wrapper, name, worker_func, interval)
            logger.info(f"Started {name} worker (interval: {interval}s)")
        
        self.state.active_threads = len(self.workers)
        logger.info(f"All {len(self.workers)} workers started")
    
    def _stop_workers(self):
        """Stop all worker threads"""
        logger.info("Stopping worker threads...")
        self.running = False
        
        # Wait for workers to finish
        for name, future in self.workers.items():
            try:
                future.result(timeout=10)
                logger.info(f"Worker {name} stopped")
            except Exception as e:
                logger.error(f"Error stopping worker {name}: {e}")
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        logger.info("All workers stopped")
    
    def _worker_wrapper(self, name: str, worker_func, interval: float):
        """Generic worker wrapper with error handling"""
        logger.info(f"Worker {name} started")
        error_count = 0
        max_errors = 10
        
        while self.running:
            try:
                # Run worker function
                worker_func()
                error_count = 0  # Reset on success
                
                # Sleep for interval
                time.sleep(interval)
                
            except Exception as e:
                error_count += 1
                logger.error(f"Error in {name} worker: {e}")
                self.state.add_error(f"{name} error: {str(e)}", name)
                
                if error_count >= max_errors:
                    logger.critical(f"Worker {name} exceeded max errors, stopping")
                    break
                
                # Exponential backoff
                time.sleep(min(interval * (2 ** error_count), 300))
        
        logger.info(f"Worker {name} stopped")
    
    def _health_monitor_worker(self):
        """Monitor system health"""
        try:
            # Get system metrics
            self.state.cpu_percent = psutil.cpu_percent(interval=1)
            self.state.memory_percent = psutil.virtual_memory().percent
            self.state.disk_percent = psutil.disk_usage('/').percent
            
            # Check component health
            for name, component in self.components.items():
                if component is not None:
                    # Simple health check - component exists
                    self.state.component_status[name] = 'healthy'
                    
                    # Specific health checks
                    if name == 'mt5_connector' and hasattr(component, 'is_connected'):
                        if not component.is_connected():
                            logger.warning("MT5 disconnected, attempting reconnection...")
                            if component.connect():
                                logger.info("MT5 reconnected")
                            else:
                                self.state.component_status[name] = 'disconnected'
            
            # Check resource limits
            if self.state.memory_percent > self.system_config.max_memory_percent:
                logger.warning(f"High memory usage: {self.state.memory_percent:.1f}%")
                self._cleanup_memory()
            
            if self.state.cpu_percent > self.system_config.max_cpu_percent:
                logger.warning(f"High CPU usage: {self.state.cpu_percent:.1f}%")
            
            # Check error rate
            recent_errors = [e for e in self.state.errors 
                           if datetime.fromisoformat(e['timestamp']) > datetime.now() - timedelta(hours=1)]
            
            if len(recent_errors) > self.system_config.max_errors_per_hour:
                logger.critical(f"High error rate: {len(recent_errors)} errors in last hour")
                self._handle_critical_error("High error rate detected")
            
            # Write metrics file
            self._write_metrics()
            
            self.state.last_health_check = datetime.now()
            
        except Exception as e:
            logger.error(f"Health monitor error: {e}")
    
    def _data_update_worker(self):
        """Update market data"""
        try:
            if self.components['data_pipeline']:
                # Get latest data
                with self.locks['data']:
                    latest_data = self._fetch_latest_data()
                    
                    if latest_data is not None:
                        # Update components
                        if self.components['strategy']:
                            self.components['strategy'].update_data(latest_data)
                        
                        if self.components['risk_manager']:
                            self.components['risk_manager'].update_market_data(latest_data)
                        
                        # Update metrics queue
                        self.queues['metrics'].put({
                            'type': 'market_data',
                            'timestamp': datetime.now(),
                            'data': latest_data
                        })
            
        except Exception as e:
            logger.error(f"Data update error: {e}")
    
    def _position_monitor_worker(self):
        """Monitor and manage positions"""
        try:
            with self.locks['positions']:
                if self.components['position_manager']:
                    positions = self.components['position_manager'].get_open_positions()
                    self.state.active_positions = len(positions)
                    
                    # Check each position
                    for position in positions:
                        # Check stops and limits
                        self.components['position_manager'].check_position_limits(position)
                    
                    # Update P&L
                    total_pnl = self.components['position_manager'].get_total_pnl()
                    self.state.total_pnl = total_pnl
                    
                    # Calculate daily P&L
                    daily_pnl = self.components['position_manager'].get_daily_pnl()
                    self.state.daily_pnl = daily_pnl
                    
                    # Update drawdown
                    if total_pnl > self.state.peak_balance:
                        self.state.peak_balance = total_pnl
                    self.state.current_drawdown = (self.state.peak_balance - total_pnl) / self.state.peak_balance
            
        except Exception as e:
            logger.error(f"Position monitor error: {e}")
    
    def _strategy_executor_worker(self):
        """Execute trading strategy"""
        try:
            if self.state.status != "running":
                return
            
            if self.components['strategy']:
                # Generate signal
                signal = self.components['strategy'].generate_signal()
                
                if signal and signal.confidence > self.config.get('trading', {}).get('confidence_threshold', 0.6):
                    # Put signal in queue for processing
                    self.queues['signals'].put(signal)
                    
                    # Process signal
                    self._process_signal(signal)
            
        except Exception as e:
            logger.error(f"Strategy executor error: {e}")
    
    def _metrics_worker(self):
        """Calculate and update metrics"""
        try:
            if self.components['metrics_calculator'] and self.components['portfolio_manager']:
                # Get data
                trades = self.components['portfolio_manager'].get_trades()
                equity_curve = self.components['portfolio_manager'].get_equity_curve()
                
                if trades and len(equity_curve) > 0:
                    # Calculate metrics
                    metrics = self.components['metrics_calculator'].calculate_all_metrics(
                        trades,
                        equity_curve,
                        self.components['portfolio_manager'].initial_balance
                    )
                    
                    # Update state
                    self.state.total_trades = metrics.total_trades
                    self.state.winning_trades = metrics.winning_trades
                    self.state.losing_trades = metrics.losing_trades
                    
                    # Update dashboard
                    if self.components['dashboard']:
                        self.components['dashboard'].update_metrics(metrics)
                    
                    # Log key metrics
                    logger.info(
                        f"Metrics Update - "
                        f"Trades: {metrics.total_trades}, "
                        f"Win Rate: {metrics.win_rate:.1%}, "
                        f"Sharpe: {metrics.sharpe_ratio:.2f}, "
                        f"P&L: ${self.state.total_pnl:,.2f}"
                    )
            
        except Exception as e:
            logger.error(f"Metrics worker error: {e}")
    
    def _state_saver_worker(self):
        """Save system state periodically"""
        try:
            if self.system_config.enable_state_persistence:
                self._save_state()
        except Exception as e:
            logger.error(f"State saver error: {e}")
    
    def _model_updater_worker(self):
        """Update ML model periodically"""
        try:
            if self.components['rl_agent'] and self.components['data_pipeline']:
                logger.info("Starting model update...")
                
                # Get recent data
                recent_data = self._get_recent_training_data(days=30)
                
                if recent_data is not None:
                    # Update model
                    self.components['rl_agent'].update_model(recent_data)
                    
                    # Save updated model
                    model_path = Path(self.system_config.model_dir) / "rl_model_updated.zip"
                    self.components['rl_agent'].save(str(model_path))
                    
                    logger.info("Model updated successfully")
                    
                    # Send notification
                    self._send_notification(
                        "Model Updated",
                        "Trading model has been updated with recent market data"
                    )
            
        except Exception as e:
            logger.error(f"Model updater error: {e}")
    
    # -------------------------------------------------------------------------
    # Main Loop
    # -------------------------------------------------------------------------
    
    def _main_loop(self):
        """Main event loop"""
        logger.info("Main loop started")
        
        while self.running:
            try:
                # Update state
                with self.locks['state']:
                    self.state.last_update = datetime.now()
                
                # Process queues
                self._process_queues()
                
                # Sleep briefly
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                self.state.add_warning(f"Main loop error: {str(e)}")
        
        logger.info("Main loop ended")
    
    def _process_queues(self):
        """Process inter-thread communication queues"""
        # Process signals queue
        try:
            while not self.queues['signals'].empty():
                signal = self.queues['signals'].get_nowait()
                logger.debug(f"Processing queued signal: {signal}")
        except queue.Empty:
            pass
        
        # Process orders queue
        try:
            while not self.queues['orders'].empty():
                order = self.queues['orders'].get_nowait()
                logger.debug(f"Processing queued order: {order}")
        except queue.Empty:
            pass
    
    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------
    
    def _process_signal(self, signal):
        """Process a trading signal"""
        try:
            # Validate with risk manager
            if self.components['risk_manager'] and not self.components['risk_manager'].validate_signal(signal):
                logger.info(f"Signal rejected by risk manager: {signal}")
                return
            
            # Execute order
            with self.locks['orders']:
                if self.components['order_executor']:
                    result = self.components['order_executor'].execute_signal(signal)
                    
                    if result.success:
                        self.state.total_trades += 1
                        logger.info(f"Order executed: {result}")
                        
                        # Send notification for significant trades
                        if abs(result.price * result.volume) > 10000:  # Threshold
                            self._send_notification(
                                "Trade Executed",
                                f"Symbol: {signal.symbol}\n"
                                f"Side: {signal.side}\n"
                                f"Price: {result.price}\n"
                                f"Volume: {result.volume}"
                            )
                    else:
                        logger.warning(f"Order failed: {result.error}")
        
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
    
    def _fetch_latest_data(self):
        """Fetch latest market data"""
        try:
            if self.system_config.use_simulator and self.components['data_simulator']:
                # Get simulated data
                return self.components['data_simulator'].get_latest_bar()
            
            elif self.components['data_pipeline']:
                # Get real data
                return self.components['data_pipeline'].get_latest_data()
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return None
    
    def _get_recent_training_data(self, days: int = 30):
        """Get recent data for model training"""
        try:
            if self.components['data_pipeline']:
                return self.components['data_pipeline'].get_historical_data(
                    start_date=datetime.now() - timedelta(days=days),
                    end_date=datetime.now()
                )
            return None
        except Exception as e:
            logger.error(f"Error getting training data: {e}")
            return None
    
    def _validate_system(self) -> bool:
        """Validate system readiness"""
        critical_components = ['config']
        
        if self.system_config.mode == "production":
            critical_components.extend(['risk_manager', 'position_manager', 'order_executor'])
        
        # Check critical components
        for component in critical_components:
            if component not in self.state.component_status or \
               self.state.component_status[component] not in ['ready', 'loaded', 'healthy']:
                logger.error(f"Critical component not ready: {component}")
                return False
        
        # Check account balance if in production
        if self.system_config.mode == "production" and self.components['mt5_connector']:
            try:
                account_info = self.components['mt5_connector'].get_account_info()
                min_balance = self.config.get('requirements', {}).get('min_balance', 1000)
                
                if account_info.balance < min_balance:
                    logger.error(f"Insufficient balance: ${account_info.balance} < ${min_balance}")
                    return False
            except Exception as e:
                logger.error(f"Failed to check account balance: {e}")
                return False
        
        logger.info("System validation passed")
        return True
    
    def _cleanup_memory(self):
        """Clean up memory usage"""
        logger.info("Performing memory cleanup...")
        
        # Clear old data
        if self.components['data_pipeline'] and hasattr(self.components['data_pipeline'], 'cleanup_old_data'):
            self.components['data_pipeline'].cleanup_old_data(days=7)
        
        # Clear old errors/warnings
        if len(self.state.errors) > 1000:
            self.state.errors = self.state.errors[-1000:]
        if len(self.state.warnings) > 1000:
            self.state.warnings = self.state.warnings[-1000:]
        
        # Force garbage collection
        gc.collect()
        
        logger.info("Memory cleanup completed")
    
    def _write_metrics(self):
        """Write metrics in Prometheus format"""
        try:
            metrics = {
                'trading_system_up': 1 if self.state.status == "running" else 0,
                'trading_system_uptime_seconds': (datetime.now() - self.state.start_time).total_seconds(),
                'trading_system_total_trades': self.state.total_trades,
                'trading_system_active_positions': self.state.active_positions,
                'trading_system_total_pnl': self.state.total_pnl,
                'trading_system_daily_pnl': self.state.daily_pnl,
                'trading_system_current_drawdown': self.state.current_drawdown,
                'trading_system_cpu_percent': self.state.cpu_percent,
                'trading_system_memory_percent': self.state.memory_percent,
                'trading_system_disk_percent': self.state.disk_percent,
                'trading_system_active_threads': self.state.active_threads,
                'trading_system_error_count': len(self.state.errors),
                'trading_system_warning_count': len(self.state.warnings)
            }
            
            # Write to file
            metrics_path = Path(self.system_config.metrics_file)
            metrics_path.parent.mkdir(exist_ok=True)
            
            with open(metrics_path, 'w') as f:
                for key, value in metrics.items():
                    f.write(f"{key} {value}\n")
            
        except Exception as e:
            logger.error(f"Failed to write metrics: {e}")
    
    def _save_state(self):
        """Save system state to disk"""
        try:
            state_data = {
                'system_state': self.state.to_dict(),
                'config_profile': self.state.profile,
                'saved_at': datetime.now().isoformat()
            }
            
            # Add component states
            if self.components['portfolio_manager'] and hasattr(self.components['portfolio_manager'], 'get_state'):
                state_data['portfolio_state'] = self.components['portfolio_manager'].get_state()
            
            if self.components['strategy'] and hasattr(self.components['strategy'], 'get_state'):
                state_data['strategy_state'] = self.components['strategy'].get_state()
            
            # Save to file
            state_path = Path(self.system_config.state_file)
            state_path.parent.mkdir(exist_ok=True)
            
            # Save to temp file first
            temp_path = state_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            # Atomic rename
            temp_path.replace(state_path)
            
            logger.debug("System state saved")
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def _load_state(self):
        """Load previous system state"""
        try:
            state_path = Path(self.system_config.state_file)
            
            if not state_path.exists():
                logger.info("No previous state found")
                return
            
            with open(state_path, 'r') as f:
                state_data = json.load(f)
            
            # Restore component states
            if 'portfolio_state' in state_data and self.components['portfolio_manager']:
                if hasattr(self.components['portfolio_manager'], 'restore_state'):
                    self.components['portfolio_manager'].restore_state(state_data['portfolio_state'])
            
            if 'strategy_state' in state_data and self.components['strategy']:
                if hasattr(self.components['strategy'], 'restore_state'):
                    self.components['strategy'].restore_state(state_data['strategy_state'])
            
            logger.info(f"Previous state loaded from {state_data['saved_at']}")
            
        except Exception as e:
            logger.warning(f"Failed to load previous state: {e}")
    
    def _close_all_positions(self):
        """Close all open positions"""
        logger.warning("Closing all open positions...")
        
        if self.components['position_manager']:
            positions = self.components['position_manager'].get_open_positions()
            closed = 0
            failed = 0
            
            for position in positions:
                try:
                    result = self.components['position_manager'].close_position(position.ticket)
                    if result:
                        closed += 1
                        logger.info(f"Closed position {position.ticket}")
                    else:
                        failed += 1
                        logger.error(f"Failed to close position {position.ticket}")
                except Exception as e:
                    failed += 1
                    logger.error(f"Error closing position {position.ticket}: {e}")
            
            logger.info(f"Position closure summary: {closed} closed, {failed} failed")
    
    def _generate_final_report(self):
        """Generate comprehensive final report"""
        try:
            report_data = {
                'system_info': {
                    'mode': self.system_config.mode,
                    'profile': self.state.profile,
                    'start_time': self.state.start_time.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'runtime_hours': (datetime.now() - self.state.start_time).total_seconds() / 3600
                },
                'trading_summary': {
                    'total_trades': self.state.total_trades,
                    'winning_trades': self.state.winning_trades,
                    'losing_trades': self.state.losing_trades,
                    'win_rate': self.state.winning_trades / max(self.state.total_trades, 1),
                    'total_pnl': self.state.total_pnl,
                    'peak_balance': self.state.peak_balance,
                    'max_drawdown': self.state.current_drawdown
                },
                'system_health': {
                    'total_errors': len(self.state.errors),
                    'total_warnings': len(self.state.warnings),
                    'avg_cpu_percent': self.state.cpu_percent,
                    'avg_memory_percent': self.state.memory_percent
                }
            }
            
            # Generate detailed metrics if available
            if self.components['metrics_calculator'] and self.components['portfolio_manager']:
                try:
                    trades = self.components['portfolio_manager'].get_trades()
                    equity_curve = self.components['portfolio_manager'].get_equity_curve()
                    
                    if trades and len(equity_curve) > 0:
                        metrics = self.components['metrics_calculator'].calculate_all_metrics(
                            trades,
                            equity_curve,
                            self.components['portfolio_manager'].initial_balance
                        )
                        
                        report_data['detailed_metrics'] = {
                            'sharpe_ratio': metrics.sharpe_ratio,
                            'sortino_ratio': metrics.sortino_ratio,
                            'calmar_ratio': metrics.calmar_ratio,
                            'max_drawdown': metrics.max_drawdown,
                            'max_drawdown_duration': metrics.max_drawdown_duration,
                            'profit_factor': metrics.profit_factor,
                            'avg_win': metrics.avg_win,
                            'avg_loss': metrics.avg_loss,
                            'avg_trade_duration': metrics.avg_trade_duration
                        }
                except Exception as e:
                    logger.error(f"Failed to calculate detailed metrics: {e}")
            
            # Save report
            report_path = Path(self.system_config.report_dir) / f"final_report_{datetime.now():%Y%m%d_%H%M%S}.json"
            report_path.parent.mkdir(exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            logger.info(f"Final report saved to {report_path}")
            
            # Log summary
            logger.info("="*60)
            logger.info("FINAL PERFORMANCE SUMMARY")
            logger.info("="*60)
            logger.info(f"Total Trades: {self.state.total_trades}")
            logger.info(f"Win Rate: {report_data['trading_summary']['win_rate']:.1%}")
            logger.info(f"Total P&L: ${self.state.total_pnl:,.2f}")
            logger.info(f"Runtime: {report_data['system_info']['runtime_hours']:.1f} hours")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Failed to generate final report: {e}")
    
    def _start_api_and_register(self):
        """Start FastAPI app and register with Consul"""
        try:
            # Check if API dependencies are available
            if not all([FastAPI, uvicorn, create_health_monitor_router, ServiceRegistry, ServiceMeta]):
                logger.warning("API dependencies not available - skipping API startup")
                return
            
            # Create FastAPI app
            app = FastAPI(
                title="USDCOP Trading System API",
                description="Health monitoring and system status API",
                version="1.0.0"
            )
            
            # Mount existing health router
            health_router = create_health_monitor_router()
            app.include_router(health_router, prefix="/health", tags=["health"])
            
            # Add root health endpoint for Docker healthcheck
            @app.get("/health")
            async def root_health():
                return {"status": "healthy", "service": "usdcop-trading-system"}
            
            # Add SSE endpoint for events
            @app.get("/events/stream")
            async def sse_events(request):
                """Server-Sent Events endpoint for real-time events"""
                from fastapi import Request
                from fastapi.responses import StreamingResponse
                
                def event_generator():
                    """Generate SSE events from event bus"""
                    if not event_bus:
                        yield ": event bus not available\n\n"
                        return
                    
                    # Subscribe to all events
                    events_queue = []
                    
                    def event_handler(event):
                        """Handle events from bus"""
                        try:
                            import json
                            from dataclasses import asdict
                            sse_message = f"event: {event.event}\ndata: {json.dumps(asdict(event))}\n\n"
                            events_queue.append(sse_message)
                        except Exception as e:
                            logger.error(f"SSE event handling error: {e}")
                    
                    # Subscribe to event bus
                    event_bus.subscribe_all(event_handler)
                    
                    try:
                        # Send initial connection message
                        yield ": connected\n\n"
                        
                        # Stream events
                        while True:
                            if hasattr(request, 'is_disconnected') and request.is_disconnected():
                                break
                            
                            if events_queue:
                                yield events_queue.pop(0)
                            else:
                                # Send heartbeat
                                yield ": heartbeat\n\n"
                                time.sleep(30)
                                
                    except Exception as e:
                        logger.error(f"SSE stream error: {e}")
                    finally:
                        # Cleanup subscription
                        try:
                            event_bus.unsubscribe_all(event_handler)
                        except Exception:
                            pass
                
                return StreamingResponse(
                    event_generator(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Access-Control-Allow-Origin": "*"
                    }
                )
            
            def run_api():
                """Run uvicorn in background thread"""
                uvicorn.run(
                    app,
                    host="0.0.0.0",
                    port=self.api_port,
                    log_level="info",
                    access_log=False
                )
            
            # Start API server
            self.api_thread = threading.Thread(target=run_api, daemon=True)
            self.api_thread.start()
            
            # Wait for API to start
            time.sleep(2)
            
            # Register with Consul
            meta = ServiceMeta(
                version=os.getenv("APP_VERSION", "1.0.0"),
                capabilities=["trading", "health", "metrics", "data"],
                env=os.getenv("APP_ENV", "dev")
            )
            
            self.service_registry = ServiceRegistry(
                name=os.getenv("SERVICE_NAME", "usdcop-trading-system"),
                port=self.api_port,
                meta=meta,
                tags=[os.getenv("APP_ENV", "dev"), "trading", "primary"],
                health_path="/health/overview"
            )
            
            # Attempt registration
            if self.service_registry.register():
                logger.info("✅ Service registered successfully in Consul")
            else:
                logger.warning("⚠️ Service registration failed - continuing without Consul")
            
        except Exception as e:
            logger.error(f"Failed to start API and register service: {e}")
            # Don't fail startup - continue without API/Consul
    
    def _send_notification(self, title: str, message: str, urgent: bool = False):
        """Send notification through configured channels"""
        try:
            if self.components['notification_manager']:
                if urgent:
                    self.components['notification_manager'].send_urgent_notification(title, message)
                else:
                    self.components['notification_manager'].send_notification(title, message)
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
    
    def _handle_critical_error(self, error_msg: str):
        """Handle critical system errors"""
        logger.critical(f"CRITICAL ERROR: {error_msg}")
        
        # Send urgent notification
        self._send_notification(
            "CRITICAL SYSTEM ERROR",
            f"{error_msg}\n\nSystem may require manual intervention.",
            urgent=True
        )
        
        # In production mode, stop the system
        if self.system_config.mode == "production":
            logger.critical("Stopping system due to critical error")
            self.running = False
    
    def _update_component_status(self, component: str, status: str):
        """Update component status safely"""
        with self.locks['state']:
            self.state.component_status[component] = status
            logger.debug(f"Component {component} status updated to: {status}")
    
    def _install_signal_handlers(self):
        """Install signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

# -----------------------------------------------------------------------------
# Mode Runners
# -----------------------------------------------------------------------------

def run_live_mode(orchestrator: TradingSystemOrchestrator, args):
    """Run live trading mode"""
    logger.info("Starting LIVE trading mode")
    
    if orchestrator.initialize():
        orchestrator.start()
    else:
        logger.error("Failed to initialize system")
        return 1
    
    return 0

def run_backfill_mode(orchestrator: TradingSystemOrchestrator, args):
    """Run data backfill mode"""
    logger.info(f"Starting BACKFILL mode: {args.start} to {args.end}")
    
    if not orchestrator.initialize():
        return 1
    
    # Run backfill through pipeline
    try:
        if orchestrator.components['data_pipeline']:
            result = orchestrator.components['data_pipeline'].run_backfill(
                start_date=args.start,
                end_date=args.end,
                timeframe=args.timeframe
            )
            logger.info(f"Backfill completed: {result}")
        else:
            # Fallback to CLI
            cmd = [
                sys.executable, "-m", "src.markets.usdcop.pipeline", "backfill",
                "--start", args.start,
                "--end", args.end,
                "--timeframe", args.timeframe
            ]
            result = subprocess.run(cmd)
            return result.returncode
    except Exception as e:
        logger.error(f"Backfill failed: {e}")
        return 1
    
    return 0

def run_train_mode(orchestrator: TradingSystemOrchestrator, args):
    """Run model training mode"""
    logger.info(f"Starting TRAIN mode: {args.algo} for {args.steps} steps")
    
    if not orchestrator.initialize():
        return 1
    
    # Run training
    try:
        if MODULES['agent']:
            from src.markets.usdcop.agent import train_agent, AgentConfig
            
            config = AgentConfig(
                algo=args.algo,
                total_steps=args.steps,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                normalize_obs=True,
                normalize_reward=False
            )
            
            # Get training data
            data_path = Path(orchestrator.system_config.data_dir) / "gold" / "USDCOP" / args.timeframe
            
            # Train
            best_model = train_agent(str(data_path), args.timeframe, config)
            
            # Save
            model_path = Path(orchestrator.system_config.model_dir) / f"{args.algo}_model.zip"
            best_model.save(str(model_path))
            
            logger.info(f"Model saved to {model_path}")
        else:
            # Fallback to CLI
            cmd = [
                sys.executable, "-m", "src.markets.usdcop.agent", "train",
                "--algo", args.algo,
                "--total-steps", str(args.steps),
                "--data", str(Path(orchestrator.system_config.data_dir) / "gold" / "USDCOP" / args.timeframe)
            ]
            result = subprocess.run(cmd)
            return result.returncode
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1
    
    return 0

def run_backtest_mode(orchestrator: TradingSystemOrchestrator, args):
    """Run backtesting mode"""
    logger.info(f"Starting BACKTEST mode: {args.start} to {args.end}")
    
    if not orchestrator.initialize():
        return 1
    
    try:
        if MODULES['backtester']:
            from src.backtesting.advanced_backtester import BacktestConfig
            
            config = BacktestConfig(
                start_date=datetime.strptime(args.start, "%Y-%m-%d"),
                end_date=datetime.strptime(args.end, "%Y-%m-%d"),
                initial_balance=args.initial_balance,
                commission=args.commission,
                slippage_model='random'
            )
            
            backtester = MODULES['backtester'](config, orchestrator.config)
            results = backtester.run()
            
            # Display results
            logger.info("\n" + "="*60)
            logger.info("BACKTEST RESULTS")
            logger.info("="*60)
            logger.info(f"Total Return: {results['metrics'].total_return_pct:.2f}%")
            logger.info(f"Sharpe Ratio: {results['metrics'].sharpe_ratio:.2f}")
            logger.info(f"Max Drawdown: {results['metrics'].max_drawdown:.1%}")
            logger.info(f"Win Rate: {results['metrics'].win_rate:.1%}")
            logger.info(f"Total Trades: {results['metrics'].total_trades}")
            logger.info("="*60)
            
            # Save detailed report
            if results.get('report_path'):
                logger.info(f"Detailed report: {results['report_path']}")
        else:
            logger.error("Backtester module not available")
            return 1
            
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        return 1
    
    return 0

def run_dashboard_mode(orchestrator: TradingSystemOrchestrator, args):
    """Run dashboard only mode"""
    logger.info(f"Starting DASHBOARD mode on {args.host}:{args.port}")
    
    orchestrator.system_config.enable_dashboard = True
    
    if not orchestrator.initialize():
        return 1
    
    try:
        # Keep running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Dashboard stopped")
    
    return 0

# -----------------------------------------------------------------------------
# CLI Interface
# -----------------------------------------------------------------------------

def create_parser():
    """Create command line parser"""
    parser = argparse.ArgumentParser(
        description="USDCOP Trading System - Unified Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Live trading
  python scripts/run_system.py live --profile prod
  
  # Backfill historical data
  python scripts/run_system.py backfill --start 2024-01-01 --end 2024-12-31
  
  # Train model
  python scripts/run_system.py train --algo ppo --steps 1000000
  
  # Run backtest
  python scripts/run_system.py backtest --start 2024-01-01 --end 2024-12-31
  
  # Launch dashboard
  python scripts/run_system.py dashboard --port 8080
        """
    )
    
    # Global arguments
    parser.add_argument('--config', default='configs/mt5_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--profile', choices=['dev', 'staging', 'prod'],
                       help='Configuration profile to use')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', required=True,
                                      help='Operation mode')
    
    # Live mode
    live_parser = subparsers.add_parser('live', help='Run live trading')
    live_parser.add_argument('--dry-run', action='store_true',
                           help='Run without executing real trades')
    live_parser.add_argument('--simulator', action='store_true',
                           help='Use market simulator instead of real data')
    live_parser.add_argument('--no-dashboard', action='store_true',
                           help='Disable web dashboard')
    live_parser.add_argument('--no-notifications', action='store_true',
                           help='Disable notifications')
    
    # Backfill mode
    backfill_parser = subparsers.add_parser('backfill',
                                           help='Backfill historical data')
    backfill_parser.add_argument('--start', required=True,
                               help='Start date (YYYY-MM-DD)')
    backfill_parser.add_argument('--end', required=True,
                               help='End date (YYYY-MM-DD)')
    backfill_parser.add_argument('--timeframe', default='M5',
                               help='Timeframe to backfill')
    
    # Train mode
    train_parser = subparsers.add_parser('train', help='Train RL model')
    train_parser.add_argument('--algo', choices=['ppo', 'dqn', 'a2c'],
                            default='ppo', help='Algorithm to use')
    train_parser.add_argument('--steps', type=int, default=500000,
                            help='Total training steps')
    train_parser.add_argument('--timeframe', default='M5',
                            help='Data timeframe')
    train_parser.add_argument('--learning-rate', type=float, default=3e-4,
                            help='Learning rate')
    train_parser.add_argument('--batch-size', type=int, default=64,
                            help='Batch size')
    
    # Backtest mode
    backtest_parser = subparsers.add_parser('backtest',
                                           help='Run backtesting')
    backtest_parser.add_argument('--start', required=True,
                               help='Start date (YYYY-MM-DD)')
    backtest_parser.add_argument('--end', required=True,
                               help='End date (YYYY-MM-DD)')
    backtest_parser.add_argument('--initial-balance', type=float,
                               default=10000, help='Initial balance')
    backtest_parser.add_argument('--commission', type=float,
                               default=0.00003, help='Commission rate')
    
    # Dashboard mode
    dashboard_parser = subparsers.add_parser('dashboard',
                                           help='Launch dashboard only')
    dashboard_parser.add_argument('--host', default='0.0.0.0',
                                help='Dashboard host')
    dashboard_parser.add_argument('--port', type=int, default=8000,
                                help='Dashboard port')
    
    # Once mode
    once_parser = subparsers.add_parser('once',
                                       help='Run single pipeline cycle')
    
    # Metrics mode
    metrics_parser = subparsers.add_parser('metrics',
                                         help='Calculate metrics')
    metrics_parser.add_argument('--input', required=True,
                              help='Input data file')
    metrics_parser.add_argument('--type', choices=['trades', 'signals', 'equity'],
                              default='trades', help='Metrics type')
    
    return parser

def main():
    """Main entry point"""
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup initial logging
    global logger
    logger = setup_logging(
        level=args.log_level,
        log_file=f"logs/trading_{args.command}_{datetime.now():%Y%m%d_%H%M%S}.log",
        console=True
    )
    
    logger.info("="*60)
    logger.info(f"USDCOP Trading System v4.0.0")
    logger.info(f"Command: {args.command}")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Working Dir: {os.getcwd()}")
    logger.info("="*60)
    
    # Create system config
    system_config = SystemConfig()
    
    # Update based on command
    if args.command == 'live':
        system_config.mode = 'production'
        system_config.dry_run = args.dry_run
        system_config.use_simulator = args.simulator
        system_config.enable_dashboard = not args.no_dashboard
        system_config.enable_notifications = not args.no_notifications
    elif args.command in ['backtest', 'train']:
        system_config.mode = 'development'
        system_config.use_simulator = True
        system_config.enable_notifications = False
    
    # Create orchestrator
    orchestrator = TradingSystemOrchestrator(args.config, system_config)
    
    # Run appropriate mode
    try:
        if args.command == 'live':
            return run_live_mode(orchestrator, args)
        elif args.command == 'backfill':
            return run_backfill_mode(orchestrator, args)
        elif args.command == 'train':
            return run_train_mode(orchestrator, args)
        elif args.command == 'backtest':
            return run_backtest_mode(orchestrator, args)
        elif args.command == 'dashboard':
            return run_dashboard_mode(orchestrator, args)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        logger.critical(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())