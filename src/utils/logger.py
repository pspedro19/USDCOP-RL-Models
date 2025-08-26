"""
Centralized Logger Module for USDCOP Trading System
===================================================

Production-ready logging system with trading-specific features:
- Multiple handlers (console with colors, rotating files, JSON)
- Trading-specific log levels (TRADE, METRIC, ALERT)
- Context management with correlation IDs
- Audit logging for compliance
- Secret redaction for security
- Performance metrics
- YAML/environment configuration
- Thread-safe operations
- FastAPI/async compatibility

Usage:
    from src.utils.logger import setup_logging, get_logger
    
    # Initialize once at startup
    setup_logging()
    
    # Get logger instance
    logger = get_logger(__name__)
    
    # Standard logging
    logger.info("System started")
    
    # Trading-specific logging
    logger.trade("BUY", "USDCOP", volume=0.1, price=4000.50)
    logger.metric("sharpe_ratio", 1.45)
    logger.alert("High volatility detected", volatility=0.025)
    
    # With context
    with logger.context(trade_id="123", strategy="momentum"):
        logger.info("Executing trade")
"""

import os
import re
import sys
import json
import time
import uuid
import logging
import logging.handlers
import warnings
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Callable
from datetime import datetime, timezone
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from functools import wraps
import contextvars

# Optional imports
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    import coloredlogs
    COLOREDLOGS_AVAILABLE = True
except ImportError:
    COLOREDLOGS_AVAILABLE = False


# ============================================
# CUSTOM LOG LEVELS
# ============================================

# Trading-specific log levels
TRADE_LEVEL = 25  # Between INFO and WARNING
METRIC_LEVEL = 26
ALERT_LEVEL = 35  # Between WARNING and ERROR
AUDIT_LEVEL = 27

logging.addLevelName(TRADE_LEVEL, "TRADE")
logging.addLevelName(METRIC_LEVEL, "METRIC")
logging.addLevelName(ALERT_LEVEL, "ALERT")
logging.addLevelName(AUDIT_LEVEL, "AUDIT")


# ============================================
# CONTEXT VARIABLES
# ============================================

# Thread-safe context variables
_correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar("correlation_id", default="")
_context_data: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar("context_data", default={})


# ============================================
# CONFIGURATION
# ============================================

@dataclass
class LoggerConfig:
    """Logger configuration with defaults"""
    # Main settings
    level: str = "INFO"
    console: bool = True
    console_level: Optional[str] = None
    color: bool = True
    
    # File settings
    file_enabled: bool = True
    file_path: str = "logs/app.log"
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 10
    
    # JSON logging
    json_enabled: bool = False
    json_path: Optional[str] = "logs/app.json"
    
    # Audit logging
    audit_enabled: bool = True
    audit_path: str = "logs/audit.log"
    audit_json: bool = True
    
    # Trading logs
    trade_log_enabled: bool = True
    trade_log_path: str = "logs/trades.log"
    
    # Features
    capture_warnings: bool = True
    capture_exceptions: bool = True
    include_process_info: bool = True
    performance_tracking: bool = True
    
    # Security
    redact_patterns: List[str] = field(default_factory=lambda: [
        r"(?i)(password|token|secret|key|authorization)[\s]*[=:]\s*([^\s,;]+)",
        r"(?i)Bearer\s+[A-Za-z0-9\-\._~\+\/]+=*",
        r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b",  # Credit card
    ])
    redact_keys: List[str] = field(default_factory=lambda: [
        "password", "token", "secret", "api_key", "authorization",
        "mt5_password", "MT5_PASSWORD", "private_key"
    ])
    
    # Environment
    env: str = field(default_factory=lambda: os.getenv("TRADING_ENV", "development"))
    profile: str = field(default_factory=lambda: os.getenv("APP_PROFILE", "dev"))


# ============================================
# FORMATTERS
# ============================================

class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'TRADE': '\033[34m',      # Blue
        'METRIC': '\033[35m',     # Magenta
        'AUDIT': '\033[94m',      # Light Blue
        'WARNING': '\033[33m',    # Yellow
        'ALERT': '\033[91m',      # Light Red
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[41m',   # Red background
    }
    RESET = '\033[0m'
    
    def __init__(self, use_color: bool = True):
        super().__init__(
            "[%(asctime)s] %(levelname)-8s %(name)s | %(cid)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.use_color = use_color
    
    def format(self, record):
        # Add correlation ID
        if not hasattr(record, 'cid'):
            cid = get_correlation_id()
            record.cid = f"CID:{cid}" if cid else "CID:----"
        
        # Add color
        if self.use_color and record.levelname in self.COLORS:
            levelname_color = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
            record.levelname = levelname_color
            result = super().format(record)
            record.levelname = record.levelname.replace(levelname_color, record.levelname)
        else:
            result = super().format(record)
        
        return result


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'thread_name': record.threadName,
            'process': record.process,
        }
        
        # Add correlation ID and context
        log_data['correlation_id'] = get_correlation_id() or None
        context = get_context_data()
        if context:
            log_data['context'] = context
        
        # Add exception info
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
            log_data['exc_type'] = record.exc_info[0].__name__
        
        # Add custom fields from extra
        standard_fields = set(vars(logging.LogRecord('', 0, '', 0, '', (), None)).keys())
        for key, value in record.__dict__.items():
            if key not in standard_fields and key not in log_data:
                try:
                    json.dumps(value)  # Check if serializable
                    log_data[key] = value
                except:
                    log_data[key] = str(value)
        
        return json.dumps(log_data, default=str)


# ============================================
# FILTERS
# ============================================

class ContextFilter(logging.Filter):
    """Add context and correlation ID to records"""
    
    def filter(self, record):
        # Add correlation ID
        record.correlation_id = get_correlation_id() or ""
        
        # Add context data
        context = get_context_data()
        for key, value in context.items():
            if not hasattr(record, key):
                setattr(record, key, value)
        
        return True


class RedactionFilter(logging.Filter):
    """Redact sensitive information from logs"""
    
    def __init__(self, config: LoggerConfig):
        super().__init__()
        self.config = config
        self.compiled_patterns = [re.compile(p) for p in config.redact_patterns]
    
    def filter(self, record):
        # Redact message
        if isinstance(record.msg, str):
            for pattern in self.compiled_patterns:
                record.msg = pattern.sub(r"\1=***REDACTED***", record.msg)
        
        # Redact args if they're a dict
        if isinstance(record.args, dict):
            record.args = self._redact_dict(record.args)
        
        return True
    
    def _redact_dict(self, data: dict) -> dict:
        """Redact sensitive keys from dictionary"""
        result = {}
        for key, value in data.items():
            if any(k.lower() in key.lower() for k in self.config.redact_keys):
                result[key] = "***REDACTED***"
            elif isinstance(value, dict):
                result[key] = self._redact_dict(value)
            else:
                result[key] = value
        return result


class TradingFilter(logging.Filter):
    """Filter for trading-specific logs"""
    
    def __init__(self, levels: List[int]):
        super().__init__()
        self.levels = levels
    
    def filter(self, record):
        return record.levelno in self.levels


# ============================================
# ENHANCED LOGGER
# ============================================

class TradingLogger:
    """Enhanced logger with trading-specific methods"""
    
    def __init__(self, logger: logging.Logger, config: LoggerConfig):
        self._logger = logger
        self._config = config
    
    # Delegate standard methods
    def __getattr__(self, name):
        return getattr(self._logger, name)
    
    # Trading-specific methods
    def trade(self, action: str, symbol: str, volume: float, price: float, **kwargs):
        """Log trading action"""
        extra = {
            'action': action,
            'symbol': symbol,
            'volume': volume,
            'price': price,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            **kwargs
        }
        self._logger.log(TRADE_LEVEL, 
            f"TRADE: {action} {volume} {symbol} @ {price}", 
            extra=extra
        )
    
    def metric(self, name: str, value: Union[float, int], unit: str = None, **kwargs):
        """Log metric"""
        extra = {
            'metric_name': name,
            'metric_value': value,
            'metric_unit': unit,
            **kwargs
        }
        msg = f"METRIC: {name}={value}"
        if unit:
            msg += f" {unit}"
        self._logger.log(METRIC_LEVEL, msg, extra=extra)
    
    def alert(self, message: str, severity: str = "HIGH", **kwargs):
        """Log alert"""
        extra = {
            'alert_severity': severity,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            **kwargs
        }
        self._logger.log(ALERT_LEVEL, f"ALERT [{severity}]: {message}", extra=extra)
    
    def audit(self, event: str, user: Optional[str] = None, **kwargs):
        """Log audit event"""
        extra = {
            'audit_event': event,
            'audit_user': user,
            'audit_timestamp': datetime.now(timezone.utc).isoformat(),
            **kwargs
        }
        self._logger.log(AUDIT_LEVEL, f"AUDIT: {event}", extra=extra)
    
    @contextmanager
    def context(self, **kwargs):
        """Context manager for adding context to logs"""
        old_context = get_context_data().copy()
        bind_context(**kwargs)
        try:
            yield
        finally:
            set_context_data(old_context)
    
    def timer(self, operation: str):
        """Context manager for timing operations"""
        @contextmanager
        def _timer():
            start = time.time()
            try:
                yield
                elapsed = (time.time() - start) * 1000
                self.metric(f"{operation}_duration_ms", elapsed, "ms", status="success")
            except Exception as e:
                elapsed = (time.time() - start) * 1000
                self.metric(f"{operation}_duration_ms", elapsed, "ms", 
                           status="error", error=str(e))
                raise
        return _timer()
    
    def performance(self, operation: str):
        """Decorator for logging operation performance"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self.timer(operation):
                    return func(*args, **kwargs)
            return wrapper
        return decorator


# ============================================
# SETUP AND FACTORY
# ============================================

_loggers: Dict[str, TradingLogger] = {}
_config: Optional[LoggerConfig] = None
_setup_done = False
_setup_lock = threading.Lock()


def load_config_from_yaml(yaml_path: str = "configs/mt5_config.yaml") -> Dict[str, Any]:
    """Load logging config from YAML file"""
    if not YAML_AVAILABLE or not os.path.exists(yaml_path):
        return {}
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f) or {}
    
    # Get logging section
    profile = os.getenv("APP_PROFILE", data.get("active_profile", "dev"))
    
    # Merge defaults with profile
    config = data.get("defaults", {}).get("logging", {})
    profile_config = data.get("profiles", {}).get(profile, {}).get("logging", {})
    config.update(profile_config)
    
    # Expand environment variables
    def expand_env(value):
        if isinstance(value, str):
            return os.path.expandvars(value)
        elif isinstance(value, dict):
            return {k: expand_env(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [expand_env(v) for v in value]
        return value
    
    return expand_env(config)


def setup_logging(config: Optional[LoggerConfig] = None, force: bool = False) -> LoggerConfig:
    """
    Initialize logging system
    
    Args:
        config: Optional configuration (uses defaults/YAML if None)
        force: Force re-initialization
    
    Returns:
        Effective configuration
    """
    global _config, _setup_done
    
    with _setup_lock:
        if _setup_done and not force:
            return _config
        
        # Build configuration
        if config is None:
            config = LoggerConfig()
            
            # Load from YAML if available
            yaml_config = load_config_from_yaml()
            if yaml_config:
                # Map YAML keys to config attributes
                for key, value in yaml_config.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
        
        # Apply environment overrides
        if os.getenv("LOG_LEVEL"):
            config.level = os.getenv("LOG_LEVEL")
        if os.getenv("LOG_FILE"):
            config.file_path = os.getenv("LOG_FILE")
        
        # Ensure directories exist
        for path in [config.file_path, config.json_path, config.audit_path, config.trade_log_path]:
            if path:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root = logging.getLogger()
        root.setLevel(getattr(logging, config.level.upper()))
        root.handlers.clear()
        
        # Add filters
        context_filter = ContextFilter()
        redaction_filter = RedactionFilter(config)
        
        # Console handler
        if config.console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_level = config.console_level or config.level
            console_handler.setLevel(getattr(logging, console_level.upper()))
            console_handler.setFormatter(ColoredFormatter(use_color=config.color))
            console_handler.addFilter(context_filter)
            console_handler.addFilter(redaction_filter)
            root.addHandler(console_handler)
        
        # File handler
        if config.file_enabled:
            file_handler = logging.handlers.RotatingFileHandler(
                config.file_path,
                maxBytes=config.max_bytes,
                backupCount=config.backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(getattr(logging, config.level.upper()))
            file_handler.setFormatter(logging.Formatter(
                "[%(asctime)s] %(levelname)-8s %(name)s:%(lineno)d | %(message)s"
            ))
            file_handler.addFilter(context_filter)
            file_handler.addFilter(redaction_filter)
            root.addHandler(file_handler)
        
        # JSON handler
        if config.json_enabled and config.json_path:
            json_handler = logging.handlers.RotatingFileHandler(
                config.json_path,
                maxBytes=config.max_bytes,
                backupCount=config.backup_count,
                encoding='utf-8'
            )
            json_handler.setLevel(getattr(logging, config.level.upper()))
            json_handler.setFormatter(JSONFormatter())
            json_handler.addFilter(context_filter)
            json_handler.addFilter(redaction_filter)
            root.addHandler(json_handler)
        
        # Audit handler
        if config.audit_enabled:
            audit_handler = logging.handlers.RotatingFileHandler(
                config.audit_path,
                maxBytes=config.max_bytes,
                backupCount=config.backup_count,
                encoding='utf-8'
            )
            audit_handler.setLevel(AUDIT_LEVEL)
            if config.audit_json:
                audit_handler.setFormatter(JSONFormatter())
            else:
                audit_handler.setFormatter(logging.Formatter(
                    "[%(asctime)s] AUDIT | %(message)s"
                ))
            audit_handler.addFilter(context_filter)
            audit_handler.addFilter(TradingFilter([AUDIT_LEVEL]))
            root.addHandler(audit_handler)
        
        # Trading handler
        if config.trade_log_enabled:
            trade_handler = logging.handlers.RotatingFileHandler(
                config.trade_log_path,
                maxBytes=config.max_bytes,
                backupCount=config.backup_count,
                encoding='utf-8'
            )
            trade_handler.setLevel(TRADE_LEVEL)
            trade_handler.setFormatter(JSONFormatter())
            trade_handler.addFilter(context_filter)
            trade_handler.addFilter(TradingFilter([TRADE_LEVEL]))
            root.addHandler(trade_handler)
        
        # Capture warnings and exceptions
        if config.capture_warnings:
            logging.captureWarnings(True)
            warnings.filterwarnings('default')
        
        if config.capture_exceptions:
            sys.excepthook = _exception_hook
        
        _config = config
        _setup_done = True
        
        # Log initialization
        logger = get_logger("logger")
        logger.info(f"Logging system initialized", extra={
            "environment": config.env,
            "profile": config.profile,
            "level": config.level,
            "handlers": len(root.handlers)
        })
        
        return config


def get_logger(name: str) -> TradingLogger:
    """Get or create logger instance"""
    if not _setup_done:
        setup_logging()
    
    if name not in _loggers:
        base_logger = logging.getLogger(name)
        _loggers[name] = TradingLogger(base_logger, _config)
    
    return _loggers[name]


# ============================================
# CONTEXT MANAGEMENT
# ============================================

def set_correlation_id(cid: Optional[str] = None) -> str:
    """Set correlation ID for current context"""
    if cid is None:
        cid = str(uuid.uuid4())[:8]
    _correlation_id.set(cid)
    return cid


def get_correlation_id() -> str:
    """Get current correlation ID"""
    return _correlation_id.get()


def clear_correlation_id():
    """Clear correlation ID"""
    _correlation_id.set("")


def bind_context(**kwargs):
    """Add key-value pairs to context"""
    context = _context_data.get().copy()
    context.update(kwargs)
    _context_data.set(context)


def get_context_data() -> Dict[str, Any]:
    """Get current context data"""
    return _context_data.get().copy()


def set_context_data(context: Dict[str, Any]):
    """Set context data"""
    _context_data.set(context)


def clear_context():
    """Clear all context data"""
    _context_data.set({})


# ============================================
# UTILITIES
# ============================================

def _exception_hook(exc_type, exc_value, exc_traceback):
    """Global exception handler"""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    logger = get_logger("system")
    logger.error(
        "Uncaught exception",
        exc_info=(exc_type, exc_value, exc_traceback)
    )


class with_correlation_id:
    """Context manager for correlation ID"""
    def __init__(self, cid: Optional[str] = None):
        self.cid = cid
        self.old_cid = None
    
    def __enter__(self):
        self.old_cid = get_correlation_id()
        return set_correlation_id(self.cid)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.old_cid:
            set_correlation_id(self.old_cid)
        else:
            clear_correlation_id()


def log_exceptions(logger: Optional[TradingLogger] = None, reraise: bool = True):
    """Decorator to log exceptions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Exception in {func.__name__}",
                    exc_info=True,
                    extra={
                        'function': func.__name__,
                        'module': func.__module__
                    }
                )
                if reraise:
                    raise
        return wrapper
    return decorator


# ============================================
# FASTAPI INTEGRATION
# ============================================

async def correlation_middleware(request, call_next):
    """FastAPI middleware for correlation IDs"""
    # Get or create correlation ID
    cid = request.headers.get("X-Correlation-ID", str(uuid.uuid4())[:8])
    set_correlation_id(cid)
    
    # Add request context
    bind_context(
        method=request.method,
        path=str(request.url.path),
        client=request.client.host if request.client else None
    )
    
    try:
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = cid
        return response
    finally:
        clear_context()
        clear_correlation_id()


# ============================================
# CLI INTERFACE
# ============================================

def demo():
    """Demo logging functionality"""
    setup_logging(force=True)
    logger = get_logger("demo")
    
    # Basic logging
    logger.debug("Debug message")
    logger.info("Info message with password=secret123")  # Will be redacted
    logger.warning("Warning message")
    logger.error("Error message")
    
    # Trading logs
    with with_correlation_id("DEMO-001"):
        logger.trade("BUY", "USDCOP", 0.1, 4000.50, strategy="momentum")
        logger.metric("sharpe_ratio", 1.45)
        logger.alert("High volatility detected", severity="MEDIUM", volatility=0.025)
        logger.audit("user_login", user="trader1", ip="192.168.1.1")
    
    # Context example
    with logger.context(session_id="12345", strategy="RL_PPO"):
        logger.info("Processing trading signals")
        with logger.timer("signal_processing"):
            time.sleep(0.1)
        logger.trade("SELL", "USDCOP", 0.05, 4001.00)
    
    # Performance decorator
    @logger.performance("data_processing")
    def process_data():
        time.sleep(0.05)
        return "processed"
    
    result = process_data()
    logger.info(f"Result: {result}")
    
    # Exception handling
    @log_exceptions(logger)
    def risky_operation():
        raise ValueError("Simulated error")
    
    try:
        risky_operation()
    except ValueError:
        pass
    
    logger.info("Demo completed successfully")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        print("Usage: python -m src.utils.logger demo")