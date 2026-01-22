# =============================================================================
# LoggerFactory - Centralized Factory for Configured Loggers
# =============================================================================
# Provides a unified logging interface for all services in the USDCOP Trading
# Platform. Outputs structured JSON logs compatible with Loki/Promtail.
#
# Key Features:
#   - JSON-formatted logs for structured querying in Loki
#   - Automatic context injection (timestamp, service, environment)
#   - Request ID propagation for distributed tracing
#   - Performance metrics extraction support
#   - Thread-safe singleton pattern with caching
#
# Configuration:
#   - LOG_LEVEL: DEBUG, INFO, WARNING, ERROR, CRITICAL
#   - LOG_FORMAT: json (default) or console
#   - SERVICE_NAME: Identifies the service in logs
#
# Author: Trading Team
# Date: 2026-01-16
# =============================================================================

import logging
import os
import sys
import json
import threading
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, Optional
from contextvars import ContextVar

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False


# =============================================================================
# Context Variables for Request Tracking
# =============================================================================
# These allow automatic propagation of context across async boundaries
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
service_name_var: ContextVar[str] = ContextVar("service_name", default="unknown")


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging compatible with Loki.

    Output format:
    {
        "timestamp": "2026-01-16T10:30:45.123456Z",
        "level": "INFO",
        "logger": "module.name",
        "message": "Log message here",
        "service": "trading-api",
        "environment": "production",
        "request_id": "abc123",
        "extra_field": "value"
    }
    """

    def __init__(self, service_name: str = "usdcop-trading", environment: str = "production"):
        super().__init__()
        self.service_name = service_name
        self.environment = environment
        self._default_keys = {
            "timestamp", "level", "logger", "message", "service",
            "environment", "request_id", "exc_info", "exc_text",
            "stack_info", "lineno", "funcName", "created", "filename",
            "module", "pathname", "process", "processName", "thread",
            "threadName", "name", "levelname", "levelno", "msg", "args"
        }

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON."""
        # Base log entry
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service_name,
            "environment": self.environment,
        }

        # Add request ID from context if available
        req_id = request_id_var.get()
        if req_id:
            log_entry["request_id"] = req_id

        # Add source location for DEBUG logs
        if record.levelno <= logging.DEBUG:
            log_entry["location"] = {
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName,
            }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add any extra fields passed to the logger
        for key, value in record.__dict__.items():
            if key not in self._default_keys and not key.startswith("_"):
                # Ensure values are JSON serializable
                try:
                    json.dumps(value)
                    log_entry[key] = value
                except (TypeError, ValueError):
                    log_entry[key] = str(value)

        return json.dumps(log_entry, default=str)


class ConsoleFormatter(logging.Formatter):
    """
    Human-readable console formatter for development.

    Output format:
    2026-01-16 10:30:45 | INFO     | module.name | Log message here | request_id=abc123
    """

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",
    }

    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record for console output."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname

        # Color the level if terminal supports it
        if self.use_colors:
            color = self.COLORS.get(level, "")
            reset = self.COLORS["RESET"]
            level_str = f"{color}{level:8s}{reset}"
        else:
            level_str = f"{level:8s}"

        # Build the log line
        parts = [
            timestamp,
            level_str,
            record.name,
            record.getMessage(),
        ]

        # Add request ID if available
        req_id = request_id_var.get()
        if req_id:
            parts.append(f"request_id={req_id}")

        # Add extra fields
        extras = []
        for key in ["duration_ms", "model_id", "action", "confidence", "error"]:
            if hasattr(record, key):
                extras.append(f"{key}={getattr(record, key)}")

        if extras:
            parts.append(" ".join(extras))

        return " | ".join(parts)


class ContextLogger(logging.Logger):
    """
    Extended Logger class that supports automatic context injection.

    Usage:
        logger = get_logger(__name__)
        logger.info("Processing trade", model_id="ppo_v1", action="BUY")
    """

    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False, **kwargs):
        """Override _log to support keyword arguments as extra fields."""
        if extra is None:
            extra = {}

        # Merge kwargs into extra
        extra.update(kwargs)

        super()._log(level, msg, args, exc_info=exc_info, extra=extra, stack_info=stack_info)


class LoggerFactory:
    """
    Centralized factory for configured loggers.

    This factory provides:
      - Thread-safe singleton configuration
      - Cached logger instances
      - Support for JSON and console output formats
      - Automatic context propagation

    Example:
        # Configure once at startup
        LoggerFactory.configure(
            level="INFO",
            json_format=True,
            service_name="trading-api"
        )

        # Get loggers throughout the application
        logger = LoggerFactory.get_logger(__name__)
        logger.info("Service started", port=8000)
    """

    _configured: bool = False
    _lock: threading.Lock = threading.Lock()
    _root_handler: Optional[logging.Handler] = None
    _service_name: str = "usdcop-trading"
    _environment: str = "production"

    @classmethod
    def configure(
        cls,
        level: str = "INFO",
        json_format: bool = True,
        service_name: Optional[str] = None,
        environment: Optional[str] = None,
    ) -> None:
        """
        Configure the logging system. Should be called once at application startup.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            json_format: If True, output JSON logs. If False, output human-readable console logs.
            service_name: Service identifier for logs. Defaults to SERVICE_NAME env var.
            environment: Environment identifier. Defaults to ENVIRONMENT env var.
        """
        with cls._lock:
            if cls._configured:
                return

            # Get configuration from environment or arguments
            cls._service_name = service_name or os.getenv("SERVICE_NAME", "usdcop-trading")
            cls._environment = environment or os.getenv("ENVIRONMENT", "production")
            log_level = os.getenv("LOG_LEVEL", level).upper()
            use_json = os.getenv("LOG_FORMAT", "json" if json_format else "console").lower() == "json"

            # Set up the custom logger class
            logging.setLoggerClass(ContextLogger)

            # Configure root logger
            root_logger = logging.getLogger()
            root_logger.setLevel(getattr(logging, log_level, logging.INFO))

            # Remove existing handlers
            root_logger.handlers.clear()

            # Create appropriate formatter
            if use_json:
                formatter = JSONFormatter(
                    service_name=cls._service_name,
                    environment=cls._environment,
                )
            else:
                formatter = ConsoleFormatter(use_colors=True)

            # Create and configure handler
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(formatter)
            handler.setLevel(getattr(logging, log_level, logging.INFO))

            root_logger.addHandler(handler)
            cls._root_handler = handler

            # Configure structlog if available (for enhanced structured logging)
            if STRUCTLOG_AVAILABLE:
                cls._configure_structlog(use_json)

            cls._configured = True

            # Log initialization
            logger = cls.get_logger("LoggerFactory")
            logger.info(
                "Logging configured",
                level=log_level,
                format="json" if use_json else "console",
                service=cls._service_name,
                environment=cls._environment,
            )

    @classmethod
    def _configure_structlog(cls, use_json: bool) -> None:
        """Configure structlog for enhanced structured logging."""
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.UnicodeDecoder(),
        ]

        if use_json:
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer(colors=True))

        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

    @classmethod
    @lru_cache(maxsize=128)
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get or create a logger with the given name.

        Args:
            name: Logger name, typically __name__ from the calling module.

        Returns:
            Configured Logger instance.

        Note:
            Results are cached for performance. The first call creates the logger,
            subsequent calls return the cached instance.
        """
        if not cls._configured:
            # Auto-configure with defaults if not explicitly configured
            cls.configure()

        return logging.getLogger(name)

    @classmethod
    def set_request_id(cls, request_id: str) -> None:
        """
        Set the request ID for the current context.

        This will be automatically included in all log messages from this context.

        Args:
            request_id: Unique identifier for the request.
        """
        request_id_var.set(request_id)

    @classmethod
    def clear_request_id(cls) -> None:
        """Clear the request ID from the current context."""
        request_id_var.set(None)

    @classmethod
    def get_request_id(cls) -> Optional[str]:
        """Get the current request ID, if set."""
        return request_id_var.get()


# =============================================================================
# Convenience Function
# =============================================================================
def get_logger(name: str) -> logging.Logger:
    """
    Convenience function to get a logger.

    This is the recommended way to get a logger in application code.

    Args:
        name: Logger name, typically __name__ from the calling module.

    Returns:
        Configured Logger instance.

    Example:
        from src.core.logging import get_logger

        logger = get_logger(__name__)
        logger.info("Processing request", request_id="abc123")
    """
    return LoggerFactory.get_logger(name)


# =============================================================================
# Context Manager for Request Tracking
# =============================================================================
class RequestContext:
    """
    Context manager for automatic request ID tracking.

    Usage:
        with RequestContext(request_id="abc123"):
            logger.info("Processing")  # Automatically includes request_id
    """

    def __init__(self, request_id: str):
        self.request_id = request_id
        self._token = None

    def __enter__(self):
        self._token = request_id_var.set(self.request_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        request_id_var.reset(self._token)
        return False


# =============================================================================
# Trading-Specific Logging Helpers
# =============================================================================
class TradingLogger:
    """
    Specialized logger for trading operations with predefined methods.

    Provides structured logging for common trading events:
      - Signal generation
      - Trade execution
      - Risk management
      - Model inference
    """

    def __init__(self, name: str):
        self._logger = get_logger(name)

    def signal(
        self,
        action: str,
        confidence: float,
        model_id: str,
        symbol: str = "USDCOP",
        **kwargs
    ) -> None:
        """Log a trading signal."""
        self._logger.info(
            "Trading signal generated",
            action=action,
            confidence=round(confidence, 4),
            model_id=model_id,
            symbol=symbol,
            **kwargs
        )

    def inference(
        self,
        model_id: str,
        duration_ms: float,
        action: Optional[str] = None,
        confidence: Optional[float] = None,
        **kwargs
    ) -> None:
        """Log an inference operation."""
        extra = {
            "model_id": model_id,
            "duration_ms": round(duration_ms, 2),
        }
        if action:
            extra["action"] = action
        if confidence is not None:
            extra["confidence"] = round(confidence, 4)
        extra.update(kwargs)

        self._logger.info("Model inference completed", **extra)

    def risk_event(
        self,
        event_type: str,
        reason: str,
        severity: str = "warning",
        **kwargs
    ) -> None:
        """Log a risk management event."""
        log_method = getattr(self._logger, severity.lower(), self._logger.warning)
        log_method(
            f"Risk event: {event_type}",
            event_type=event_type,
            reason=reason,
            **kwargs
        )

    def circuit_breaker(
        self,
        state: str,
        reason: str,
        **kwargs
    ) -> None:
        """Log circuit breaker state change."""
        level = "critical" if state == "OPEN" else "info"
        log_method = getattr(self._logger, level, self._logger.info)
        log_method(
            f"Circuit breaker {state}",
            circuit_breaker_state=state,
            reason=reason,
            **kwargs
        )


def get_trading_logger(name: str) -> TradingLogger:
    """Get a trading-specific logger instance."""
    return TradingLogger(name)
