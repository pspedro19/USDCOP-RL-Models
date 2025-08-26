"""
JSON Logging Formatter
======================
Structured JSON logging with trace context and correlation IDs.
"""
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional
from ..tracing.context import get_trace_context, create_correlation_id

class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def __init__(self, service_name: str = "unknown", service_version: str = "dev"):
        super().__init__()
        self.service_name = service_name
        self.service_version = service_version
        
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Get trace context
        trace_ctx = get_trace_context()
        
        # Create log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "service": self.service_name,
            "version": self.service_version,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "trace_id": trace_ctx.get("trace_id"),
            "span_id": trace_ctx.get("span_id"),
            "correlation_id": getattr(record, 'correlation_id', None),
            "request_id": getattr(record, 'request_id', None)
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 
                          'exc_text', 'stack_info', 'correlation_id', 'request_id']:
                if isinstance(value, (str, int, float, bool, type(None))):
                    log_entry[key] = value
                else:
                    log_entry[key] = str(value)
        
        return json.dumps(log_entry, default=str)

def setup_logging(service_name: str, service_version: str = "dev", 
                  log_level: str = None, log_file: str = None) -> logging.Logger:
    """Setup structured logging for a service."""
    # Get log level from environment or use default
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO")
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    
    # Console handler with JSON formatter
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_formatter = JsonFormatter(service_name, service_version)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = JsonFormatter(service_name, service_version)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Add redaction filter
    from .redactor import RedactFilter
    redact_filter = RedactFilter()
    logger.addFilter(redact_filter)
    
    logger.info(f"Logging initialized for {service_name} v{service_version}")
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)


# Alias for backward compatibility
JSONFormatter = JsonFormatter
