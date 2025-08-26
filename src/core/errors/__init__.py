"""
Core Error Handling Module
=========================
Provides unified error handling and recovery strategies.
"""

from .handlers import (
    TradingSystemError,
    DataError,
    ModelError,
    IntegrationError,
    ConfigurationError,
    handle_errors,
    ErrorHandler,
    error_handler
)

__all__ = [
    'TradingSystemError',
    'DataError',
    'ModelError',
    'IntegrationError',
    'ConfigurationError',
    'handle_errors',
    'ErrorHandler',
    'error_handler'
]