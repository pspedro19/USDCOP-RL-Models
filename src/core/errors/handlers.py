"""
Unified Error Handling System
=============================
Provides comprehensive error handling, tracking, and recovery strategies
for the trading system.
"""

import logging
import traceback
import sys
from typing import Optional, Callable, Any, Dict, List, Tuple
from functools import wraps
from datetime import datetime, timezone
from collections import defaultdict, deque

# Try to import event bus, but make it optional
try:
    from ..events.bus import event_bus, Event, EventType
    HAS_EVENT_BUS = True
except ImportError:
    HAS_EVENT_BUS = False
    event_bus = None
    Event = None
    
    # Define EventType enum if not available
    from enum import Enum
    class EventType(Enum):
        ERROR = "ERROR"
        CRITICAL_ERROR = "CRITICAL_ERROR"
        WARNING = "WARNING"

logger = logging.getLogger(__name__)


# ===========================
# Custom Exception Classes
# ===========================

class TradingSystemError(Exception):
    """Base exception for trading system"""
    
    def __init__(self, message: str, code: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}
        self.timestamp = datetime.now(timezone.utc)


class DataError(TradingSystemError):
    """Data-related errors"""
    pass


class ModelError(TradingSystemError):
    """Model and ML-related errors"""
    pass


class IntegrationError(TradingSystemError):
    """Integration and communication errors"""
    pass


class ConfigurationError(TradingSystemError):
    """Configuration-related errors"""
    pass


class ConnectionError(TradingSystemError):
    """Connection and network errors"""
    pass


class ValidationError(TradingSystemError):
    """Data validation errors"""
    pass


# ===========================
# Error Handling Decorator
# ===========================

def handle_errors(
    component: str,
    fallback_value: Any = None,
    publish_event: bool = True,
    reraise: bool = True,
    max_retries: int = 0,
    log_level: str = "ERROR"
):
    """
    Decorator for comprehensive error handling.
    
    Args:
        component: Name of the component for logging
        fallback_value: Value to return on error
        publish_event: Whether to publish error event
        reraise: Whether to reraise the exception
        max_retries: Number of retries on failure
        log_level: Logging level for errors
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_count = 0
            last_error = None
            
            while retry_count <= max_retries:
                try:
                    return func(*args, **kwargs)
                    
                except TradingSystemError as e:
                    # Handle known errors
                    last_error = e
                    _log_error(component, func.__name__, e, log_level)
                    
                    if publish_event:
                        _publish_error_event(component, func.__name__, e, "known")
                    
                    if retry_count < max_retries:
                        retry_count += 1
                        logger.info(f"Retrying {func.__name__} (attempt {retry_count + 1}/{max_retries + 1})")
                        continue
                    
                    if reraise:
                        raise
                    return fallback_value
                    
                except Exception as e:
                    # Handle unexpected errors
                    last_error = e
                    logger.critical(
                        f"{component}: Unexpected error in {func.__name__}: {e}",
                        exc_info=True
                    )
                    
                    if publish_event:
                        _publish_error_event(component, func.__name__, e, "unexpected")
                    
                    if retry_count < max_retries:
                        retry_count += 1
                        logger.info(f"Retrying {func.__name__} (attempt {retry_count + 1}/{max_retries + 1})")
                        continue
                    
                    if reraise:
                        raise IntegrationError(f"Integration failed in {component}: {e}") from e
                    return fallback_value
            
            # If we get here, all retries failed
            if last_error:
                if reraise:
                    raise last_error
            return fallback_value
                
        return wrapper
    return decorator


def _log_error(component: str, function: str, error: Exception, level: str = "ERROR"):
    """Log error with appropriate level"""
    log_func = getattr(logger, level.lower(), logger.error)
    log_func(f"{component}.{function}: {error}", exc_info=True)


def _publish_error_event(component: str, function: str, error: Exception, error_type: str):
    """Publish error event to event bus"""
    if not HAS_EVENT_BUS or not event_bus:
        return
    
    try:
        event_type = EventType.CRITICAL_ERROR if error_type == "unexpected" else EventType.ERROR
        
        event_data = {
            'component': component,
            'function': function,
            'error_type': type(error).__name__,
            'error': str(error),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        if error_type == "unexpected":
            event_data['traceback'] = traceback.format_exc()
        
        if hasattr(error, 'code'):
            event_data['code'] = error.code
        
        if hasattr(error, 'details'):
            event_data['details'] = error.details
        
        event = Event(
            event=event_type.value,
            payload=event_data
        )
        
        event_bus.publish(event)
        
    except Exception as e:
        logger.warning(f"Failed to publish error event: {e}")


# ===========================
# Error Handler Class
# ===========================

class ErrorHandler:
    """Centralized error handler with tracking and statistics"""
    
    def __init__(self, max_history: int = 1000):
        self.error_count = defaultdict(int)
        self.error_history = deque(maxlen=max_history)
        self.last_errors = {}
        self.error_rate = defaultdict(lambda: deque(maxlen=100))
        self.start_time = datetime.now(timezone.utc)
        
    def handle(self, component: str, error: Exception, 
               context: Dict[str, Any] = None, severity: str = "ERROR") -> bool:
        """
        Handle an error with tracking.
        
        Args:
            component: Component where error occurred
            error: The exception that was raised
            context: Additional context information
            severity: Error severity level
            
        Returns:
            True if error was handled, False if it should be reraised
        """
        try:
            # Create error record
            error_record = {
                'component': component,
                'error': str(error),
                'type': type(error).__name__,
                'severity': severity,
                'context': context or {},
                'timestamp': datetime.now(timezone.utc),
                'traceback': traceback.format_exc() if severity == "CRITICAL" else None
            }
            
            # Track error count
            error_key = f"{component}:{type(error).__name__}"
            self.error_count[error_key] += 1
            
            # Store in history
            self.error_history.append(error_record)
            
            # Store last error per component
            self.last_errors[component] = error_record
            
            # Track error rate
            self.error_rate[component].append(datetime.now(timezone.utc))
            
            # Log error
            self._log_error(error_record)
            
            # Publish event
            self._publish_event(error_record)
            
            # Determine if we should handle or reraise
            return self._should_handle(component, error)
            
        except Exception as e:
            logger.error(f"Error in error handler: {e}")
            return False
    
    def _log_error(self, error_record: Dict[str, Any]):
        """Log error based on severity"""
        severity = error_record['severity']
        message = f"{error_record['component']}: {error_record['error']}"
        
        if severity == "CRITICAL":
            logger.critical(message, extra=error_record)
        elif severity == "ERROR":
            logger.error(message, extra=error_record)
        elif severity == "WARNING":
            logger.warning(message, extra=error_record)
        else:
            logger.info(message, extra=error_record)
    
    def _publish_event(self, error_record: Dict[str, Any]):
        """Publish error event"""
        if not HAS_EVENT_BUS or not event_bus:
            return
        
        try:
            event_type = {
                "CRITICAL": EventType.CRITICAL_ERROR,
                "ERROR": EventType.ERROR,
                "WARNING": EventType.WARNING
            }.get(error_record['severity'], EventType.ERROR)
            
            event = Event(
                event=event_type.value,
                payload=error_record
            )
            
            event_bus.publish(event)
            
        except Exception as e:
            logger.warning(f"Failed to publish error event: {e}")
    
    def _should_handle(self, component: str, error: Exception) -> bool:
        """
        Determine if error should be handled or reraised.
        Implements circuit breaker pattern.
        """
        error_key = f"{component}:{type(error).__name__}"
        error_count = self.error_count[error_key]
        
        # Circuit breaker thresholds
        if error_count > 100:
            # Too many errors, stop handling
            logger.critical(f"Circuit breaker open for {error_key}: {error_count} errors")
            return False
        
        # Check error rate
        recent_errors = len(self.error_rate[component])
        if recent_errors > 10:
            # Calculate time window
            time_window = (self.error_rate[component][-1] - self.error_rate[component][0]).total_seconds()
            if time_window > 0:
                error_rate = recent_errors / time_window
                if error_rate > 1.0:  # More than 1 error per second
                    logger.critical(f"High error rate for {component}: {error_rate:.2f} errors/sec")
                    return False
        
        return True
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        total_errors = sum(self.error_count.values())
        
        # Calculate error rates by component
        component_rates = {}
        for component, timestamps in self.error_rate.items():
            if len(timestamps) >= 2:
                time_span = (timestamps[-1] - timestamps[0]).total_seconds()
                if time_span > 0:
                    component_rates[component] = len(timestamps) / time_span
        
        # Find most common errors
        top_errors = sorted(
            self.error_count.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            'uptime_seconds': uptime,
            'total_errors': total_errors,
            'error_rate': total_errors / uptime if uptime > 0 else 0,
            'error_counts': dict(self.error_count),
            'component_error_rates': component_rates,
            'top_errors': top_errors,
            'last_errors': self.last_errors,
            'recent_errors': list(self.error_history)[-10:]
        }
    
    def reset_stats(self):
        """Reset error statistics"""
        self.error_count.clear()
        self.error_history.clear()
        self.last_errors.clear()
        self.error_rate.clear()
        self.start_time = datetime.now(timezone.utc)
        logger.info("Error statistics reset")
    
    def export_errors(self, filepath: str):
        """Export error history to file"""
        import json
        
        errors = []
        for error in self.error_history:
            error_dict = error.copy()
            error_dict['timestamp'] = error_dict['timestamp'].isoformat()
            errors.append(error_dict)
        
        with open(filepath, 'w') as f:
            json.dump(errors, f, indent=2)
        
        logger.info(f"Exported {len(errors)} errors to {filepath}")


# ===========================
# Global Error Handler
# ===========================

# Create global error handler instance
error_handler = ErrorHandler()


# ===========================
# Context Managers
# ===========================

class ErrorContext:
    """Context manager for error handling"""
    
    def __init__(self, component: str, operation: str = None,
                 reraise: bool = True, fallback: Any = None):
        self.component = component
        self.operation = operation or "operation"
        self.reraise = reraise
        self.fallback = fallback
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # An error occurred
            handled = error_handler.handle(
                self.component,
                exc_val,
                context={'operation': self.operation},
                severity="ERROR" if isinstance(exc_val, TradingSystemError) else "CRITICAL"
            )
            
            if not handled or self.reraise:
                return False  # Reraise the exception
            
            return True  # Suppress the exception
        
        return True


# ===========================
# Utility Functions
# ===========================

def safe_execute(func: Callable, *args, **kwargs) -> Tuple[bool, Any]:
    """
    Safely execute a function and return success status and result.
    
    Returns:
        Tuple of (success, result/error)
    """
    try:
        result = func(*args, **kwargs)
        return True, result
    except Exception as e:
        return False, e


def get_error_message(error: Exception) -> str:
    """Get formatted error message"""
    if isinstance(error, TradingSystemError):
        msg = str(error)
        if error.code:
            msg = f"[{error.code}] {msg}"
        if error.details:
            msg += f" | Details: {error.details}"
        return msg
    return str(error)


def log_and_raise(component: str, message: str, 
                  error_class: type = TradingSystemError,
                  **kwargs):
    """Log error and raise exception"""
    logger.error(f"{component}: {message}")
    raise error_class(message, **kwargs)