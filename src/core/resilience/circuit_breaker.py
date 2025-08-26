"""
Advanced Circuit Breaker Implementation
=====================================
Circuit breaker pattern with adaptive thresholds and health correlation.
"""

import time
import threading
import logging
from enum import Enum
from typing import Callable, Optional, Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states"""
    CLOSED = "CLOSED"           # Normal operation
    OPEN = "OPEN"               # Failing, reject requests
    HALF_OPEN = "HALF_OPEN"    # Testing recovery
    DEGRADED = "DEGRADED"       # Partial functionality


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    name: str
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3
    error_types: List[str] = field(default_factory=lambda: ["Exception"])
    health_impact: str = "critical"  # critical, degraded, warning
    adaptive_thresholds: bool = True
    min_threshold: int = 3
    max_threshold: int = 20
    load_factor: float = 1.0  # Multiplier based on system load
    metrics_enabled: bool = True


class CircuitBreaker:
    """Advanced circuit breaker with adaptive thresholds"""
    
    def __init__(self, config: CircuitBreakerConfig, 
                 on_state_change: Optional[Callable] = None,
                 metrics_callback: Optional[Callable] = None):
        self.config = config
        self.on_state_change = on_state_change
        self.metrics_callback = metrics_callback
        
        # State management
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._successes = 0
        self._opened_at = 0.0
        self._last_failure_time = 0.0
        self._lock = threading.RLock()
        
        # Adaptive thresholds
        self._current_threshold = config.failure_threshold
        self._load_history = []
        self._failure_history = []
        
        # Half-open testing
        self._half_open_calls = 0
        self._half_open_start = 0.0
        
        # Health correlation
        self._health_score = 100.0
        self._degradation_start = None
        
        logger.info(f"Circuit breaker '{config.name}' initialized with threshold {self._current_threshold}")
    
    @property
    def state(self) -> CircuitState:
        """Current circuit state"""
        return self._state
    
    @property
    def health_score(self) -> float:
        """Current health score (0-100)"""
        return self._health_score
    
    @property
    def is_healthy(self) -> bool:
        """Check if circuit is in healthy state"""
        return self._state in [CircuitState.CLOSED, CircuitState.DEGRADED]
    
    def _transition(self, new_state: CircuitState, reason: str = ""):
        """Transition to new state"""
        if new_state != self._state:
            old_state = self._state
            self._state = new_state
            
            logger.info(f"Circuit '{self.config.name}' state changed: {old_state} -> {new_state} ({reason})")
            
            # Notify state change
            if self.on_state_change:
                try:
                    self.on_state_change(old_state, new_state, self.config.name, reason)
                except Exception as e:
                    logger.warning(f"State change callback failed: {e}")
            
            # Update metrics
            if self.metrics_callback:
                try:
                    self.metrics_callback("circuit_state", {
                        "name": self.config.name,
                        "state": new_state.value,
                        "old_state": old_state.value,
                        "reason": reason
                    })
                except Exception as e:
                    logger.warning(f"Metrics callback failed: {e}")
    
    def _update_adaptive_threshold(self):
        """Update failure threshold based on load and failure patterns"""
        if not self.config.adaptive_thresholds:
            return
        
        with self._lock:
            # Calculate load factor
            current_load = self._calculate_load_factor()
            
            # Calculate failure rate
            failure_rate = self._calculate_failure_rate()
            
            # Adjust threshold
            base_threshold = self.config.failure_threshold
            load_adjustment = current_load * self.config.load_factor
            failure_adjustment = 1.0 + (failure_rate * 0.5)
            
            new_threshold = int(base_threshold * load_adjustment * failure_adjustment)
            new_threshold = max(self.config.min_threshold, 
                              min(self.config.max_threshold, new_threshold))
            
            if new_threshold != self._current_threshold:
                logger.info(f"Circuit '{self.config.name}' threshold adjusted: {self._current_threshold} -> {new_threshold}")
                self._current_threshold = new_threshold
    
    def _calculate_load_factor(self) -> float:
        """Calculate current load factor"""
        if not self._load_history:
            return 1.0
        
        # Use recent load history (last 10 measurements)
        recent_loads = self._load_history[-10:]
        return sum(recent_loads) / len(recent_loads)
    
    def _calculate_failure_rate(self) -> float:
        """Calculate recent failure rate"""
        if not self._failure_history:
            return 0.0
        
        # Use failures in last 5 minutes
        cutoff_time = time.time() - 300
        recent_failures = [f for f in self._failure_history if f > cutoff_time]
        
        if not recent_failures:
            return 0.0
        
        return len(recent_failures) / 5.0  # Failures per minute
    
    def _update_health_score(self, success: bool):
        """Update health score based on operation result"""
        if success:
            # Gradual recovery
            self._health_score = min(100.0, self._health_score + 2.0)
            if self._degradation_start:
                self._degradation_start = None
        else:
            # Gradual degradation
            self._health_score = max(0.0, self._health_score - 5.0)
            if not self._degradation_start:
                self._degradation_start = datetime.utcnow()
    
    def before_call(self) -> None:
        """Check if call should be allowed"""
        with self._lock:
            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if (time.time() - self._opened_at) >= self.config.recovery_timeout:
                    self._transition(CircuitState.HALF_OPEN, "recovery_timeout")
                    self._half_open_start = time.time()
                    self._half_open_calls = 0
                else:
                    raise RuntimeError(f"Circuit '{self.config.name}' is OPEN")
            
            elif self._state == CircuitState.HALF_OPEN:
                # Limit calls in half-open state
                if self._half_open_calls >= self.config.half_open_max_calls:
                    raise RuntimeError(f"Circuit '{self.config.name}' HALF_OPEN busy")
                
                # Check half-open timeout
                if (time.time() - self._half_open_start) >= (self.config.recovery_timeout / 2):
                    self._transition(CircuitState.OPEN, "half_open_timeout")
                    raise RuntimeError(f"Circuit '{self.config.name}' HALF_OPEN timeout")
                
                self._half_open_calls += 1
            
            elif self._state == CircuitState.DEGRADED:
                # Allow calls but with reduced capacity
                if self._health_score < 20:
                    raise RuntimeError(f"Circuit '{self.config.name}' DEGRADED critical")
    
    def after_call(self, success: bool, duration: float = 0.0, error: Optional[Exception] = None):
        """Process call result"""
        with self._lock:
            # Update adaptive threshold
            self._update_adaptive_threshold()
            
            # Update health score
            self._update_health_score(success)
            
            if success:
                self._successes += 1
                self._failures = 0
                
                # Record load factor
                self._load_history.append(duration)
                if len(self._load_history) > 100:
                    self._load_history.pop(0)
                
                # Transition to CLOSED if recovering
                if self._state in [CircuitState.OPEN, CircuitState.HALF_OPEN]:
                    self._transition(CircuitState.CLOSED, "success_recovery")
                    self._half_open_calls = 0
                
                # Check if should transition to DEGRADED
                elif self._state == CircuitState.CLOSED and self._health_score < 80:
                    self._transition(CircuitState.DEGRADED, "health_degradation")
                
                # Check if should return to CLOSED from DEGRADED
                elif self._state == CircuitState.DEGRADED and self._health_score > 90:
                    self._transition(CircuitState.CLOSED, "health_recovery")
                
            else:
                self._failures += 1
                self._last_failure_time = time.time()
                
                # Record failure
                self._failure_history.append(time.time())
                if len(self._failure_history) > 100:
                    self._failure_history.pop(0)
                
                # Check if should open circuit
                if self._failures >= self._current_threshold:
                    self._opened_at = time.time()
                    self._transition(CircuitState.OPEN, "failure_threshold_reached")
                
                # Check if should degrade
                elif self._state == CircuitState.CLOSED and self._health_score < 50:
                    self._transition(CircuitState.DEGRADED, "health_degradation")
            
            # Update metrics
            if self.metrics_callback:
                try:
                    self.metrics_callback("circuit_operation", {
                        "name": self.config.name,
                        "success": success,
                        "duration": duration,
                        "failures": self._failures,
                        "successes": self._successes,
                        "health_score": self._health_score,
                        "state": self._state.value
                    })
                except Exception as e:
                    logger.warning(f"Metrics callback failed: {e}")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function through the circuit breaker"""
        start_time = time.time()
        
        try:
            self.before_call()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            self.after_call(True, duration)
            return result
        except Exception as e:
            duration = time.time() - start_time
            self.after_call(False, duration, e)
            raise
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator for wrapping functions with circuit breaker"""
        def wrapped(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        
        return wrapped
    
    def force_open(self, reason: str = "manual"):
        """Force circuit to open"""
        with self._lock:
            self._opened_at = time.time()
            self._transition(CircuitState.OPEN, reason)
    
    def force_close(self, reason: str = "manual"):
        """Force circuit to close"""
        with self._lock:
            self._failures = 0
            self._successes = 0
            self._health_score = 100.0
            self._transition(CircuitState.CLOSED, reason)
    
    def force_degraded(self, reason: str = "manual"):
        """Force circuit to degraded state"""
        with self._lock:
            self._health_score = 50.0
            self._transition(CircuitState.DEGRADED, reason)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        with self._lock:
            return {
                "name": self.config.name,
                "state": self._state.value,
                "failures": self._failures,
                "successes": self._successes,
                "health_score": self._health_score,
                "current_threshold": self._current_threshold,
                "opened_at": self._opened_at,
                "last_failure_time": self._last_failure_time,
                "half_open_calls": self._half_open_calls,
                "load_history_length": len(self._load_history),
                "failure_history_length": len(self._failure_history)
            }
    
    def reset(self):
        """Reset circuit breaker to initial state"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failures = 0
            self._successes = 0
            self._opened_at = 0.0
            self._last_failure_time = 0.0
            self._half_open_calls = 0
            self._half_open_start = 0.0
            self._health_score = 100.0
            self._degradation_start = None
            self._current_threshold = self.config.failure_threshold
            
            logger.info(f"Circuit breaker '{self.config.name}' reset")
