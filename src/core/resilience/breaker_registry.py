"""
Circuit Breaker Registry
=======================
Global registry for managing circuit breakers across services.
"""

import logging
from typing import Dict, Optional, Callable
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig

logger = logging.getLogger(__name__)


class CircuitBreakerRegistry:
    """Global registry for circuit breakers"""
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._global_callbacks: Dict[str, Callable] = {}
        
        logger.info("Circuit breaker registry initialized")
    
    def register(self, name: str, breaker: CircuitBreaker) -> None:
        """Register a circuit breaker"""
        self._breakers[name] = breaker
        logger.info(f"Registered circuit breaker: {name}")
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name"""
        return self._breakers.get(name)
    
    def create_and_register(self, config: CircuitBreakerConfig,
                           on_state_change: Optional[Callable] = None,
                           metrics_callback: Optional[Callable] = None) -> CircuitBreaker:
        """Create and register a circuit breaker"""
        breaker = CircuitBreaker(config, on_state_change, metrics_callback)
        self.register(config.name, breaker)
        return breaker
    
    def unregister(self, name: str) -> bool:
        """Unregister a circuit breaker"""
        if name in self._breakers:
            del self._breakers[name]
            logger.info(f"Unregistered circuit breaker: {name}")
            return True
        return False
    
    def list_breakers(self) -> list:
        """List all registered breaker names"""
        return list(self._breakers.keys())
    
    def get_all_stats(self) -> Dict[str, dict]:
        """Get statistics for all breakers"""
        return {name: breaker.get_stats() for name, breaker in self._breakers.items()}
    
    def get_health_summary(self) -> Dict[str, str]:
        """Get health summary for all breakers"""
        summary = {}
        for name, breaker in self._breakers.items():
            if breaker.state.value == "OPEN":
                summary[name] = "critical"
            elif breaker.state.value == "DEGRADED":
                summary[name] = "degraded"
            elif breaker.state.value == "HALF_OPEN":
                summary[name] = "warning"
            else:
                summary[name] = "healthy"
        return summary
    
    def force_open_all(self, reason: str = "manual"):
        """Force all breakers to open state"""
        for name, breaker in self._breakers.items():
            breaker.force_open(reason)
        logger.warning(f"All circuit breakers forced open: {reason}")
    
    def force_close_all(self, reason: str = "manual"):
        """Force all breakers to close state"""
        for name, breaker in self._breakers.items():
            breaker.force_close(reason)
        logger.info(f"All circuit breakers forced closed: {reason}")
    
    def reset_all(self):
        """Reset all breakers to initial state"""
        for name, breaker in self._breakers.items():
            breaker.reset()
        logger.info("All circuit breakers reset")
    
    def add_global_callback(self, event: str, callback: Callable):
        """Add a global callback for circuit breaker events"""
        self._global_callbacks[event] = callback
        logger.info(f"Added global callback for event: {event}")
    
    def remove_global_callback(self, event: str):
        """Remove a global callback"""
        if event in self._global_callbacks:
            del self._global_callbacks[event]
            logger.info(f"Removed global callback for event: {event}")
    
    def _notify_global_callbacks(self, event: str, data: dict):
        """Notify global callbacks of an event"""
        if event in self._global_callbacks:
            try:
                self._global_callbacks[event](data)
            except Exception as e:
                logger.warning(f"Global callback for {event} failed: {e}")


# Global instance
_global_registry: Optional[CircuitBreakerRegistry] = None


def get_global_registry() -> CircuitBreakerRegistry:
    """Get the global circuit breaker registry"""
    global _global_registry
    if _global_registry is None:
        _global_registry = CircuitBreakerRegistry()
    return _global_registry


def register_breaker(name: str, breaker: CircuitBreaker) -> None:
    """Register a breaker in the global registry"""
    get_global_registry().register(name, breaker)


def get_breaker(name: str) -> Optional[CircuitBreaker]:
    """Get a breaker from the global registry"""
    return get_global_registry().get(name)


def create_breaker(config: CircuitBreakerConfig,
                  on_state_change: Optional[Callable] = None,
                  metrics_callback: Optional[Callable] = None) -> CircuitBreaker:
    """Create and register a breaker in the global registry"""
    return get_global_registry().create_and_register(config, on_state_change, metrics_callback)
