#!/usr/bin/env python3
"""
Resilience Tests
================
Tests for circuit breakers, fallback strategies, and resilience patterns.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from src.core.resilience.fallback_strategies import FallbackManager


class TestCircuitBreaker:
    """Test circuit breaker pattern"""
    
    def test_circuit_breaker_states(self):
        """Test circuit breaker state transitions"""
        config = CircuitBreakerConfig(
            name="test_breaker",
            failure_threshold=3,
            recovery_timeout=60
        )
        breaker = CircuitBreaker(config)
        
        # Initially closed
        assert breaker.state.value == "CLOSED"
        
        # Record failures using call method
        def failing_func():
            raise Exception("Test failure")
        
        for _ in range(3):
            try:
                breaker.call(failing_func)
            except:
                pass
        
        # Should be open after threshold
        assert breaker.state.value == "OPEN"
        
        # Should not allow calls when open (will raise exception)
        with pytest.raises(Exception):
            breaker.call(lambda: "test")


class TestFallbackStrategies:
    """Test fallback strategies"""
    
    def test_fallback_chain(self):
        """Test fallback chain execution"""
        manager = FallbackManager()
        
        # Define fallback chain
        def primary():
            raise Exception("Primary failed")
        
        def secondary():
            return "Secondary success"
        
        # Register custom strategies
        manager.register_strategy("primary", primary)
        manager.register_strategy("secondary", secondary)
        
        # Execute with fallback - try primary, fallback to secondary
        try:
            result = manager.execute("primary")
        except:
            result = manager.execute("secondary")
            
        assert result == "Secondary success"


class TestResilience:
    """Test overall resilience patterns"""
    
    def test_retry_with_backoff(self):
        """Test retry with exponential backoff"""
        from src.core.resilience import retry_with_backoff
        
        attempt_count = 0
        
        @retry_with_backoff(max_retries=3)
        def flaky_operation():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception("Temporary failure")
            return "Success"
        
        result = flaky_operation()
        assert result == "Success"
        assert attempt_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])