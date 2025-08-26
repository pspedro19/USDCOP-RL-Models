#!/usr/bin/env python3
"""
Observability Tests
===================
Tests for monitoring, logging, and tracing components.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.observability.metrics.prometheus_registry import PrometheusRegistry
from src.core.observability.logging.json_formatter import JSONFormatter
from src.core.observability.tracing.tracer import Tracer


class TestMetrics:
    """Test metrics collection"""
    
    def test_prometheus_registry(self):
        """Test Prometheus registry"""
        registry = PrometheusRegistry()
        
        # Register a counter
        counter = registry.register_counter(
            'test_counter',
            'Test counter metric'
        )
        
        assert counter is not None
        counter.inc()


class TestLogging:
    """Test logging functionality"""
    
    def test_json_formatter(self):
        """Test JSON log formatter"""
        formatter = JSONFormatter()
        
        import logging
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        assert "Test message" in formatted


class TestTracing:
    """Test distributed tracing"""
    
    def test_tracer_initialization(self):
        """Test tracer initialization"""
        tracer = Tracer("test-service")
        assert tracer is not None
        assert hasattr(tracer, 'start_span')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])