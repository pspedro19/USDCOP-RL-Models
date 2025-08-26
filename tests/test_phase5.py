#!/usr/bin/env python3
"""
Phase 5 Advanced Features Test
===============================
Tests for rate limiting, DLQ, and graceful shutdown features.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.ratelimit.redis_limiter import RedisRateLimiter
from src.core.messaging.dlq.dlq_manager import DLQManager
from src.core.lifecycle.shutdown_manager import ShutdownManager


class TestRateLimiting:
    """Test rate limiting functionality"""
    
    def test_token_bucket(self):
        """Test token bucket rate limiter"""
        limiter = RedisRateLimiter(max_requests=10, window_seconds=60)
        
        # Should allow first 10 requests
        for _ in range(10):
            assert limiter.allow_request("test_key")
        
        # Should block 11th request
        assert not limiter.allow_request("test_key")


class TestDeadLetterQueue:
    """Test DLQ functionality"""
    
    def test_dlq_retry(self):
        """Test DLQ retry mechanism"""
        dlq = DLQManager()
        
        # Add message to DLQ
        message_id = dlq.add_message({
            "data": "test",
            "error": "Processing failed"
        })
        
        assert message_id is not None
        
        # Retry message
        success = dlq.retry_message(message_id)
        assert success


class TestGracefulShutdown:
    """Test graceful shutdown"""
    
    def test_shutdown_sequence(self):
        """Test shutdown sequence"""
        manager = ShutdownManager()
        
        # Register shutdown handlers
        handler_called = False
        
        def test_handler():
            nonlocal handler_called
            handler_called = True
        
        manager.register_handler(test_handler)
        
        # Trigger shutdown
        manager.initiate_shutdown()
        
        assert handler_called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])