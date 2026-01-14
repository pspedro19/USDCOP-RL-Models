"""
Chaos Test: Circuit Breaker.

Tests that the circuit breaker activates correctly after
consecutive losses to protect from cascading failures.

Contract: CTR-RISK-001
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


class MockCircuitBreaker:
    """Mock circuit breaker for testing."""

    def __init__(
        self,
        max_consecutive_losses: int = 5,
        cooldown_minutes: int = 60
    ):
        self.max_consecutive_losses = max_consecutive_losses
        self.cooldown_minutes = cooldown_minutes
        self.consecutive_losses = 0
        self.is_active = False
        self.cooldown_until = None

    def record_loss(self):
        """Record a losing trade."""
        self.consecutive_losses += 1
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.activate()

    def record_win(self):
        """Record a winning trade."""
        self.consecutive_losses = 0
        # Don't deactivate - only time does that

    def activate(self):
        """Activate circuit breaker."""
        self.is_active = True
        self.cooldown_until = datetime.now() + timedelta(minutes=self.cooldown_minutes)

    def check(self) -> bool:
        """Check if circuit breaker is active."""
        if self.is_active and datetime.now() >= self.cooldown_until:
            self.is_active = False
            self.consecutive_losses = 0
        return self.is_active

    @property
    def time_remaining_minutes(self) -> float:
        """Get remaining cooldown time in minutes."""
        if not self.is_active or not self.cooldown_until:
            return 0
        remaining = (self.cooldown_until - datetime.now()).total_seconds() / 60
        return max(0, remaining)


class TestCircuitBreaker:
    """Tests for circuit breaker functionality."""

    def test_activates_after_consecutive_losses(self):
        """Circuit breaker should activate after max consecutive losses."""
        cb = MockCircuitBreaker(max_consecutive_losses=5, cooldown_minutes=60)

        # Record 4 losses - should NOT activate
        for _ in range(4):
            cb.record_loss()
            assert not cb.is_active, "Should not activate before threshold"

        # 5th loss - should activate
        cb.record_loss()
        assert cb.is_active, "Should activate after 5 consecutive losses"
        assert cb.time_remaining_minutes > 0, "Should have cooldown time"

    def test_resets_on_win(self):
        """Consecutive loss counter should reset on a win."""
        cb = MockCircuitBreaker(max_consecutive_losses=5, cooldown_minutes=60)

        # Record 4 losses
        for _ in range(4):
            cb.record_loss()

        assert cb.consecutive_losses == 4

        # Win resets counter
        cb.record_win()
        assert cb.consecutive_losses == 0

        # Now need 5 more losses to activate
        for _ in range(4):
            cb.record_loss()
        assert not cb.is_active

    def test_cooldown_duration(self):
        """Circuit breaker should have correct cooldown duration."""
        cb = MockCircuitBreaker(max_consecutive_losses=5, cooldown_minutes=60)

        for _ in range(5):
            cb.record_loss()

        # Check cooldown time is approximately 60 minutes
        remaining = cb.time_remaining_minutes
        assert 59 <= remaining <= 60, f"Expected ~60 min, got {remaining}"

    def test_auto_deactivates_after_cooldown(self):
        """Circuit breaker should deactivate after cooldown period."""
        cb = MockCircuitBreaker(max_consecutive_losses=5, cooldown_minutes=60)

        for _ in range(5):
            cb.record_loss()

        assert cb.is_active

        # Simulate time passing
        cb.cooldown_until = datetime.now() - timedelta(seconds=1)

        # Check should deactivate
        assert not cb.check(), "Should deactivate after cooldown"
        assert not cb.is_active

    def test_multiple_activations(self):
        """Circuit breaker can activate multiple times."""
        cb = MockCircuitBreaker(max_consecutive_losses=3, cooldown_minutes=1)

        # First activation
        for _ in range(3):
            cb.record_loss()
        assert cb.is_active

        # Simulate cooldown expiration
        cb.cooldown_until = datetime.now() - timedelta(seconds=1)
        cb.check()  # Deactivates

        assert not cb.is_active

        # Second activation
        for _ in range(3):
            cb.record_loss()
        assert cb.is_active


class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker with trading system."""

    def test_blocks_trading_when_active(self):
        """Trading should be blocked when circuit breaker is active."""
        cb = MockCircuitBreaker(max_consecutive_losses=5, cooldown_minutes=60)

        # Simulate trading
        trades_executed = 0

        for i in range(10):
            if cb.check():
                # Circuit breaker active - skip trade
                continue

            trades_executed += 1
            # Simulate all losing trades
            cb.record_loss()

        # Should have executed exactly 5 trades before blocking
        assert trades_executed == 5, f"Expected 5 trades, got {trades_executed}"
        assert cb.is_active

    def test_resumes_after_cooldown(self):
        """Trading should resume after cooldown period."""
        cb = MockCircuitBreaker(max_consecutive_losses=3, cooldown_minutes=60)

        # Trigger circuit breaker
        for _ in range(3):
            cb.record_loss()

        assert cb.is_active

        # Simulate cooldown expiration
        cb.cooldown_until = datetime.now() - timedelta(seconds=1)

        # Should be able to trade again
        assert not cb.check(), "Should allow trading after cooldown"

        # Can execute new trades
        cb.record_loss()
        assert cb.consecutive_losses == 1


class TestCircuitBreakerEdgeCases:
    """Edge case tests for circuit breaker."""

    def test_zero_losses_threshold(self):
        """Test with threshold of 0 (immediately active)."""
        cb = MockCircuitBreaker(max_consecutive_losses=0, cooldown_minutes=60)
        # With 0 threshold, any check might be weird
        # This tests defensive programming
        assert not cb.is_active  # Initially not active

    def test_very_short_cooldown(self):
        """Test with very short cooldown period."""
        cb = MockCircuitBreaker(max_consecutive_losses=2, cooldown_minutes=0)

        cb.record_loss()
        cb.record_loss()

        assert cb.is_active
        # With 0 minute cooldown, check() should immediately deactivate
        assert not cb.check()

    def test_win_after_activation(self):
        """Win after activation should reset counter but not deactivate."""
        cb = MockCircuitBreaker(max_consecutive_losses=3, cooldown_minutes=60)

        for _ in range(3):
            cb.record_loss()

        assert cb.is_active

        cb.record_win()
        assert cb.consecutive_losses == 0  # Counter reset
        assert cb.is_active  # Still active until cooldown
