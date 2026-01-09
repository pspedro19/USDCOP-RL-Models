"""
Unit Tests for RiskManager
==========================

Tests for the safety layer that validates trading signals
and prevents catastrophic losses.

Run with: pytest src/tests/test_risk_manager.py -v
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.risk.risk_manager import RiskManager, RiskLimits, TradeRecord


class TestRiskManagerBasics:
    """Basic functionality tests for RiskManager."""

    def test_initialization_default_limits(self):
        """Test RiskManager initializes with default limits."""
        rm = RiskManager()

        assert rm.limits.max_drawdown_pct == 15.0
        assert rm.limits.max_daily_loss_pct == 5.0
        assert rm.limits.max_trades_per_day == 20
        assert rm.limits.cooldown_after_losses == 3
        assert rm.limits.cooldown_minutes == 30

    def test_initialization_custom_limits(self):
        """Test RiskManager initializes with custom limits."""
        limits = RiskLimits(
            max_drawdown_pct=10.0,
            max_daily_loss_pct=3.0,
            max_trades_per_day=15,
            cooldown_after_losses=2,
            cooldown_minutes=15
        )
        rm = RiskManager(limits)

        assert rm.limits.max_drawdown_pct == 10.0
        assert rm.limits.max_daily_loss_pct == 3.0
        assert rm.limits.max_trades_per_day == 15
        assert rm.limits.cooldown_after_losses == 2
        assert rm.limits.cooldown_minutes == 15

    def test_initial_status(self):
        """Test initial status is clean/unblocked."""
        rm = RiskManager()
        status = rm.get_status()

        assert status['kill_switch_active'] is False
        assert status['daily_blocked'] is False
        assert status['cooldown_active'] is False
        assert status['trade_count_today'] == 0
        assert status['daily_pnl_pct'] == 0.0
        assert status['consecutive_losses'] == 0


class TestNormalTrades:
    """Tests for normal trading conditions."""

    def test_allows_normal_trade(self):
        """Test that normal trades are allowed when no limits breached."""
        rm = RiskManager()

        allowed, reason = rm.validate_signal("long", current_drawdown_pct=5.0)

        assert allowed is True
        assert "allowed" in reason.lower()

    def test_allows_trade_with_zero_drawdown(self):
        """Test trades allowed at zero drawdown."""
        rm = RiskManager()

        allowed, reason = rm.validate_signal("short", current_drawdown_pct=0.0)

        assert allowed is True

    def test_allows_multiple_trades(self):
        """Test multiple trades allowed within limits."""
        rm = RiskManager()

        for i in range(10):
            allowed, reason = rm.validate_signal("long", current_drawdown_pct=3.0)
            assert allowed is True
            rm.record_trade_result(0.1)  # Small profit

    def test_exit_signals_always_allowed(self):
        """Test that close/flat signals are always allowed."""
        rm = RiskManager()

        # Trigger kill switch
        allowed, _ = rm.validate_signal("long", current_drawdown_pct=20.0)
        assert allowed is False  # Kill switch triggered

        # Close signal should still be allowed
        allowed, reason = rm.validate_signal("close", current_drawdown_pct=20.0)
        assert allowed is True
        assert "exit" in reason.lower() or "risk reduction" in reason.lower()

        # Flat signal should also be allowed
        allowed, reason = rm.validate_signal("flat", current_drawdown_pct=20.0)
        assert allowed is True


class TestKillSwitch:
    """Tests for kill switch functionality."""

    def test_kills_on_max_drawdown(self):
        """Test kill switch triggers when drawdown exceeds limit."""
        limits = RiskLimits(max_drawdown_pct=10.0)
        rm = RiskManager(limits)

        # First trade at 9% drawdown - should be allowed
        allowed, _ = rm.validate_signal("long", current_drawdown_pct=9.0)
        assert allowed is True

        # Trade at exactly 10% drawdown - should trigger kill switch
        allowed, reason = rm.validate_signal("long", current_drawdown_pct=10.0)

        assert allowed is False
        assert "kill switch" in reason.lower()

    def test_kill_switch_persists(self):
        """Test kill switch remains active even with good conditions."""
        rm = RiskManager(RiskLimits(max_drawdown_pct=10.0))

        # Trigger kill switch
        rm.validate_signal("long", current_drawdown_pct=12.0)

        # Try with lower drawdown - should still be blocked
        allowed, reason = rm.validate_signal("long", current_drawdown_pct=2.0)

        assert allowed is False
        assert "kill switch" in reason.lower()

    def test_kill_switch_reset_requires_confirmation(self):
        """Test kill switch reset requires explicit confirmation."""
        rm = RiskManager(RiskLimits(max_drawdown_pct=10.0))

        # Trigger kill switch
        rm.validate_signal("long", current_drawdown_pct=12.0)

        # Try reset without confirmation
        result = rm.reset_kill_switch(confirm=False)
        assert result is False
        assert rm._kill_switch_active is True

        # Reset with confirmation
        result = rm.reset_kill_switch(confirm=True)
        assert result is True
        assert rm._kill_switch_active is False

    def test_kill_switch_not_reset_by_daily_reset(self):
        """Test that daily reset does not clear kill switch."""
        rm = RiskManager(RiskLimits(max_drawdown_pct=10.0))

        # Trigger kill switch
        rm.validate_signal("long", current_drawdown_pct=12.0)
        assert rm._kill_switch_active is True

        # Daily reset
        rm.reset_daily()

        # Kill switch should still be active
        assert rm._kill_switch_active is True


class TestDailyLossLimit:
    """Tests for daily loss limit functionality."""

    def test_blocks_after_daily_loss(self):
        """Test trading blocked after daily loss limit reached."""
        limits = RiskLimits(max_daily_loss_pct=5.0)
        rm = RiskManager(limits)

        # Record losses totaling more than 5%
        rm.record_trade_result(-2.0)
        rm.record_trade_result(-2.0)
        rm.record_trade_result(-1.5)  # Total: -5.5%

        # Should be blocked
        allowed, reason = rm.validate_signal("long", current_drawdown_pct=3.0)

        assert allowed is False
        assert "daily" in reason.lower()

    def test_daily_loss_accumulates(self):
        """Test daily losses accumulate correctly."""
        rm = RiskManager(RiskLimits(max_daily_loss_pct=5.0))

        rm.record_trade_result(-1.0)
        rm.record_trade_result(-1.0)
        rm.record_trade_result(-1.0)

        status = rm.get_status()
        assert status['daily_pnl_pct'] == -3.0

        # Not blocked yet
        allowed, _ = rm.validate_signal("long", current_drawdown_pct=2.0)
        assert allowed is True

        # Add more loss to breach limit
        rm.record_trade_result(-2.5)

        allowed, _ = rm.validate_signal("long", current_drawdown_pct=2.0)
        assert allowed is False

    def test_daily_block_cleared_on_reset(self):
        """Test daily block is cleared on daily reset."""
        rm = RiskManager(RiskLimits(max_daily_loss_pct=3.0))

        # Trigger daily block
        rm.record_trade_result(-4.0)
        assert rm._daily_blocked is True

        # Reset
        rm.reset_daily()

        assert rm._daily_blocked is False
        allowed, _ = rm.validate_signal("long", current_drawdown_pct=2.0)
        assert allowed is True


class TestCooldown:
    """Tests for cooldown functionality after consecutive losses."""

    def test_activates_cooldown_after_losses(self):
        """Test cooldown activates after consecutive losses."""
        limits = RiskLimits(
            cooldown_after_losses=3,
            cooldown_minutes=30,
            max_daily_loss_pct=50.0  # High to avoid daily block
        )
        rm = RiskManager(limits)

        # Record consecutive losses
        rm.record_trade_result(-0.5)
        rm.record_trade_result(-0.5)
        rm.record_trade_result(-0.5)  # Third loss triggers cooldown

        # Should be in cooldown
        status = rm.get_status()
        assert status['cooldown_active'] is True
        assert status['cooldown_remaining_minutes'] > 0

        # Trade should be blocked
        allowed, reason = rm.validate_signal("long", current_drawdown_pct=2.0)
        assert allowed is False
        assert "cooldown" in reason.lower()

    def test_cooldown_reset_on_win(self):
        """Test consecutive loss counter resets on winning trade."""
        limits = RiskLimits(
            cooldown_after_losses=3,
            max_daily_loss_pct=50.0
        )
        rm = RiskManager(limits)

        # Two losses
        rm.record_trade_result(-0.5)
        rm.record_trade_result(-0.5)
        assert rm._consecutive_losses == 2

        # One win
        rm.record_trade_result(0.5)
        assert rm._consecutive_losses == 0

        # Two more losses - should not trigger cooldown yet
        rm.record_trade_result(-0.5)
        rm.record_trade_result(-0.5)
        assert rm._consecutive_losses == 2
        assert rm._cooldown_until is None

    def test_cooldown_expires(self):
        """Test cooldown expires after specified time."""
        limits = RiskLimits(
            cooldown_after_losses=2,
            cooldown_minutes=1,  # 1 minute for faster testing
            max_daily_loss_pct=50.0
        )
        rm = RiskManager(limits)

        # Trigger cooldown
        rm.record_trade_result(-0.5)
        rm.record_trade_result(-0.5)

        # Verify cooldown is active
        assert rm._cooldown_until is not None

        # Manually expire cooldown
        rm._cooldown_until = datetime.now() - timedelta(seconds=1)

        # Should be allowed now
        allowed, _ = rm.validate_signal("long", current_drawdown_pct=2.0)
        assert allowed is True
        assert rm._cooldown_until is None


class TestTradeLimit:
    """Tests for daily trade limit functionality."""

    def test_max_trades_per_day(self):
        """Test trading blocked after max daily trades reached."""
        limits = RiskLimits(max_trades_per_day=5)
        rm = RiskManager(limits)

        # Execute max trades
        for i in range(5):
            allowed, _ = rm.validate_signal("long", current_drawdown_pct=2.0)
            assert allowed is True
            rm.record_trade_result(0.1)

        # 6th trade should be blocked
        allowed, reason = rm.validate_signal("long", current_drawdown_pct=2.0)
        assert allowed is False
        assert "trade limit" in reason.lower() or "5" in reason

    def test_trade_count_tracking(self):
        """Test trade count is tracked correctly."""
        rm = RiskManager(RiskLimits(max_trades_per_day=10))

        rm.record_trade_result(0.1)
        rm.record_trade_result(-0.1)
        rm.record_trade_result(0.2)

        status = rm.get_status()
        assert status['trade_count_today'] == 3
        assert status['trades_remaining'] == 7

    def test_trade_limit_reset_on_daily(self):
        """Test trade count resets on daily reset."""
        rm = RiskManager(RiskLimits(max_trades_per_day=3))

        # Use up all trades
        for _ in range(3):
            rm.record_trade_result(0.1)

        assert rm._trade_count_today == 3

        # Reset
        rm.reset_daily()

        assert rm._trade_count_today == 0
        allowed, _ = rm.validate_signal("long", current_drawdown_pct=2.0)
        assert allowed is True


class TestDailyReset:
    """Tests for daily reset functionality."""

    def test_daily_reset(self):
        """Test daily reset clears appropriate counters."""
        rm = RiskManager()

        # Record some activity
        rm.record_trade_result(-1.0)
        rm.record_trade_result(-1.0)
        rm.record_trade_result(-1.0)

        assert rm._trade_count_today == 3
        assert rm._daily_pnl_pct == -3.0
        assert rm._consecutive_losses == 3

        # Reset
        rm.reset_daily()

        assert rm._trade_count_today == 0
        assert rm._daily_pnl_pct == 0.0
        assert rm._consecutive_losses == 0
        assert rm._daily_blocked is False
        assert rm._cooldown_until is None

    def test_automatic_daily_reset(self):
        """Test automatic reset on day change."""
        rm = RiskManager()

        # Record some activity
        rm.record_trade_result(-1.0)
        rm._current_day = datetime.now().date() - timedelta(days=1)

        # Next validation should trigger auto-reset
        allowed, _ = rm.validate_signal("long", current_drawdown_pct=2.0)

        assert rm._current_day == datetime.now().date()
        assert rm._trade_count_today == 0
        assert rm._daily_pnl_pct == 0.0


class TestStatusReporting:
    """Tests for status reporting functionality."""

    def test_get_status_structure(self):
        """Test get_status returns correct structure."""
        rm = RiskManager()
        status = rm.get_status()

        # Required fields
        assert 'kill_switch_active' in status
        assert 'daily_blocked' in status
        assert 'cooldown_active' in status
        assert 'cooldown_remaining_minutes' in status
        assert 'trade_count_today' in status
        assert 'trades_remaining' in status
        assert 'daily_pnl_pct' in status
        assert 'consecutive_losses' in status
        assert 'daily_loss_remaining_pct' in status
        assert 'limits' in status
        assert 'current_day' in status
        assert 'last_updated' in status

    def test_get_status_limits(self):
        """Test get_status includes correct limits."""
        limits = RiskLimits(
            max_drawdown_pct=12.0,
            max_daily_loss_pct=4.0,
            max_trades_per_day=25,
            cooldown_after_losses=4,
            cooldown_minutes=45
        )
        rm = RiskManager(limits)
        status = rm.get_status()

        assert status['limits']['max_drawdown_pct'] == 12.0
        assert status['limits']['max_daily_loss_pct'] == 4.0
        assert status['limits']['max_trades_per_day'] == 25
        assert status['limits']['cooldown_after_losses'] == 4
        assert status['limits']['cooldown_minutes'] == 45

    def test_get_trade_history(self):
        """Test trade history retrieval."""
        rm = RiskManager()

        rm.record_trade_result(0.5, "long")
        rm.record_trade_result(-0.3, "short")
        rm.record_trade_result(0.2, "long")

        history = rm.get_trade_history()

        assert len(history) == 3
        assert history[0]['pnl_pct'] == 0.5
        assert history[0]['signal'] == "long"
        assert history[1]['pnl_pct'] == -0.3
        assert history[2]['pnl_pct'] == 0.2

    def test_trade_history_limit(self):
        """Test trade history respects limit parameter."""
        rm = RiskManager()

        for i in range(10):
            rm.record_trade_result(0.1 * i)

        history = rm.get_trade_history(limit=5)
        assert len(history) == 5


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_exactly_at_drawdown_limit(self):
        """Test behavior at exactly the drawdown limit."""
        rm = RiskManager(RiskLimits(max_drawdown_pct=10.0))

        allowed, _ = rm.validate_signal("long", current_drawdown_pct=10.0)
        assert allowed is False  # Should trigger at exactly the limit

    def test_exactly_at_daily_loss_limit(self):
        """Test behavior at exactly the daily loss limit."""
        rm = RiskManager(RiskLimits(max_daily_loss_pct=5.0))

        rm.record_trade_result(-5.0)  # Exactly at limit

        allowed, _ = rm.validate_signal("long", current_drawdown_pct=2.0)
        assert allowed is False

    def test_negative_pnl_precision(self):
        """Test handling of precise negative P&L values."""
        rm = RiskManager(RiskLimits(max_daily_loss_pct=5.0))

        rm.record_trade_result(-1.111)
        rm.record_trade_result(-2.222)
        rm.record_trade_result(-1.668)  # Total: -5.001

        allowed, _ = rm.validate_signal("long", current_drawdown_pct=2.0)
        assert allowed is False

    def test_zero_pnl_trade(self):
        """Test handling of break-even trades."""
        rm = RiskManager()

        rm.record_trade_result(0.0)

        assert rm._consecutive_losses == 0
        assert rm._trade_count_today == 1

    def test_very_small_profit(self):
        """Test handling of very small profits."""
        rm = RiskManager()

        rm.record_trade_result(-0.1)
        rm.record_trade_result(-0.1)
        rm.record_trade_result(0.001)  # Tiny profit resets consecutive losses

        assert rm._consecutive_losses == 0


class TestMultipleConditions:
    """Tests for multiple risk conditions occurring simultaneously."""

    def test_kill_switch_takes_priority(self):
        """Test kill switch blocks even if other conditions would allow."""
        rm = RiskManager(RiskLimits(
            max_drawdown_pct=10.0,
            max_daily_loss_pct=20.0  # Won't be hit
        ))

        # Trigger kill switch
        rm.validate_signal("long", current_drawdown_pct=12.0)

        # Even with zero drawdown now, should be blocked
        allowed, reason = rm.validate_signal("long", current_drawdown_pct=0.0)

        assert allowed is False
        assert "kill switch" in reason.lower()

    def test_all_conditions_interact_correctly(self):
        """Test complex scenario with multiple conditions."""
        limits = RiskLimits(
            max_drawdown_pct=15.0,
            max_daily_loss_pct=5.0,
            max_trades_per_day=10,
            cooldown_after_losses=3,
            cooldown_minutes=30
        )
        rm = RiskManager(limits)

        # Record 3 losses to trigger cooldown
        rm.record_trade_result(-1.0)
        rm.record_trade_result(-1.0)
        rm.record_trade_result(-1.0)

        # Should be in cooldown (not daily blocked yet, only -3%)
        status = rm.get_status()
        assert status['cooldown_active'] is True
        assert status['daily_blocked'] is False

        # Manually expire cooldown
        rm._cooldown_until = datetime.now() - timedelta(seconds=1)
        rm._consecutive_losses = 0

        # Record more losses to trigger daily block
        rm.record_trade_result(-3.0)  # Now at -6%

        status = rm.get_status()
        assert status['daily_blocked'] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
