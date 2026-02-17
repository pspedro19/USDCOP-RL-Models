"""
Tests for MultiDayExecutor — weekly H=5 execution with trailing stop + re-entry.
"""

import pytest
from datetime import datetime, timedelta, timezone

from src.execution.multiday_executor import (
    MultiDayConfig,
    MultiDayExecutor,
    SubtradeState,
    WeekExecutionState,
    WeekStatus,
)


@pytest.fixture
def config():
    return MultiDayConfig(
        activation_pct=0.002,   # 0.20% (v2: tight trailing)
        trail_pct=0.001,        # 0.10% (v2: micro-profit capture)
        hard_stop_pct=0.035,    # 3.50% (v2: tighter hard stop)
        cooldown_minutes=20,
    )


@pytest.fixture
def executor(config):
    return MultiDayExecutor(config)


@pytest.fixture
def base_ts():
    return datetime(2026, 2, 16, 14, 0, tzinfo=timezone.utc)  # Mon 09:00 COT


class TestEntry:
    def test_initial_entry_long(self, executor, base_ts):
        state = executor.enter("2026-02-16", 1, 1.5, 4200.0, base_ts)

        assert state.status == WeekStatus.POSITIONED
        assert state.direction == 1
        assert state.leverage == 1.5
        assert state.entry_price == 4200.0
        assert len(state.subtrades) == 1
        assert state.subtrades[0].subtrade_index == 0
        assert state.subtrades[0].trailing_state == "waiting"

    def test_initial_entry_short(self, executor, base_ts):
        state = executor.enter("2026-02-16", -1, 2.0, 4200.0, base_ts)

        assert state.status == WeekStatus.POSITIONED
        assert state.direction == -1
        assert state.leverage == 2.0

    def test_invalid_direction(self, executor, base_ts):
        with pytest.raises(AssertionError):
            executor.enter("2026-02-16", 0, 1.0, 4200.0, base_ts)

    def test_invalid_leverage(self, executor, base_ts):
        with pytest.raises(AssertionError):
            executor.enter("2026-02-16", 1, -1.0, 4200.0, base_ts)


class TestTrailingStopFires:
    def test_long_trailing_activation_and_trigger(self, executor, base_ts):
        """Trailing should fire after activation + drawback from peak."""
        state = executor.enter("2026-02-16", 1, 1.0, 4200.0, base_ts)

        # Move up 0.286% (> 0.20% activation) -> peak = 4212
        # Low stays above trail threshold: 4212*(1-0.001) = 4207.79, so low=4209 is safe
        ts1 = base_ts + timedelta(minutes=5)
        state, event = executor.update(state, 4212.0, 4209.0, 4210.0, ts1)
        assert state.status == WeekStatus.MONITORING  # Trailing activated
        assert event is None

        # Next bar: drawback from peak exceeds 0.10%
        # 4212*(1-0.001) = 4207.79, bar_low=4206 triggers trailing
        ts2 = base_ts + timedelta(minutes=10)
        state, event = executor.update(state, 4210.0, 4206.0, 4208.0, ts2)
        assert event == "trailing_exit"
        assert state.status == WeekStatus.COOLDOWN

        sub = state.subtrades[0]
        assert sub.exit_reason == "trailing_stop"
        assert sub.pnl_pct is not None

    def test_short_trailing_activation_and_trigger(self, executor, base_ts):
        """SHORT trailing: peak tracks bar_low, drawback checks bar_high."""
        state = executor.enter("2026-02-16", -1, 1.0, 4200.0, base_ts)

        # Price drops 0.238% -> peak (low) = 4190
        # bar_high stays below trail threshold: 4190*(1+0.001) = 4194.19, so high=4193 is safe
        ts1 = base_ts + timedelta(minutes=5)
        state, event = executor.update(state, 4193.0, 4190.0, 4192.0, ts1)
        assert state.status == WeekStatus.MONITORING

        # Price bounces up > 0.10% from peak -> 4190*(1+0.001) = 4194.19
        # bar_high=4196 triggers trailing
        ts2 = base_ts + timedelta(minutes=10)
        state, event = executor.update(state, 4196.0, 4191.0, 4194.0, ts2)
        assert event == "trailing_exit"
        assert state.status == WeekStatus.COOLDOWN


class TestHardStop:
    def test_long_hard_stop(self, executor, base_ts):
        """Hard stop fires at 3.5% adverse move."""
        state = executor.enter("2026-02-16", 1, 1.0, 4200.0, base_ts)

        # Price drops 3.6%: 4200 * (1 - 0.036) = 4048.8
        ts1 = base_ts + timedelta(minutes=5)
        state, event = executor.update(state, 4200.0, 4048.0, 4050.0, ts1)
        assert event == "hard_stop"
        assert state.subtrades[0].exit_reason == "hard_stop"

    def test_short_hard_stop(self, executor, base_ts):
        """SHORT hard stop: price rises 3.5%."""
        state = executor.enter("2026-02-16", -1, 1.0, 4200.0, base_ts)

        # Price rises 3.6%: 4200 * (1 + 0.036) = 4351.2
        ts1 = base_ts + timedelta(minutes=5)
        state, event = executor.update(state, 4352.0, 4200.0, 4350.0, ts1)
        assert event == "hard_stop"


class TestReEntry:
    def test_cooldown_enforced_after_trailing_exit(self, executor, base_ts):
        """After trailing exit, must wait 20 minutes before re-entry."""
        state = executor.enter("2026-02-16", 1, 1.0, 4200.0, base_ts)

        # Bar 1: Activate trailing (low stays above trail threshold)
        ts1 = base_ts + timedelta(minutes=5)
        state, _ = executor.update(state, 4212.0, 4209.0, 4210.0, ts1)
        assert state.status == WeekStatus.MONITORING

        # Bar 2: Trigger trailing
        ts2 = base_ts + timedelta(minutes=10)
        state, event = executor.update(state, 4210.0, 4206.0, 4208.0, ts2)
        assert event == "trailing_exit"
        assert state.status == WeekStatus.COOLDOWN

        # Check 10 min later — still in cooldown
        ts3 = ts2 + timedelta(minutes=10)
        state, event = executor.update(state, 4210.0, 4208.0, 4209.0, ts3)
        assert event is None
        assert state.status == WeekStatus.COOLDOWN

        # Check 20 min later — cooldown expired, ready for re-entry
        ts4 = ts2 + timedelta(minutes=20)
        state, event = executor.update(state, 4210.0, 4208.0, 4209.0, ts4)
        assert event == "re_entry_ready"

    def test_re_entry_creates_new_subtrade(self, executor, base_ts):
        """Re-entry appends a new subtrade with incremented index."""
        state = executor.enter("2026-02-16", 1, 1.0, 4200.0, base_ts)
        assert len(state.subtrades) == 1

        # Simulate trailing exit (2 bars: activate then trigger)
        ts1 = base_ts + timedelta(minutes=5)
        state, _ = executor.update(state, 4212.0, 4209.0, 4210.0, ts1)
        ts2 = base_ts + timedelta(minutes=10)
        state, _ = executor.update(state, 4210.0, 4206.0, 4208.0, ts2)

        # Re-enter after cooldown
        ts_reentry = ts2 + timedelta(minutes=20)
        state = executor.enter_subtrade(state, 4210.0, ts_reentry)
        assert len(state.subtrades) == 2
        assert state.subtrades[1].subtrade_index == 1
        assert state.subtrades[1].entry_price == 4210.0
        assert state.subtrades[1].trailing_state == "waiting"
        assert state.status == WeekStatus.POSITIONED


class TestWeekEndClose:
    def test_close_week_with_active_position(self, executor, base_ts):
        """Friday close should exit remaining position."""
        state = executor.enter("2026-02-16", 1, 1.5, 4200.0, base_ts)

        # A few bars pass, no trailing trigger
        ts1 = base_ts + timedelta(minutes=30)
        state, _ = executor.update(state, 4205.0, 4198.0, 4202.0, ts1)

        # Friday close at 12:50 COT
        close_ts = base_ts + timedelta(days=4, hours=3, minutes=50)
        state = executor.close_week(state, 4215.0, close_ts)

        assert state.status == WeekStatus.CLOSED
        assert state.exit_reason == "week_end"
        assert state.exit_price == 4215.0
        assert state.week_pnl_pct is not None

        sub = state.subtrades[0]
        assert sub.exit_reason == "week_end"
        assert sub.trailing_state == "expired"

    def test_close_week_no_active_position(self, executor, base_ts):
        """If already closed (trailing exit, no re-entry), just aggregate PnL."""
        state = executor.enter("2026-02-16", 1, 1.0, 4200.0, base_ts)

        # Trigger trailing exit (v2: act=0.20%, trail=0.10%)
        ts1 = base_ts + timedelta(minutes=5)
        state, _ = executor.update(state, 4212.0, 4209.0, 4210.0, ts1)
        ts2 = base_ts + timedelta(minutes=10)
        state, _ = executor.update(state, 4210.0, 4206.0, 4208.0, ts2)

        # Status is COOLDOWN, not POSITIONED — no active subtrade
        close_ts = base_ts + timedelta(days=4)
        state = executor.close_week(state, 4210.0, close_ts)
        assert state.status == WeekStatus.CLOSED
        assert state.week_pnl_pct is not None


class TestMultiSubtradePnL:
    def test_aggregate_pnl_from_multiple_subtrades(self, executor, base_ts):
        """Week PnL = sum of all subtrade PnLs."""
        state = executor.enter("2026-02-16", -1, 1.2, 4200.0, base_ts)

        # Subtrade 0: activate then trigger trailing exit with profit (v2 params)
        ts1 = base_ts + timedelta(minutes=5)
        state, _ = executor.update(state, 4193.0, 4190.0, 4192.0, ts1)  # Activate
        ts2 = base_ts + timedelta(minutes=10)
        state, _ = executor.update(state, 4196.0, 4191.0, 4194.0, ts2)  # Trigger

        pnl_sub0 = state.subtrades[0].pnl_pct
        assert pnl_sub0 is not None

        # Re-enter after cooldown (v2: 20 min)
        ts3 = ts2 + timedelta(minutes=20)
        state = executor.enter_subtrade(state, 4190.0, ts3)

        # Close week with subtrade 1 still open
        close_ts = ts3 + timedelta(hours=4)
        state = executor.close_week(state, 4185.0, close_ts)

        assert state.status == WeekStatus.CLOSED
        pnl_sub1 = state.subtrades[1].pnl_pct
        assert pnl_sub1 is not None
        assert abs(state.week_pnl_pct - (pnl_sub0 + pnl_sub1)) < 1e-10


class TestCircuitBreaker:
    def test_circuit_breaker_closes_position(self, executor, base_ts):
        """Circuit breaker should close active subtrade and set PAUSED."""
        state = executor.enter("2026-02-16", 1, 1.0, 4200.0, base_ts)

        close_ts = base_ts + timedelta(hours=1)
        state = executor.close_circuit_breaker(state, 4100.0, close_ts)

        assert state.status == WeekStatus.PAUSED
        assert state.exit_reason == "circuit_breaker"
        assert state.subtrades[0].exit_reason == "circuit_breaker"
        assert state.week_pnl_pct is not None


class TestShouldMonitor:
    def test_positioned_needs_monitoring(self, executor, base_ts):
        state = executor.enter("2026-02-16", 1, 1.0, 4200.0, base_ts)
        assert executor.should_monitor(state) is True

    def test_closed_does_not_need_monitoring(self, executor, base_ts):
        state = executor.enter("2026-02-16", 1, 1.0, 4200.0, base_ts)
        state = executor.close_week(state, 4200.0, base_ts + timedelta(days=4))
        assert executor.should_monitor(state) is False

    def test_cooldown_needs_monitoring(self, executor, base_ts):
        state = executor.enter("2026-02-16", 1, 1.0, 4200.0, base_ts)
        # Force cooldown status
        state.status = WeekStatus.COOLDOWN
        assert executor.should_monitor(state) is True


class TestShouldCloseWeek:
    def test_before_week_end(self):
        bar_ts = datetime(2026, 2, 20, 17, 0, tzinfo=timezone.utc)  # Fri 12:00 COT
        end_ts = datetime(2026, 2, 20, 17, 50, tzinfo=timezone.utc)  # Fri 12:50 COT
        assert MultiDayExecutor.should_close_week(bar_ts, end_ts) is False

    def test_at_week_end(self):
        end_ts = datetime(2026, 2, 20, 17, 50, tzinfo=timezone.utc)
        assert MultiDayExecutor.should_close_week(end_ts, end_ts) is True

    def test_after_week_end(self):
        bar_ts = datetime(2026, 2, 20, 18, 0, tzinfo=timezone.utc)
        end_ts = datetime(2026, 2, 20, 17, 50, tzinfo=timezone.utc)
        assert MultiDayExecutor.should_close_week(bar_ts, end_ts) is True


class TestPnLCalculation:
    def test_long_positive_pnl(self, executor, base_ts):
        """LONG: entry 4200, exit 4242 -> +1% unleveraged."""
        state = executor.enter("2026-02-16", 1, 2.0, 4200.0, base_ts)
        state = executor.close_week(state, 4242.0, base_ts + timedelta(days=4))

        sub = state.subtrades[0]
        assert abs(sub.pnl_unleveraged_pct - 0.01) < 1e-6  # +1%
        assert abs(sub.pnl_pct - 0.02) < 1e-6              # +2% leveraged

    def test_short_positive_pnl(self, executor, base_ts):
        """SHORT: entry 4200, exit 4158 -> +1% unleveraged."""
        state = executor.enter("2026-02-16", -1, 1.5, 4200.0, base_ts)
        state = executor.close_week(state, 4158.0, base_ts + timedelta(days=4))

        sub = state.subtrades[0]
        assert abs(sub.pnl_unleveraged_pct - 0.01) < 1e-6  # +1%
        assert abs(sub.pnl_pct - 0.015) < 1e-6             # +1.5% leveraged

    def test_long_negative_pnl(self, executor, base_ts):
        """LONG: entry 4200, exit 4158 -> -1%."""
        state = executor.enter("2026-02-16", 1, 1.0, 4200.0, base_ts)
        state = executor.close_week(state, 4158.0, base_ts + timedelta(days=4))

        sub = state.subtrades[0]
        assert abs(sub.pnl_unleveraged_pct - (-0.01)) < 1e-6
