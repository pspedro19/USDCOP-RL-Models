"""
Unit Tests for Command Pattern - Risk Management Commands
==========================================================

Tests for the Command Pattern implementation in src/risk/commands.py

Author: USD/COP Trading System
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.risk.commands import (
    Command,
    CommandResult,
    CommandInvoker,
    TriggerCircuitBreakerCommand,
    SetCooldownCommand,
    ClearCooldownCommand,
    ResetKillSwitchCommand,
    UpdateRiskLimitsCommand,
    BlockTradingCommand,
)
from src.risk.risk_manager import RiskManager, RiskLimits


@pytest.fixture
def risk_manager():
    """Create a fresh RiskManager for each test."""
    return RiskManager(RiskLimits(
        max_drawdown_pct=15.0,
        max_daily_loss_pct=5.0,
        max_trades_per_day=20,
        cooldown_after_losses=5,
        cooldown_minutes=60
    ))


@pytest.fixture
def invoker():
    """Create a fresh CommandInvoker for each test."""
    return CommandInvoker()


class TestCommandResult:
    """Tests for CommandResult dataclass."""

    def test_create_result(self):
        """Test creating a command result."""
        result = CommandResult(
            success=True,
            message="Operation completed"
        )

        assert result.success is True
        assert result.message == "Operation completed"
        assert isinstance(result.timestamp, datetime)

    def test_result_with_data(self):
        """Test command result with additional data."""
        result = CommandResult(
            success=True,
            message="Updated",
            data={"old_value": 10, "new_value": 20}
        )

        assert result.data["old_value"] == 10
        assert result.data["new_value"] == 20


class TestTriggerCircuitBreakerCommand:
    """Tests for TriggerCircuitBreakerCommand."""

    def test_execute_triggers_cooldown(self, risk_manager):
        """Test that execute triggers cooldown."""
        cmd = TriggerCircuitBreakerCommand(
            risk_manager=risk_manager,
            reason="5 consecutive losses"
        )

        result = cmd.execute()

        assert result.success is True
        assert risk_manager._cooldown_until is not None
        assert risk_manager._cooldown_until > datetime.now()

    def test_custom_cooldown_duration(self, risk_manager):
        """Test custom cooldown duration."""
        cmd = TriggerCircuitBreakerCommand(
            risk_manager=risk_manager,
            reason="Manual trigger",
            cooldown_minutes=30
        )

        result = cmd.execute()

        assert result.success is True
        expected_end = datetime.now() + timedelta(minutes=30)
        assert abs((risk_manager._cooldown_until - expected_end).total_seconds()) < 2

    def test_undo_restores_state(self, risk_manager):
        """Test that undo restores previous state."""
        # Set initial cooldown
        original_cooldown = datetime.now() + timedelta(minutes=10)
        risk_manager._cooldown_until = original_cooldown

        cmd = TriggerCircuitBreakerCommand(
            risk_manager=risk_manager,
            reason="Test"
        )

        cmd.execute()
        result = cmd.undo()

        assert result.success is True
        assert risk_manager._cooldown_until == original_cooldown

    def test_cannot_undo_unexecuted_command(self, risk_manager):
        """Test that unexecuted commands cannot be undone."""
        cmd = TriggerCircuitBreakerCommand(
            risk_manager=risk_manager,
            reason="Test"
        )

        result = cmd.undo()

        assert result.success is False
        assert "has not been executed" in result.message

    def test_get_description(self, risk_manager):
        """Test command description."""
        cmd = TriggerCircuitBreakerCommand(
            risk_manager=risk_manager,
            reason="5 consecutive losses"
        )

        desc = cmd.get_description()

        assert "circuit breaker" in desc.lower()
        assert "5 consecutive losses" in desc


class TestSetCooldownCommand:
    """Tests for SetCooldownCommand."""

    def test_execute_sets_cooldown(self, risk_manager):
        """Test that execute sets cooldown."""
        cmd = SetCooldownCommand(
            risk_manager=risk_manager,
            cooldown_minutes=30,
            reason="Manual cooldown"
        )

        result = cmd.execute()

        assert result.success is True
        assert risk_manager._cooldown_until is not None

    def test_undo_restores_cooldown(self, risk_manager):
        """Test that undo restores previous cooldown."""
        risk_manager._cooldown_until = None

        cmd = SetCooldownCommand(
            risk_manager=risk_manager,
            cooldown_minutes=30
        )

        cmd.execute()
        result = cmd.undo()

        assert result.success is True
        assert risk_manager._cooldown_until is None

    def test_result_includes_data(self, risk_manager):
        """Test that result includes execution data."""
        cmd = SetCooldownCommand(
            risk_manager=risk_manager,
            cooldown_minutes=45,
            reason="Test"
        )

        result = cmd.execute()

        assert result.data["duration_minutes"] == 45
        assert result.data["reason"] == "Test"


class TestClearCooldownCommand:
    """Tests for ClearCooldownCommand."""

    def test_execute_clears_cooldown(self, risk_manager):
        """Test that execute clears active cooldown."""
        risk_manager._cooldown_until = datetime.now() + timedelta(minutes=30)
        risk_manager._consecutive_losses = 3

        cmd = ClearCooldownCommand(
            risk_manager=risk_manager,
            reason="Manual clear"
        )

        result = cmd.execute()

        assert result.success is True
        assert risk_manager._cooldown_until is None
        assert risk_manager._consecutive_losses == 0

    def test_fails_when_no_cooldown(self, risk_manager):
        """Test that clearing fails when no cooldown active."""
        risk_manager._cooldown_until = None

        cmd = ClearCooldownCommand(
            risk_manager=risk_manager,
            reason="Test"
        )

        result = cmd.execute()

        assert result.success is False
        assert "no active cooldown" in result.message.lower()

    def test_undo_restores_cooldown(self, risk_manager):
        """Test that undo restores cleared cooldown."""
        original = datetime.now() + timedelta(minutes=30)
        risk_manager._cooldown_until = original
        risk_manager._consecutive_losses = 4

        cmd = ClearCooldownCommand(
            risk_manager=risk_manager,
            reason="Test"
        )

        cmd.execute()
        result = cmd.undo()

        assert result.success is True
        assert risk_manager._cooldown_until == original
        assert risk_manager._consecutive_losses == 4


class TestResetKillSwitchCommand:
    """Tests for ResetKillSwitchCommand."""

    def test_requires_confirmation(self, risk_manager):
        """Test that reset requires explicit confirmation."""
        risk_manager._kill_switch_active = True

        cmd = ResetKillSwitchCommand(
            risk_manager=risk_manager,
            confirmed=False
        )

        result = cmd.execute()

        assert result.success is False
        assert "confirmation" in result.message.lower()
        assert risk_manager._kill_switch_active is True

    def test_execute_with_confirmation(self, risk_manager):
        """Test that reset works with confirmation."""
        risk_manager._kill_switch_active = True

        cmd = ResetKillSwitchCommand(
            risk_manager=risk_manager,
            confirmed=True,
            reason="Reviewed and approved"
        )

        result = cmd.execute()

        assert result.success is True
        assert risk_manager._kill_switch_active is False

    def test_fails_when_not_active(self, risk_manager):
        """Test that reset fails when kill switch is not active."""
        risk_manager._kill_switch_active = False

        cmd = ResetKillSwitchCommand(
            risk_manager=risk_manager,
            confirmed=True
        )

        result = cmd.execute()

        assert result.success is False
        assert "not active" in result.message.lower()

    def test_undo_reactivates_kill_switch(self, risk_manager):
        """Test that undo reactivates kill switch."""
        risk_manager._kill_switch_active = True

        cmd = ResetKillSwitchCommand(
            risk_manager=risk_manager,
            confirmed=True
        )

        cmd.execute()
        result = cmd.undo()

        assert result.success is True
        assert risk_manager._kill_switch_active is True


class TestUpdateRiskLimitsCommand:
    """Tests for UpdateRiskLimitsCommand."""

    def test_execute_updates_limits(self, risk_manager):
        """Test that execute updates risk limits."""
        cmd = UpdateRiskLimitsCommand(
            risk_manager=risk_manager,
            new_limits={"max_drawdown_pct": 20.0, "max_trades_per_day": 30},
            reason="Adjusted for volatility"
        )

        result = cmd.execute()

        assert result.success is True
        assert risk_manager.limits.max_drawdown_pct == 20.0
        assert risk_manager.limits.max_trades_per_day == 30

    def test_undo_restores_limits(self, risk_manager):
        """Test that undo restores previous limits."""
        original_drawdown = risk_manager.limits.max_drawdown_pct

        cmd = UpdateRiskLimitsCommand(
            risk_manager=risk_manager,
            new_limits={"max_drawdown_pct": 25.0}
        )

        cmd.execute()
        result = cmd.undo()

        assert result.success is True
        assert risk_manager.limits.max_drawdown_pct == original_drawdown

    def test_ignores_unknown_limits(self, risk_manager):
        """Test that unknown limit names are ignored."""
        cmd = UpdateRiskLimitsCommand(
            risk_manager=risk_manager,
            new_limits={"unknown_limit": 100}
        )

        result = cmd.execute()

        assert result.success is False
        assert "no valid limits" in result.message.lower()

    def test_result_includes_changes(self, risk_manager):
        """Test that result includes change details."""
        original = risk_manager.limits.max_drawdown_pct

        cmd = UpdateRiskLimitsCommand(
            risk_manager=risk_manager,
            new_limits={"max_drawdown_pct": 25.0}
        )

        result = cmd.execute()

        assert result.data["updates"]["max_drawdown_pct"]["old"] == original
        assert result.data["updates"]["max_drawdown_pct"]["new"] == 25.0


class TestBlockTradingCommand:
    """Tests for BlockTradingCommand."""

    def test_execute_blocks_trading(self, risk_manager):
        """Test that execute blocks trading."""
        cmd = BlockTradingCommand(
            risk_manager=risk_manager,
            reason="End of day"
        )

        result = cmd.execute()

        assert result.success is True
        assert risk_manager._daily_blocked is True

    def test_undo_unblocks_trading(self, risk_manager):
        """Test that undo unblocks trading."""
        risk_manager._daily_blocked = False

        cmd = BlockTradingCommand(
            risk_manager=risk_manager,
            reason="Test"
        )

        cmd.execute()
        result = cmd.undo()

        assert result.success is True
        assert risk_manager._daily_blocked is False


class TestCommandInvoker:
    """Tests for CommandInvoker."""

    def test_execute_command(self, invoker, risk_manager):
        """Test executing a command through invoker."""
        cmd = SetCooldownCommand(
            risk_manager=risk_manager,
            cooldown_minutes=30
        )

        result = invoker.execute(cmd)

        assert result.success is True
        assert len(invoker.get_history()) == 1

    def test_undo_command(self, invoker, risk_manager):
        """Test undoing a command through invoker."""
        cmd = SetCooldownCommand(
            risk_manager=risk_manager,
            cooldown_minutes=30
        )

        invoker.execute(cmd)
        result = invoker.undo()

        assert result.success is True
        assert risk_manager._cooldown_until is None

    def test_redo_command(self, invoker, risk_manager):
        """Test redoing an undone command."""
        cmd = SetCooldownCommand(
            risk_manager=risk_manager,
            cooldown_minutes=30
        )

        invoker.execute(cmd)
        invoker.undo()
        result = invoker.redo()

        assert result.success is True
        assert risk_manager._cooldown_until is not None

    def test_undo_clears_redo_stack(self, invoker, risk_manager):
        """Test that new command clears redo stack."""
        cmd1 = SetCooldownCommand(
            risk_manager=risk_manager,
            cooldown_minutes=30
        )
        cmd2 = SetCooldownCommand(
            risk_manager=risk_manager,
            cooldown_minutes=60
        )

        invoker.execute(cmd1)
        invoker.undo()

        assert invoker.can_redo() is True

        invoker.execute(cmd2)

        assert invoker.can_redo() is False

    def test_cannot_undo_empty_history(self, invoker):
        """Test undo fails with empty history."""
        result = invoker.undo()

        assert result.success is False
        assert "no commands to undo" in result.message.lower()

    def test_cannot_redo_empty_stack(self, invoker):
        """Test redo fails with empty redo stack."""
        result = invoker.redo()

        assert result.success is False
        assert "no commands to redo" in result.message.lower()

    def test_can_undo_can_redo(self, invoker, risk_manager):
        """Test can_undo and can_redo methods."""
        assert invoker.can_undo() is False
        assert invoker.can_redo() is False

        cmd = SetCooldownCommand(
            risk_manager=risk_manager,
            cooldown_minutes=30
        )

        invoker.execute(cmd)
        assert invoker.can_undo() is True
        assert invoker.can_redo() is False

        invoker.undo()
        assert invoker.can_undo() is False
        assert invoker.can_redo() is True

    def test_history_limit(self, risk_manager):
        """Test history respects max limit."""
        invoker = CommandInvoker(max_history=5)

        for i in range(10):
            cmd = SetCooldownCommand(
                risk_manager=risk_manager,
                cooldown_minutes=i + 1
            )
            invoker.execute(cmd)

        history = invoker.get_history(limit=10)
        assert len(history) == 5

    def test_get_history(self, invoker, risk_manager):
        """Test getting command history."""
        cmd1 = SetCooldownCommand(risk_manager=risk_manager, cooldown_minutes=30)
        cmd2 = BlockTradingCommand(risk_manager=risk_manager, reason="Test")

        invoker.execute(cmd1)
        invoker.execute(cmd2)

        history = invoker.get_history()

        assert len(history) == 2
        assert history[0]["success"] is True  # Most recent first
        assert "description" in history[0]

    def test_clear_history(self, invoker, risk_manager):
        """Test clearing command history."""
        cmd = SetCooldownCommand(risk_manager=risk_manager, cooldown_minutes=30)
        invoker.execute(cmd)
        invoker.undo()

        invoker.clear_history()

        assert invoker.can_undo() is False
        assert invoker.can_redo() is False
        assert len(invoker.get_history()) == 0

    def test_execute_batch(self, invoker, risk_manager):
        """Test batch command execution."""
        commands = [
            SetCooldownCommand(risk_manager=risk_manager, cooldown_minutes=30),
            UpdateRiskLimitsCommand(
                risk_manager=risk_manager,
                new_limits={"max_drawdown_pct": 20.0}
            ),
        ]

        results = invoker.execute_batch(commands)

        assert len(results) == 2
        assert all(r.success for r in results)

    def test_batch_stops_on_failure(self, risk_manager):
        """Test batch execution stops on failure when configured."""
        risk_manager._cooldown_until = None  # No cooldown to clear

        invoker = CommandInvoker()
        commands = [
            SetCooldownCommand(risk_manager=risk_manager, cooldown_minutes=30),
            ClearCooldownCommand(risk_manager=risk_manager, reason="Test"),  # Will work
            ClearCooldownCommand(risk_manager=risk_manager, reason="Test"),  # Will fail
        ]

        results = invoker.execute_batch(commands, stop_on_failure=True)

        # Should stop after the failing command
        assert len(results) == 3
        assert results[0].success is True
        assert results[1].success is True
        assert results[2].success is False


class TestCommandExecutionTracking:
    """Tests for command execution tracking."""

    def test_executed_flag(self, risk_manager):
        """Test executed flag is set after execution."""
        cmd = SetCooldownCommand(risk_manager=risk_manager, cooldown_minutes=30)

        assert cmd.executed is False

        cmd.execute()

        assert cmd.executed is True

    def test_executed_at_timestamp(self, risk_manager):
        """Test execution timestamp is recorded."""
        cmd = SetCooldownCommand(risk_manager=risk_manager, cooldown_minutes=30)

        assert cmd.executed_at is None

        before = datetime.now()
        cmd.execute()
        after = datetime.now()

        assert cmd.executed_at is not None
        assert before <= cmd.executed_at <= after

    def test_result_property(self, risk_manager):
        """Test result property stores execution result."""
        cmd = SetCooldownCommand(risk_manager=risk_manager, cooldown_minutes=30)

        assert cmd.result is None

        cmd.execute()

        assert cmd.result is not None
        assert cmd.result.success is True

    def test_can_undo_method(self, risk_manager):
        """Test can_undo method."""
        cmd = SetCooldownCommand(risk_manager=risk_manager, cooldown_minutes=30)

        assert cmd.can_undo() is False

        cmd.execute()

        assert cmd.can_undo() is True

        cmd.undo()

        assert cmd.can_undo() is False
