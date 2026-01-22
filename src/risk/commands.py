"""
Risk Management Command Pattern - Encapsulated Operations with Undo Support
============================================================================

This module implements the Command Pattern for risk management operations.
Commands provide:
- Encapsulation of operations as objects
- Undo/redo support for risk actions
- Command history for audit trails
- Decoupling of invoker from executor

Available Commands:
- TriggerCircuitBreakerCommand: Activate circuit breaker
- SetCooldownCommand: Set cooldown period
- ResetKillSwitchCommand: Reset kill switch (with confirmation)
- UpdateRiskLimitsCommand: Modify risk limits

Usage:
    from src.risk.commands import (
        CommandInvoker, TriggerCircuitBreakerCommand,
        SetCooldownCommand, CommandResult
    )

    # Create invoker and execute commands
    invoker = CommandInvoker()

    # Execute command
    cmd = TriggerCircuitBreakerCommand(risk_manager, reason="5 consecutive losses")
    result = invoker.execute(cmd)

    # Undo if needed
    if result.success:
        undo_result = invoker.undo()

Author: USD/COP Trading System
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import logging
import copy

if TYPE_CHECKING:
    from .risk_manager import RiskManager, RiskLimits

logger = logging.getLogger(__name__)


@dataclass
class CommandResult:
    """
    Result of a command execution.

    Attributes:
        success: Whether the command executed successfully
        message: Human-readable result message
        timestamp: When the command was executed
        data: Optional additional data from execution
    """
    success: bool
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    data: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class Command(ABC):
    """
    Base class for all commands.

    Implements the Command Pattern with execute and undo operations.
    Each command encapsulates a single action that can be undone.
    """

    def __init__(self):
        """Initialize command with execution tracking."""
        self._executed: bool = False
        self._executed_at: Optional[datetime] = None
        self._result: Optional[CommandResult] = None

    @abstractmethod
    def execute(self) -> CommandResult:
        """
        Execute the command.

        Returns:
            CommandResult indicating success/failure
        """
        pass

    @abstractmethod
    def undo(self) -> CommandResult:
        """
        Undo the command.

        Returns:
            CommandResult indicating success/failure of undo
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """
        Get human-readable description of the command.

        Returns:
            Description string
        """
        pass

    @property
    def executed(self) -> bool:
        """Check if command has been executed."""
        return self._executed

    @property
    def executed_at(self) -> Optional[datetime]:
        """Get execution timestamp."""
        return self._executed_at

    @property
    def result(self) -> Optional[CommandResult]:
        """Get execution result."""
        return self._result

    def can_undo(self) -> bool:
        """
        Check if the command can be undone.

        Returns:
            True if undo is possible
        """
        return self._executed

    def _mark_executed(self, result: CommandResult) -> None:
        """Mark command as executed."""
        self._executed = True
        self._executed_at = datetime.now()
        self._result = result

    def _mark_undone(self) -> None:
        """Mark command as undone."""
        self._executed = False

    def __repr__(self) -> str:
        status = "executed" if self._executed else "pending"
        return f"{self.__class__.__name__}(status={status})"


class TriggerCircuitBreakerCommand(Command):
    """
    Command to trigger the circuit breaker.

    Activates the circuit breaker (cooldown state) on the risk manager
    to pause trading after consecutive losses or other risk events.
    """

    def __init__(
        self,
        risk_manager: "RiskManager",
        reason: str,
        cooldown_minutes: Optional[int] = None
    ):
        """
        Initialize circuit breaker command.

        Args:
            risk_manager: RiskManager instance to modify
            reason: Reason for triggering circuit breaker
            cooldown_minutes: Override cooldown duration (uses default if None)
        """
        super().__init__()
        self._risk_manager = risk_manager
        self._reason = reason
        self._cooldown_minutes = cooldown_minutes

        # State for undo
        self._previous_cooldown_until: Optional[datetime] = None
        self._previous_consecutive_losses: int = 0

    def execute(self) -> CommandResult:
        """
        Execute circuit breaker trigger.

        Returns:
            CommandResult with execution status
        """
        try:
            # Save current state for undo
            self._previous_cooldown_until = self._risk_manager._cooldown_until
            self._previous_consecutive_losses = self._risk_manager._consecutive_losses

            # Determine cooldown duration
            duration = self._cooldown_minutes or self._risk_manager.limits.cooldown_minutes
            cooldown_until = datetime.now() + timedelta(minutes=duration)

            # Trigger cooldown
            self._risk_manager._cooldown_until = cooldown_until

            logger.warning(
                f"CIRCUIT BREAKER TRIGGERED: {self._reason}. "
                f"Trading paused until {cooldown_until.strftime('%H:%M:%S')}"
            )

            result = CommandResult(
                success=True,
                message=f"Circuit breaker activated until {cooldown_until.strftime('%H:%M:%S')}",
                data={
                    "reason": self._reason,
                    "cooldown_until": cooldown_until.isoformat(),
                    "duration_minutes": duration
                }
            )
            self._mark_executed(result)
            return result

        except Exception as e:
            logger.error(f"Failed to trigger circuit breaker: {e}")
            return CommandResult(
                success=False,
                message=f"Failed to trigger circuit breaker: {e}"
            )

    def undo(self) -> CommandResult:
        """
        Undo circuit breaker trigger.

        Returns:
            CommandResult with undo status
        """
        if not self._executed:
            return CommandResult(
                success=False,
                message="Command has not been executed"
            )

        try:
            # Restore previous state
            self._risk_manager._cooldown_until = self._previous_cooldown_until
            self._risk_manager._consecutive_losses = self._previous_consecutive_losses

            logger.info("Circuit breaker trigger undone")

            self._mark_undone()
            return CommandResult(
                success=True,
                message="Circuit breaker deactivated"
            )

        except Exception as e:
            logger.error(f"Failed to undo circuit breaker: {e}")
            return CommandResult(
                success=False,
                message=f"Failed to undo: {e}"
            )

    def get_description(self) -> str:
        """Get command description."""
        return f"Trigger circuit breaker: {self._reason}"


class SetCooldownCommand(Command):
    """
    Command to set or extend the cooldown period.

    Sets a specific cooldown end time on the risk manager,
    preventing new trades until the cooldown expires.
    """

    def __init__(
        self,
        risk_manager: "RiskManager",
        cooldown_minutes: int,
        reason: str = "manual"
    ):
        """
        Initialize cooldown command.

        Args:
            risk_manager: RiskManager instance to modify
            cooldown_minutes: Duration of cooldown in minutes
            reason: Reason for setting cooldown
        """
        super().__init__()
        self._risk_manager = risk_manager
        self._cooldown_minutes = cooldown_minutes
        self._reason = reason

        # State for undo
        self._previous_cooldown_until: Optional[datetime] = None

    def execute(self) -> CommandResult:
        """
        Execute cooldown set.

        Returns:
            CommandResult with execution status
        """
        try:
            # Save current state for undo
            self._previous_cooldown_until = self._risk_manager._cooldown_until

            # Set new cooldown
            cooldown_until = datetime.now() + timedelta(minutes=self._cooldown_minutes)
            self._risk_manager._cooldown_until = cooldown_until

            logger.info(
                f"Cooldown set: {self._cooldown_minutes} minutes "
                f"(until {cooldown_until.strftime('%H:%M:%S')}), reason: {self._reason}"
            )

            result = CommandResult(
                success=True,
                message=f"Cooldown set for {self._cooldown_minutes} minutes",
                data={
                    "cooldown_until": cooldown_until.isoformat(),
                    "duration_minutes": self._cooldown_minutes,
                    "reason": self._reason
                }
            )
            self._mark_executed(result)
            return result

        except Exception as e:
            logger.error(f"Failed to set cooldown: {e}")
            return CommandResult(
                success=False,
                message=f"Failed to set cooldown: {e}"
            )

    def undo(self) -> CommandResult:
        """
        Undo cooldown set.

        Returns:
            CommandResult with undo status
        """
        if not self._executed:
            return CommandResult(
                success=False,
                message="Command has not been executed"
            )

        try:
            # Restore previous cooldown state
            self._risk_manager._cooldown_until = self._previous_cooldown_until

            logger.info("Cooldown setting undone")

            self._mark_undone()
            return CommandResult(
                success=True,
                message="Cooldown restored to previous state"
            )

        except Exception as e:
            logger.error(f"Failed to undo cooldown: {e}")
            return CommandResult(
                success=False,
                message=f"Failed to undo: {e}"
            )

    def get_description(self) -> str:
        """Get command description."""
        return f"Set cooldown: {self._cooldown_minutes} minutes ({self._reason})"


class ClearCooldownCommand(Command):
    """
    Command to clear the active cooldown.

    Removes any active cooldown, allowing trading to resume immediately.
    """

    def __init__(
        self,
        risk_manager: "RiskManager",
        reason: str = "manual"
    ):
        """
        Initialize clear cooldown command.

        Args:
            risk_manager: RiskManager instance to modify
            reason: Reason for clearing cooldown
        """
        super().__init__()
        self._risk_manager = risk_manager
        self._reason = reason

        # State for undo
        self._previous_cooldown_until: Optional[datetime] = None
        self._previous_consecutive_losses: int = 0

    def execute(self) -> CommandResult:
        """
        Execute cooldown clear.

        Returns:
            CommandResult with execution status
        """
        try:
            # Check if cooldown is active
            if self._risk_manager._cooldown_until is None:
                return CommandResult(
                    success=False,
                    message="No active cooldown to clear"
                )

            # Save current state for undo
            self._previous_cooldown_until = self._risk_manager._cooldown_until
            self._previous_consecutive_losses = self._risk_manager._consecutive_losses

            # Clear cooldown
            self._risk_manager._cooldown_until = None
            self._risk_manager._consecutive_losses = 0

            logger.info(f"Cooldown cleared: {self._reason}")

            result = CommandResult(
                success=True,
                message="Cooldown cleared, trading can resume",
                data={"reason": self._reason}
            )
            self._mark_executed(result)
            return result

        except Exception as e:
            logger.error(f"Failed to clear cooldown: {e}")
            return CommandResult(
                success=False,
                message=f"Failed to clear cooldown: {e}"
            )

    def undo(self) -> CommandResult:
        """
        Undo cooldown clear.

        Returns:
            CommandResult with undo status
        """
        if not self._executed:
            return CommandResult(
                success=False,
                message="Command has not been executed"
            )

        try:
            # Restore previous cooldown state
            self._risk_manager._cooldown_until = self._previous_cooldown_until
            self._risk_manager._consecutive_losses = self._previous_consecutive_losses

            logger.info("Cooldown clear undone")

            self._mark_undone()
            return CommandResult(
                success=True,
                message="Cooldown restored"
            )

        except Exception as e:
            logger.error(f"Failed to undo cooldown clear: {e}")
            return CommandResult(
                success=False,
                message=f"Failed to undo: {e}"
            )

    def get_description(self) -> str:
        """Get command description."""
        return f"Clear cooldown: {self._reason}"


class ResetKillSwitchCommand(Command):
    """
    Command to reset the kill switch.

    Resets the kill switch state, allowing trading to resume
    after a major drawdown event. Requires explicit confirmation.
    """

    def __init__(
        self,
        risk_manager: "RiskManager",
        confirmed: bool = False,
        reason: str = "manual reset"
    ):
        """
        Initialize kill switch reset command.

        Args:
            risk_manager: RiskManager instance to modify
            confirmed: Explicit confirmation required for execution
            reason: Reason for resetting kill switch
        """
        super().__init__()
        self._risk_manager = risk_manager
        self._confirmed = confirmed
        self._reason = reason

        # State for undo
        self._previous_kill_switch_state: bool = False

    def execute(self) -> CommandResult:
        """
        Execute kill switch reset.

        Returns:
            CommandResult with execution status
        """
        # Require confirmation for safety
        if not self._confirmed:
            return CommandResult(
                success=False,
                message="Kill switch reset requires explicit confirmation"
            )

        try:
            # Check if kill switch is active
            if not self._risk_manager._kill_switch_active:
                return CommandResult(
                    success=False,
                    message="Kill switch is not active"
                )

            # Save state for undo
            self._previous_kill_switch_state = self._risk_manager._kill_switch_active

            # Reset kill switch
            self._risk_manager._kill_switch_active = False

            logger.critical(
                f"KILL SWITCH RESET: {self._reason}. "
                "Trading can resume. Review risk limits before continuing."
            )

            result = CommandResult(
                success=True,
                message="Kill switch reset, trading can resume",
                data={
                    "reason": self._reason,
                    "warning": "Review risk limits before continuing trading"
                }
            )
            self._mark_executed(result)
            return result

        except Exception as e:
            logger.error(f"Failed to reset kill switch: {e}")
            return CommandResult(
                success=False,
                message=f"Failed to reset kill switch: {e}"
            )

    def undo(self) -> CommandResult:
        """
        Undo kill switch reset (re-activate kill switch).

        Returns:
            CommandResult with undo status
        """
        if not self._executed:
            return CommandResult(
                success=False,
                message="Command has not been executed"
            )

        try:
            # Re-activate kill switch
            self._risk_manager._kill_switch_active = self._previous_kill_switch_state

            logger.warning("Kill switch reset undone - kill switch re-activated")

            self._mark_undone()
            return CommandResult(
                success=True,
                message="Kill switch re-activated"
            )

        except Exception as e:
            logger.error(f"Failed to undo kill switch reset: {e}")
            return CommandResult(
                success=False,
                message=f"Failed to undo: {e}"
            )

    def get_description(self) -> str:
        """Get command description."""
        return f"Reset kill switch: {self._reason}"


class UpdateRiskLimitsCommand(Command):
    """
    Command to update risk limits.

    Modifies the risk limits configuration on the risk manager.
    Supports partial updates to individual limit values.
    """

    def __init__(
        self,
        risk_manager: "RiskManager",
        new_limits: Dict[str, Any],
        reason: str = "limit adjustment"
    ):
        """
        Initialize risk limits update command.

        Args:
            risk_manager: RiskManager instance to modify
            new_limits: Dict of limit names to new values
            reason: Reason for updating limits
        """
        super().__init__()
        self._risk_manager = risk_manager
        self._new_limits = new_limits
        self._reason = reason

        # State for undo
        self._previous_limits: Dict[str, Any] = {}

    def execute(self) -> CommandResult:
        """
        Execute risk limits update.

        Returns:
            CommandResult with execution status
        """
        try:
            limits = self._risk_manager.limits

            # Save current values for undo
            updated = {}
            for key, value in self._new_limits.items():
                if hasattr(limits, key):
                    self._previous_limits[key] = getattr(limits, key)
                    setattr(limits, key, value)
                    updated[key] = {"old": self._previous_limits[key], "new": value}
                else:
                    logger.warning(f"Unknown risk limit: {key}")

            if not updated:
                return CommandResult(
                    success=False,
                    message="No valid limits to update"
                )

            logger.info(
                f"Risk limits updated: {updated}, reason: {self._reason}"
            )

            result = CommandResult(
                success=True,
                message=f"Updated {len(updated)} risk limits",
                data={
                    "updates": updated,
                    "reason": self._reason
                }
            )
            self._mark_executed(result)
            return result

        except Exception as e:
            logger.error(f"Failed to update risk limits: {e}")
            return CommandResult(
                success=False,
                message=f"Failed to update limits: {e}"
            )

    def undo(self) -> CommandResult:
        """
        Undo risk limits update.

        Returns:
            CommandResult with undo status
        """
        if not self._executed:
            return CommandResult(
                success=False,
                message="Command has not been executed"
            )

        try:
            limits = self._risk_manager.limits

            # Restore previous values
            for key, value in self._previous_limits.items():
                setattr(limits, key, value)

            logger.info("Risk limits update undone")

            self._mark_undone()
            return CommandResult(
                success=True,
                message="Risk limits restored to previous values"
            )

        except Exception as e:
            logger.error(f"Failed to undo risk limits update: {e}")
            return CommandResult(
                success=False,
                message=f"Failed to undo: {e}"
            )

    def get_description(self) -> str:
        """Get command description."""
        return f"Update risk limits: {list(self._new_limits.keys())}"


class BlockTradingCommand(Command):
    """
    Command to block trading for the day.

    Activates the daily block flag, preventing new trades
    until the next trading day.
    """

    def __init__(
        self,
        risk_manager: "RiskManager",
        reason: str = "manual block"
    ):
        """
        Initialize block trading command.

        Args:
            risk_manager: RiskManager instance to modify
            reason: Reason for blocking trading
        """
        super().__init__()
        self._risk_manager = risk_manager
        self._reason = reason

        # State for undo
        self._previous_daily_blocked: bool = False

    def execute(self) -> CommandResult:
        """
        Execute trading block.

        Returns:
            CommandResult with execution status
        """
        try:
            # Save state for undo
            self._previous_daily_blocked = self._risk_manager._daily_blocked

            # Block trading
            self._risk_manager._daily_blocked = True

            logger.warning(f"TRADING BLOCKED: {self._reason}")

            result = CommandResult(
                success=True,
                message="Trading blocked for today",
                data={"reason": self._reason}
            )
            self._mark_executed(result)
            return result

        except Exception as e:
            logger.error(f"Failed to block trading: {e}")
            return CommandResult(
                success=False,
                message=f"Failed to block trading: {e}"
            )

    def undo(self) -> CommandResult:
        """
        Undo trading block.

        Returns:
            CommandResult with undo status
        """
        if not self._executed:
            return CommandResult(
                success=False,
                message="Command has not been executed"
            )

        try:
            # Restore previous state
            self._risk_manager._daily_blocked = self._previous_daily_blocked

            logger.info("Trading block undone")

            self._mark_undone()
            return CommandResult(
                success=True,
                message="Trading block removed"
            )

        except Exception as e:
            logger.error(f"Failed to undo trading block: {e}")
            return CommandResult(
                success=False,
                message=f"Failed to undo: {e}"
            )

    def get_description(self) -> str:
        """Get command description."""
        return f"Block trading: {self._reason}"


class CommandInvoker:
    """
    Executes commands and maintains command history.

    Provides:
    - Command execution with result tracking
    - Undo/redo support
    - Command history for audit trails
    - Macro command support (execute multiple commands)
    """

    def __init__(self, max_history: int = 100):
        """
        Initialize command invoker.

        Args:
            max_history: Maximum commands to keep in history
        """
        self._history: List[Command] = []
        self._redo_stack: List[Command] = []
        self._max_history = max_history

    def execute(self, command: Command) -> CommandResult:
        """
        Execute a command and add to history.

        Args:
            command: Command to execute

        Returns:
            CommandResult from execution
        """
        logger.debug(f"Executing command: {command.get_description()}")

        result = command.execute()

        if result.success:
            self._history.append(command)
            self._redo_stack.clear()  # Clear redo stack on new command

            # Trim history if needed
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

        logger.info(
            f"Command '{command.get_description()}' "
            f"{'succeeded' if result.success else 'failed'}: {result.message}"
        )

        return result

    def undo(self) -> CommandResult:
        """
        Undo the last executed command.

        Returns:
            CommandResult from undo operation
        """
        if not self._history:
            return CommandResult(
                success=False,
                message="No commands to undo"
            )

        command = self._history.pop()
        logger.debug(f"Undoing command: {command.get_description()}")

        result = command.undo()

        if result.success:
            self._redo_stack.append(command)

        return result

    def redo(self) -> CommandResult:
        """
        Redo the last undone command.

        Returns:
            CommandResult from redo operation
        """
        if not self._redo_stack:
            return CommandResult(
                success=False,
                message="No commands to redo"
            )

        command = self._redo_stack.pop()
        logger.debug(f"Redoing command: {command.get_description()}")

        result = command.execute()

        if result.success:
            self._history.append(command)

        return result

    def can_undo(self) -> bool:
        """Check if undo is available."""
        return len(self._history) > 0

    def can_redo(self) -> bool:
        """Check if redo is available."""
        return len(self._redo_stack) > 0

    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get command history.

        Args:
            limit: Maximum commands to return

        Returns:
            List of command info dicts
        """
        commands = self._history[-limit:] if self._history else []
        return [
            {
                "description": cmd.get_description(),
                "executed_at": cmd.executed_at.isoformat() if cmd.executed_at else None,
                "success": cmd.result.success if cmd.result else None,
                "message": cmd.result.message if cmd.result else None
            }
            for cmd in reversed(commands)
        ]

    def clear_history(self) -> None:
        """Clear all command history."""
        self._history.clear()
        self._redo_stack.clear()
        logger.info("Command history cleared")

    def execute_batch(
        self,
        commands: List[Command],
        stop_on_failure: bool = True
    ) -> List[CommandResult]:
        """
        Execute multiple commands in sequence.

        Args:
            commands: List of commands to execute
            stop_on_failure: Stop execution on first failure

        Returns:
            List of CommandResults
        """
        results = []

        for command in commands:
            result = self.execute(command)
            results.append(result)

            if not result.success and stop_on_failure:
                logger.warning(
                    f"Batch execution stopped: {command.get_description()} failed"
                )
                break

        return results

    def __repr__(self) -> str:
        return (
            f"CommandInvoker(history={len(self._history)}, "
            f"redo_stack={len(self._redo_stack)})"
        )
