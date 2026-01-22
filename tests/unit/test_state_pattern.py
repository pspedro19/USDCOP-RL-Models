"""
Unit Tests for State Pattern - Trade Lifecycle States
======================================================

Tests for the State Pattern implementation in src/trading/states.py

Author: USD/COP Trading System
"""

import pytest
from datetime import datetime

from src.trading.states import (
    TradeState,
    PendingState,
    OpenState,
    ClosingState,
    ClosedState,
    Trade,
)


class TestPendingState:
    """Tests for PendingState."""

    def test_initial_status(self):
        """Test pending state returns correct status."""
        state = PendingState()
        assert state.get_status() == "pending"

    def test_cannot_close_pending_trade(self):
        """Test that pending trades cannot be closed."""
        state = PendingState()
        assert state.can_close() is False

    def test_can_modify_pending_trade(self):
        """Test that pending trades can be modified."""
        state = PendingState()
        assert state.can_modify() is True

    def test_price_update_no_transition(self):
        """Test that price updates don't transition pending trades."""
        state = PendingState()
        trade = Trade(trade_id="T001", direction="LONG")

        new_state = state.on_price_update(trade, 100.0)

        assert new_state is state  # Same instance, no transition
        assert isinstance(new_state, PendingState)

    def test_execute_transitions_to_open(self):
        """Test that executing a pending trade transitions to open."""
        state = PendingState()
        trade = Trade(trade_id="T001", direction="LONG")

        new_state = state.execute(trade, 100.0)

        assert isinstance(new_state, OpenState)


class TestOpenState:
    """Tests for OpenState."""

    def test_initial_status(self):
        """Test open state returns correct status."""
        state = OpenState()
        assert state.get_status() == "open"

    def test_can_close_open_trade(self):
        """Test that open trades can be closed."""
        state = OpenState()
        assert state.can_close() is True

    def test_can_modify_open_trade(self):
        """Test that open trades can be modified."""
        state = OpenState()
        assert state.can_modify() is True

    def test_price_update_no_sl_tp(self):
        """Test price update without SL/TP doesn't trigger transition."""
        state = OpenState()
        trade = Trade(trade_id="T001", direction="LONG", entry_price=100.0)

        new_state = state.on_price_update(trade, 105.0)

        assert isinstance(new_state, OpenState)

    def test_long_stop_loss_triggered(self):
        """Test stop loss triggers for LONG position."""
        state = OpenState()
        trade = Trade(trade_id="T001", direction="LONG", entry_price=100.0)

        new_state = state.on_price_update(trade, 90.0, stop_loss=95.0)

        assert isinstance(new_state, ClosingState)
        assert new_state.reason == "stop_loss"
        assert new_state.trigger_price == 90.0

    def test_long_take_profit_triggered(self):
        """Test take profit triggers for LONG position."""
        state = OpenState()
        trade = Trade(trade_id="T001", direction="LONG", entry_price=100.0)

        new_state = state.on_price_update(trade, 120.0, take_profit=115.0)

        assert isinstance(new_state, ClosingState)
        assert new_state.reason == "take_profit"
        assert new_state.trigger_price == 120.0

    def test_short_stop_loss_triggered(self):
        """Test stop loss triggers for SHORT position."""
        state = OpenState()
        trade = Trade(trade_id="T001", direction="SHORT", entry_price=100.0)

        new_state = state.on_price_update(trade, 110.0, stop_loss=105.0)

        assert isinstance(new_state, ClosingState)
        assert new_state.reason == "stop_loss"

    def test_short_take_profit_triggered(self):
        """Test take profit triggers for SHORT position."""
        state = OpenState()
        trade = Trade(trade_id="T001", direction="SHORT", entry_price=100.0)

        new_state = state.on_price_update(trade, 85.0, take_profit=90.0)

        assert isinstance(new_state, ClosingState)
        assert new_state.reason == "take_profit"

    def test_no_trigger_within_bounds(self):
        """Test no trigger when price is within SL/TP bounds."""
        state = OpenState()
        trade = Trade(trade_id="T001", direction="LONG", entry_price=100.0)

        new_state = state.on_price_update(
            trade, 102.0, stop_loss=95.0, take_profit=115.0
        )

        assert isinstance(new_state, OpenState)

    def test_request_close(self):
        """Test requesting manual close transitions to closing."""
        state = OpenState()

        new_state = state.request_close("manual")

        assert isinstance(new_state, ClosingState)
        assert new_state.reason == "manual"


class TestClosingState:
    """Tests for ClosingState."""

    def test_status_includes_reason(self):
        """Test closing state status includes reason."""
        state = ClosingState(reason="stop_loss")
        assert state.get_status() == "closing:stop_loss"

    def test_cannot_close_closing_trade(self):
        """Test that closing trades cannot be closed again."""
        state = ClosingState(reason="manual")
        assert state.can_close() is False

    def test_cannot_modify_closing_trade(self):
        """Test that closing trades cannot be modified."""
        state = ClosingState(reason="manual")
        assert state.can_modify() is False

    def test_price_update_no_transition(self):
        """Test that price updates don't affect closing trades."""
        state = ClosingState(reason="manual")
        trade = Trade(trade_id="T001", direction="LONG")

        new_state = state.on_price_update(trade, 200.0)

        assert new_state is state

    def test_complete_transitions_to_closed(self):
        """Test completing a closing trade transitions to closed."""
        state = ClosingState(reason="take_profit")

        new_state = state.complete()

        assert isinstance(new_state, ClosedState)
        assert new_state.close_reason == "take_profit"


class TestClosedState:
    """Tests for ClosedState."""

    def test_status_includes_reason(self):
        """Test closed state status includes reason."""
        state = ClosedState(close_reason="stop_loss")
        assert state.get_status() == "closed:stop_loss"

    def test_cannot_close_closed_trade(self):
        """Test that closed trades cannot be closed."""
        state = ClosedState()
        assert state.can_close() is False

    def test_cannot_modify_closed_trade(self):
        """Test that closed trades cannot be modified."""
        state = ClosedState()
        assert state.can_modify() is False

    def test_price_update_no_effect(self):
        """Test that price updates have no effect on closed trades."""
        state = ClosedState(close_reason="manual")
        trade = Trade(trade_id="T001", direction="LONG")

        new_state = state.on_price_update(trade, 999.0, stop_loss=1.0, take_profit=9999.0)

        assert new_state is state
        assert isinstance(new_state, ClosedState)


class TestTrade:
    """Tests for Trade entity with state-based behavior."""

    def test_initial_state_is_pending(self):
        """Test trade starts in pending state by default."""
        trade = Trade(trade_id="T001", direction="LONG")

        assert trade.get_status() == "pending"
        assert isinstance(trade.state, PendingState)

    def test_execute_pending_trade(self):
        """Test executing a pending trade."""
        trade = Trade(trade_id="T001", direction="LONG")

        success = trade.execute(100.0)

        assert success is True
        assert trade.entry_price == 100.0
        assert trade.get_status() == "open"

    def test_cannot_execute_non_pending_trade(self):
        """Test that non-pending trades cannot be executed."""
        trade = Trade(trade_id="T001", direction="LONG", state=OpenState())

        success = trade.execute(100.0)

        assert success is False

    def test_update_price_triggers_stop_loss(self):
        """Test price update triggering stop loss."""
        trade = Trade(
            trade_id="T001",
            direction="LONG",
            entry_price=100.0,
            stop_loss=95.0,
            state=OpenState()
        )

        trade.update_price(90.0)

        assert "closing" in trade.get_status()

    def test_close_trade(self):
        """Test closing a trade."""
        trade = Trade(
            trade_id="T001",
            direction="LONG",
            entry_price=100.0,
            state=OpenState()
        )

        success = trade.close(105.0, reason="manual")

        assert success is True
        assert trade.exit_price == 105.0
        assert "closed" in trade.get_status()

    def test_cannot_close_pending_trade(self):
        """Test that pending trades cannot be closed."""
        trade = Trade(trade_id="T001", direction="LONG")

        success = trade.close(100.0)

        assert success is False

    def test_modify_stop_loss(self):
        """Test modifying stop loss."""
        trade = Trade(
            trade_id="T001",
            direction="LONG",
            stop_loss=95.0,
            state=OpenState()
        )

        success = trade.modify_stop_loss(92.0)

        assert success is True
        assert trade.stop_loss == 92.0

    def test_cannot_modify_closed_trade(self):
        """Test that closed trades cannot be modified."""
        trade = Trade(
            trade_id="T001",
            direction="LONG",
            state=ClosedState(close_reason="manual")
        )

        success = trade.modify_stop_loss(90.0)

        assert success is False

    def test_full_lifecycle(self):
        """Test full trade lifecycle: pending -> open -> closing -> closed."""
        # Create pending trade
        trade = Trade(
            trade_id="T001",
            direction="LONG",
            stop_loss=95.0,
            take_profit=110.0
        )
        assert trade.get_status() == "pending"

        # Execute trade
        trade.execute(100.0)
        assert trade.get_status() == "open"

        # Modify stop loss while open
        trade.modify_stop_loss(97.0)
        assert trade.stop_loss == 97.0

        # Price update doesn't trigger
        trade.update_price(102.0)
        assert trade.get_status() == "open"

        # Price hits take profit
        trade.update_price(112.0)
        assert "closing" in trade.get_status()

        # Complete the close
        trade.state = trade.state.complete()
        assert "closed" in trade.get_status()

        # Cannot modify closed trade
        assert trade.modify_stop_loss(80.0) is False

    def test_repr(self):
        """Test trade string representation."""
        trade = Trade(trade_id="T001", direction="LONG")
        repr_str = repr(trade)

        assert "T001" in repr_str
        assert "LONG" in repr_str
        assert "pending" in repr_str
