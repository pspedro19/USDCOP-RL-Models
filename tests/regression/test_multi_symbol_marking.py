"""A multi-asset book may not be marked to a single price.

Contract: CTR-QUANT-PORTFOLIO-001

`PositionTracker.get_total_unrealized_pnl` took one scalar `current_price` and applied it to
every open position:

    sum(pos.calculate_unrealized_pnl(current_price) for pos in self._positions.values())

For a single-asset book that is correct, and it was never wrong in production because nothing
here has ever run a book. The moment the ERC portfolio (gold + BTC + SPX500) becomes
executable, that line marks gold at BTC's price and returns a confident number that is not P&L.

Positions also had no `symbol` field at all, so the tracker was structurally incapable of
knowing which price belonged to which position.

The fix refuses rather than guesses: a scalar is still accepted for a genuinely single-symbol
book, but raises when the book spans symbols, and a partial price map raises too. The one thing
worse than failing to compute P&L is computing a wrong one that looks fine.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from src.trading.position_tracker import (
    Position, PositionDirection, PositionTracker,
)


def _pos(i: int, symbol: str | None, entry: float) -> Position:
    return Position(
        position_id=i, model_id=f"m{i}", direction=PositionDirection.LONG,
        size=1.0, entry_price=entry, entry_time=datetime.now(), symbol=symbol,
    )


def _tracker(*positions: Position) -> PositionTracker:
    t = PositionTracker()
    t._positions = {p.model_id: p for p in positions}
    return t


def test_position_carries_a_symbol():
    assert "symbol" in Position.__dataclass_fields__, (
        "Position has no symbol field, so no price can be matched to it. A tracker without "
        "symbols cannot hold a book."
    )


def test_scalar_price_refused_for_multi_symbol_book():
    t = _tracker(_pos(1, "XAUUSD", 2000.0), _pos(2, "BTCUSDT", 60000.0))
    with pytest.raises(ValueError, match="span 2 symbols"):
        t.get_total_unrealized_pnl(2100.0)


def test_price_map_marks_each_position_correctly():
    t = _tracker(_pos(1, "XAUUSD", 2000.0), _pos(2, "BTCUSDT", 60000.0))
    # gold +100, BTC +6000
    assert t.get_total_unrealized_pnl({"XAUUSD": 2100.0, "BTCUSDT": 66000.0}) == pytest.approx(6100.0)


def test_partial_price_map_refused():
    """Silently skipping an unpriced leg under-reports the book, which is the flattering error."""
    t = _tracker(_pos(1, "XAUUSD", 2000.0), _pos(2, "BTCUSDT", 60000.0))
    with pytest.raises(ValueError, match="no price supplied"):
        t.get_total_unrealized_pnl({"XAUUSD": 2100.0})


def test_scalar_still_works_for_single_symbol_book():
    """Backwards compatibility: the pre-existing single-asset callers must keep working."""
    t = _tracker(_pos(1, "XAUUSD", 2000.0), _pos(2, "XAUUSD", 2050.0))
    assert t.get_total_unrealized_pnl(2100.0) == pytest.approx(150.0)
