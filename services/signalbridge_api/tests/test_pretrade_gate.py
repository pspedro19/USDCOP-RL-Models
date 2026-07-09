"""PreTradeGate unit tests (audit OLA-3 S2/S3) — mode semantics + fail-safe behavior.

The gate is the last line of defense before an order reaches an exchange; these tests pin
its contract: PAPER simulates, KILLED blocks, no-opt-in blocks, notional cap blocks, and
ANY internal error blocks (never fail-open).
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.pretrade import PreTradeAction, PreTradeGate  # noqa: E402

UID = uuid.uuid4()


class _Row:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Result:
    def __init__(self, row):
        self._row = row

    def first(self):
        return self._row

    def scalar(self):
        return self._row


class FakeDB:
    """Async session stub: returns queued rows per execute() call, or raises."""

    def __init__(self, rows=(), raise_on_execute=False):
        self.rows = list(rows)
        self.raise_on_execute = raise_on_execute

    async def execute(self, *_a, **_k):
        if self.raise_on_execute:
            raise RuntimeError("db down")
        return _Result(self.rows.pop(0) if self.rows else None)


def _check(gate, **kw):
    import asyncio
    defaults = dict(user_id=UID, symbol="BTC/USDT", quantity=0.0001, price=60000.0,
                    is_testnet_credential=False)
    defaults.update(kw)
    return asyncio.get_event_loop().run_until_complete(gate.check(**defaults))


@pytest.fixture(autouse=True)
def _event_loop_policy():
    import asyncio
    asyncio.set_event_loop(asyncio.new_event_loop())
    yield


def _with_mode(mode):
    return patch("app.services.pretrade.settings") and mode


def test_paper_mode_simulates_never_sends():
    with patch("app.services.pretrade.settings") as s:
        s.trading_mode = "PAPER"
        d = _check(PreTradeGate(FakeDB()))
    assert d.action == PreTradeAction.SIMULATE and d.simulated and not d.blocked


def test_killed_mode_blocks_everything():
    with patch("app.services.pretrade.settings") as s:
        s.trading_mode = "KILLED"
        d = _check(PreTradeGate(FakeDB()))
    assert d.blocked


def test_staging_requires_testnet_credential():
    with patch("app.services.pretrade.settings") as s:
        s.trading_mode = "STAGING"
        d = _check(PreTradeGate(FakeDB()), is_testnet_credential=False)
    assert d.blocked and "testnet" in d.reason


def test_live_blocks_without_user_opt_in():
    """No sb_trading_configs row => trading_enabled default False => BLOCK (safe-by-default)."""
    with patch("app.services.pretrade.settings") as s:
        s.trading_mode = "LIVE"
        s.max_position_size_usd = 1000.0
        d = _check(PreTradeGate(FakeDB(rows=[None])))
    assert d.blocked and "disabled" in d.reason


def test_live_blocks_when_notional_exceeds_cap():
    cfg = _Row(trading_enabled=True, allowed_symbols=[], blocked_symbols=[], max_daily_trades=0)
    with patch("app.services.pretrade.settings") as s:
        s.trading_mode = "LIVE"
        s.max_position_size_usd = 10.0  # $10 cap; order below is ~$6k
        d = _check(PreTradeGate(FakeDB(rows=[cfg, None])), quantity=0.1, price=60000.0)
    assert d.blocked and "exceeds max" in d.reason


def test_live_allows_within_limits():
    cfg = _Row(trading_enabled=True, allowed_symbols=[], blocked_symbols=[], max_daily_trades=0)
    lim = _Row(max_position_size_usd=100.0, max_trades_per_day=None)
    with patch("app.services.pretrade.settings") as s:
        s.trading_mode = "LIVE"
        s.max_position_size_usd = 100.0
        d = _check(PreTradeGate(FakeDB(rows=[cfg, lim])), quantity=0.0001, price=60000.0)
    assert d.action == PreTradeAction.ALLOW


def test_blocked_symbol_is_rejected():
    cfg = _Row(trading_enabled=True, allowed_symbols=[], blocked_symbols=["BTC/USDT"],
               max_daily_trades=0)
    with patch("app.services.pretrade.settings") as s:
        s.trading_mode = "LIVE"
        d = _check(PreTradeGate(FakeDB(rows=[cfg])))
    assert d.blocked and "blocked" in d.reason


def test_gate_error_fails_safe_to_block():
    """DB down in LIVE => BLOCK, never fail-open with real money."""
    with patch("app.services.pretrade.settings") as s:
        s.trading_mode = "LIVE"
        d = _check(PreTradeGate(FakeDB(raise_on_execute=True)))
    assert d.blocked
