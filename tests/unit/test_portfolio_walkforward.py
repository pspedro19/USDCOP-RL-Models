"""Regression locks for temporal boundaries in the portfolio analysis harness."""

from datetime import date, timedelta

import numpy as np

from scripts.analysis.portfolio_walkforward import sleeve_is_live, strategy_daily_exact


def test_sleeve_liveness_uses_only_the_declared_trailing_window() -> None:
    position = np.zeros(300)
    cutoff = 250
    position[cutoff - 64] = 1.0

    assert sleeve_is_live(position, cutoff) is False

    position[cutoff - 63] = 1.0
    assert sleeve_is_live(position, cutoff) is True


def test_sleeve_liveness_does_not_look_at_the_selection_day() -> None:
    position = np.zeros(100)
    cutoff = 80
    position[cutoff] = 1.0

    assert sleeve_is_live(position, cutoff) is False


def test_exact_daily_return_does_not_credit_the_pre_entry_gap() -> None:
    start = date(2026, 1, 1)
    days = [start + timedelta(days=i) for i in range(3)]
    closes = np.array([100.0, 120.0, 132.0])
    trades = [{
        "timestamp": "2026-01-02",
        "exit_timestamp": "2026-01-03",
        "entry_price": 110.0,
        "exit_price": 132.0,
        "side": "LONG",
        "leverage": 1.0,
    }]

    returns, position = strategy_daily_exact(trades, days, closes)

    np.testing.assert_allclose(returns, [0.0, 120.0 / 110.0 - 1.0, 132.0 / 120.0 - 1.0])
    np.testing.assert_allclose(position, [0.0, 1.0, 1.0])
