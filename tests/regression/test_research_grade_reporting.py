import numpy as np
import pytest

from src.research.reporting_v2 import (
    capital_path,
    hierarchical_sharpe,
    holm,
    paired_difference,
    path_counts,
    summarize,
)


def test_null_centered_mean_bootstrap_uses_one_plus_correction():
    result = paired_difference(
        [-0.01] * 30,
        [0.0] * 30,
        min_trades_a=30,
        min_trades_b=0,
        flat_reference=True,
        replications=100,
    )
    assert result["null_exceedances"] == 0
    assert result["p_centered_stationary"] == pytest.approx(1 / 101)
    assert result["confirmatory"] is False


def test_first_loss_is_drawdown():
    r = summarize([-0.1, -0.1], [0, 0], min_round_trips=2)
    assert r["return_compounded_pct"] == pytest.approx(-19)
    assert r["max_drawdown_pct"] == pytest.approx(-19)
    assert len(capital_path([-0.1, -0.1])) == 3


def test_cash_accounting_and_daily_percentages_are_distinct():
    r = summarize([0.1, -0.1], [0.01, 0.02], min_round_trips=20, initial=100.0)
    assert r["cost_account_units"] == pytest.approx(1 + 109 * 0.02)
    assert r["final_capital_account_units"] - 100 == pytest.approx(
        r["gross_pnl_account_units"] - r["cost_account_units"]
    )
    assert r["net_sum_pct"] != pytest.approx(r["return_compounded_pct"])


def test_zero_variance_and_low_trade_count_are_suppressed():
    assert summarize([0, 0], [0, 0], min_round_trips=0)["sharpe"] is None
    assert summarize([0.1, -0.1], [0, 0], min_round_trips=2)["sharpe"] is None
    assert (
        paired_difference([0.1, -0.1], [0, 0], min_trades_a=2, min_trades_b=0, flat_reference=True)[
            "status"
        ]
        == "SUPPRESSED"
    )


def test_turnover_is_not_round_trip_count():
    c = path_counts([1, 0.5, 1, -1, 0, 0.5])
    assert c["round_trips"] == 3
    assert c["changes_including_terminal"] == 7
    assert c["turnover"] == 6


def test_holm_is_monotonic_and_uses_full_family():
    assert holm({"a": 0.01, "b": 0.04, "c": 0.3}) == {"a": 0.03, "b": 0.08, "c": 0.3}


def test_hierarchical_equal_policies_have_zero_difference():
    x = np.random.default_rng(1).normal(0, 0.01, (5, 50))
    result = hierarchical_sharpe(x, x, replications=100)
    assert result["portfolio"]["ci95"] == [0.0, 0.0]
    assert result["mean_seed_sharpe"]["p_bootstrap_percentile"] == 1


@pytest.mark.parametrize("returns", [[float("nan")], [-1], [-1.1]])
def test_invalid_or_bankrupt_capital_is_not_silently_plotted(returns):
    with pytest.raises(ValueError):
        capital_path(returns)
