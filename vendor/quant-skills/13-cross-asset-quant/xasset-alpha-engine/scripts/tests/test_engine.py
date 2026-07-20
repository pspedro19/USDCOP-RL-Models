"""Tests for instruments, carry, sizing and signals."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xasset.carry import (  # noqa: E402
    carry_to_zscore,
    commodity_roll_carry,
    crypto_perp_carry,
    equity_index_carry,
    fx_carry,
    fx_carry_from_forward,
)
from xasset.instruments import Unit, get_instrument  # noqa: E402
from xasset.signals import (  # noqa: E402
    blended_tsmom,
    combine_signals,
    commodity_value_reversal,
    equity_value_earnings_yield,
    fx_value_ppp,
    tsmom,
    vol_scaled_signal,
)
from xasset.sizing import (  # noqa: E402
    fractional_kelly,
    portfolio_vol,
    realised_vol,
    risk_parity_weights,
    vol_target_weights,
)

RNG = np.random.default_rng(7)


# ==========================================================================
# Instruments - the unit-correctness guarantee
# ==========================================================================

def test_eurusd_sizes_in_base_currency_units_not_shares():
    """The bug this module exists to prevent."""
    pos = get_instrument("EURUSD").size_from_risk(
        entry=1.0850, stop=1.0800, risk_budget=500.0
    )
    assert pos.unit is Unit.BASE_CCY_UNITS
    assert pos.quantity == pytest.approx(100_000.0)  # 50 pips, $10/pip
    assert pos.lots == pytest.approx(1.0)


def test_risk_at_stop_matches_budget_for_continuous_units():
    pos = get_instrument("EURUSD").size_from_risk(1.10, 1.09, 750.0)
    assert pos.risk_at_stop == pytest.approx(750.0)


def test_gold_futures_floor_to_whole_contracts():
    """A partial gold contract does not exist; risk lands BELOW budget."""
    pos = get_instrument("GC").size_from_risk(
        entry=2400.0, stop=2380.0, risk_budget=5000.0
    )
    assert pos.unit is Unit.CONTRACTS
    assert pos.quantity == 2.0  # raw 2.5 -> floored
    assert pos.raw_quantity == pytest.approx(2.5)
    assert pos.risk_at_stop == pytest.approx(4000.0)
    assert pos.risk_at_stop < pos.risk_budget


def test_budget_too_small_for_one_contract_is_not_tradeable():
    pos = get_instrument("GC").size_from_risk(2400.0, 2380.0, risk_budget=500.0)
    assert pos.quantity == 0
    assert not pos.is_tradeable
    assert "NOT TRADEABLE" in pos.describe()


def test_jpy_pair_uses_two_decimal_pips():
    inst = get_instrument("USDJPY")
    assert inst.pip_size == 0.01
    # 1 standard lot, quote ccy JPY -> convert at 1/150
    assert inst.pip_value(100_000, fx_rate_to_account=1 / 150.0) == pytest.approx(
        6.6667, rel=1e-3
    )


def test_fx_conversion_scales_position():
    inst = get_instrument("USDJPY")
    unconverted = inst.size_from_risk(150.0, 149.0, 1000.0, fx_rate_to_account=1.0)
    converted = inst.size_from_risk(150.0, 149.0, 1000.0, fx_rate_to_account=1 / 150.0)
    assert converted.quantity == pytest.approx(unconverted.quantity * 150.0)


def test_index_future_multiplier_applied():
    pos = get_instrument("ES").size_from_risk(5000.0, 4990.0, 5000.0)
    # 10 points x $50 = $500 risk per contract -> 10 contracts
    assert pos.quantity == 10.0


def test_em_pairs_flagged_and_annotated():
    assert get_instrument("USDMXN").is_em
    assert get_instrument("USDBRL").nondeliverable
    assert get_instrument("USDTRY").capital_controls
    assert "EM" in get_instrument("USDZAR").size_from_risk(18.0, 17.8, 400.0).describe()


def test_unknown_symbol_raises_rather_than_defaulting():
    with pytest.raises(KeyError, match="Unknown instrument"):
        get_instrument("NOTAREALPAIR")


def test_stop_inside_one_tick_rejected():
    with pytest.raises(ValueError, match="smaller than"):
        get_instrument("ES").size_from_risk(5000.0, 5000.1, 1000.0)


def test_zero_stop_distance_rejected():
    with pytest.raises(ValueError, match="stop distance is zero"):
        get_instrument("EURUSD").size_from_risk(1.10, 1.10, 500.0)


def test_max_notional_caps_position():
    pos = get_instrument("EURUSD").size_from_risk(
        1.10, 1.099, risk_budget=5000.0, max_notional=1_000_000.0
    )
    assert pos.capped_by == "max_notional"
    assert pos.notional <= 1_000_000.0 + 1e-6


def test_symbol_lookup_is_normalised():
    assert get_instrument("eur/usd").symbol == "EURUSD"
    assert get_instrument("btc-usd").symbol == "BTCUSD"


# ==========================================================================
# Carry
# ==========================================================================

def test_fx_carry_sign_convention():
    """Long EURUSD earns EUR, pays USD."""
    assert fx_carry("EURUSD", rate_base=0.02, rate_quote=0.05).annualised == pytest.approx(-0.03)


def test_em_short_usdmxn_is_the_positive_carry_side():
    """Long USDMXN has negative carry; the carry trade is the short."""
    c = fx_carry("USDMXN", rate_base=0.045, rate_quote=0.11, is_em=True)
    assert c.annualised < 0


def test_em_negative_real_rate_marked_suspect():
    c = fx_carry(
        "USDTRY", rate_base=0.50, rate_quote=0.045,
        is_em=True, annual_inflation_base=0.60,
    )
    assert c.reliability == "suspect"
    assert "real rate" in c.note


def test_high_em_carry_without_inflation_input_is_suspect():
    c = fx_carry("USDZAR", rate_base=0.20, rate_quote=0.045, is_em=True)
    assert c.reliability == "suspect"


def test_g10_high_carry_not_flagged():
    """The suspect heuristic must not fire on developed markets."""
    assert fx_carry("AUDUSD", 0.045, 0.02).reliability == "normal"


def test_forward_implied_carry_matches_rate_differential_under_cip():
    """Under CIP the forward-implied carry equals the EXACT rate differential.

    Compared against (1+r_b)/(1+r_q) - 1, not the r_b - r_q linear
    approximation: at a 3-point differential those differ by ~14bp, which is
    real money in a carry book.
    """
    spot, r_base, r_quote, days = 1.10, 0.02, 0.05, 365
    fwd = spot * (1 + r_quote) / (1 + r_base)
    c = fx_carry_from_forward("EURUSD", spot, fwd, days)
    exact = (1 + r_base) / (1 + r_quote) - 1
    assert c.annualised == pytest.approx(exact, rel=1e-9)
    assert c.annualised == pytest.approx(r_base - r_quote, abs=2e-3)  # approx holds


def test_positive_funding_means_negative_carry_to_longs():
    c = crypto_perp_carry("BTCUSDT.P", funding_rate=0.0001)
    assert c.annualised == pytest.approx(-0.1095, rel=1e-3)


def test_extreme_funding_flagged():
    assert crypto_perp_carry("BTCUSDT.P", funding_rate=0.001).reliability == "suspect"


def test_backwardation_pays_longs_contango_bleeds():
    assert commodity_roll_carry("CL", 80.0, 78.0, 30).annualised > 0
    assert commodity_roll_carry("GC", 2400.0, 2420.0, 30).annualised < 0


def test_equity_index_carry_negative_when_rates_exceed_dividends():
    assert equity_index_carry("ES", 0.013, 0.045).annualised == pytest.approx(-0.032)


def test_zscore_drops_suspect_estimates():
    ests = [
        fx_carry("EURUSD", 0.02, 0.05),
        fx_carry("AUDUSD", 0.045, 0.02),
        fx_carry("USDTRY", 0.50, 0.045, is_em=True, annual_inflation_base=0.60),
    ]
    z = carry_to_zscore(ests)
    assert "USDTRY" not in z
    assert set(z) == {"EURUSD", "AUDUSD"}


def test_zscore_needs_two_usable_estimates():
    with pytest.raises(ValueError, match="need >= 2"):
        carry_to_zscore([fx_carry("EURUSD", 0.02, 0.05)])


# ==========================================================================
# Sizing
# ==========================================================================

def test_realised_vol_annualises():
    r = RNG.normal(0, 0.01, 5000)
    assert realised_vol(r) == pytest.approx(0.01 * math.sqrt(252), rel=0.05)


def test_ewma_vol_reacts_faster_to_a_vol_spike():
    calm = RNG.normal(0, 0.005, 400)
    spike = RNG.normal(0, 0.04, 40)
    series = np.concatenate([calm, spike])
    assert realised_vol(series, halflife=20) > realised_vol(series)


def test_vol_targeting_equalises_risk_across_wildly_different_vols():
    """Bitcoin at 60% vol must not dominate EURUSD at 7%."""
    res = vol_target_weights(
        signals={"BTCUSD": 1.0, "EURUSD": 1.0},
        vols={"BTCUSD": 0.60, "EURUSD": 0.07},
        target_vol=0.10,
    )
    assert abs(res.weights["EURUSD"]) > abs(res.weights["BTCUSD"]) * 5


def test_vol_target_is_hit():
    vols = {"A": 0.20, "B": 0.15, "C": 0.35}
    res = vol_target_weights({"A": 1.0, "B": -1.0, "C": 0.5}, vols, target_vol=0.12)
    assert res.ex_ante_vol == pytest.approx(0.12, rel=1e-6)


def test_covariance_path_accounts_for_correlation():
    """A correlated book carries more risk than the zero-correlation assumption."""
    vols = {"A": 0.20, "B": 0.20}
    cov = np.array([[0.04, 0.036], [0.036, 0.04]])  # rho = 0.9
    naive = vol_target_weights({"A": 1.0, "B": 1.0}, vols, target_vol=0.10)
    with_cov = vol_target_weights({"A": 1.0, "B": 1.0}, vols, cov=cov, target_vol=0.10)
    assert with_cov.gross_leverage < naive.gross_leverage


def test_gross_leverage_cap_binds():
    res = vol_target_weights(
        {"A": 1.0, "B": 1.0}, {"A": 0.02, "B": 0.02},
        target_vol=0.30, max_gross_leverage=2.0,
    )
    assert res.binding_constraint == "max_gross_leverage"
    assert res.gross_leverage <= 2.0 + 1e-9


def test_missing_vol_is_an_error_not_a_default():
    with pytest.raises(ValueError, match="no volatility for"):
        vol_target_weights({"A": 1.0, "B": 1.0}, {"A": 0.2})


def test_risk_parity_equalises_risk_contributions():
    cov = np.array([[0.04, 0.006, 0.0], [0.006, 0.09, 0.0], [0.0, 0.0, 0.0025]])
    w = risk_parity_weights(cov)
    rc = w * (cov @ w) / portfolio_vol(w, cov)
    assert np.allclose(rc, rc[0], rtol=1e-4)
    assert w.sum() == pytest.approx(1.0)


def test_risk_parity_gives_least_weight_to_riskiest_asset():
    cov = np.diag([0.01, 0.04, 0.16])
    w = risk_parity_weights(cov)
    assert w[0] > w[1] > w[2]


def test_full_kelly_is_refused():
    """The 18.5%-on-one-trade failure mode is unreachable by construction."""
    with pytest.raises(ValueError, match="ruinous"):
        fractional_kelly(0.08, 0.20, fraction=1.0)


def test_kelly_cap_binds():
    assert fractional_kelly(0.50, 0.10, fraction=0.5, cap=0.20) == pytest.approx(0.20)


def test_quarter_kelly_is_a_quarter_of_full():
    mu, sigma = 0.06, 0.20
    assert fractional_kelly(mu, sigma, 0.25, cap=1.0) == pytest.approx(
        0.25 * mu / sigma**2
    )


def test_negative_edge_gives_short():
    assert fractional_kelly(-0.06, 0.20, 0.25, cap=1.0) < 0


# ==========================================================================
# Signals
# ==========================================================================

def test_tsmom_detects_direction():
    up = np.linspace(100, 150, 300)
    down = np.linspace(150, 100, 300)
    assert tsmom(up, 252) == 1.0
    assert tsmom(down, 252) == -1.0


def test_tsmom_needs_enough_history():
    with pytest.raises(ValueError, match="need >="):
        tsmom(np.linspace(100, 110, 50), lookback=252)


def test_blended_tsmom_is_continuous_when_horizons_disagree():
    # Long uptrend with a recent sharp reversal.
    p = np.concatenate([np.linspace(100, 200, 300), np.linspace(200, 150, 100)])
    v = blended_tsmom(p).value
    assert -1.0 < v < 1.0


def test_blended_tsmom_flags_insufficient_history():
    p = np.linspace(100, 130, 200)  # enough for 63/126, not 252
    assert blended_tsmom(p).confidence == "weak"


def test_vol_scaling_shrinks_high_vol_positions():
    assert vol_scaled_signal(1.0, 0.60) < vol_scaled_signal(1.0, 0.07)


def test_ppp_value_positive_when_base_is_cheap():
    assert fx_value_ppp("EURUSD", spot=1.00, ppp_rate=1.30).value > 0
    assert fx_value_ppp("EURUSD", spot=1.40, ppp_rate=1.20).value < 0


def test_em_ppp_marked_weak():
    assert fx_value_ppp("USDMXN", 18.0, 15.0, is_em=True).confidence == "weak"


def test_commodity_value_cheap_after_decline():
    p = np.concatenate([np.full(1000, 100.0), np.linspace(100, 60, 300)])
    assert commodity_value_reversal("GC", p).value > 0


def test_equity_value_depends_on_real_rate():
    cheap = equity_value_earnings_yield("SPY", 0.05, -0.01).value
    rich = equity_value_earnings_yield("SPY", 0.05, 0.03).value
    assert cheap > rich


def test_combine_averages_the_three_premia():
    from xasset.signals import Signal
    out = combine_signals([
        Signal("EURUSD", 1.0, "trend"),
        Signal("EURUSD", -1.0, "carry"),
        Signal("EURUSD", 0.0, "value"),
    ])
    assert out["EURUSD"] == pytest.approx(0.0, abs=1e-9)


def test_weak_signals_are_downweighted_not_dropped():
    from xasset.signals import Signal
    strong = combine_signals([
        Signal("X", 1.0, "trend"), Signal("X", -1.0, "value"),
    ])
    weak = combine_signals([
        Signal("X", 1.0, "trend"), Signal("X", -1.0, "value", "weak", "n/a"),
    ])
    assert weak["X"] > strong["X"]


def test_signal_without_symbol_rejected():
    from xasset.signals import Signal
    with pytest.raises(ValueError, match="no symbol"):
        combine_signals([Signal("", 1.0, "trend")])


def test_unknown_signal_kind_rejected():
    from xasset.signals import Signal
    with pytest.raises(ValueError, match="no weight given"):
        combine_signals([Signal("X", 1.0, "momentum_burst")])
