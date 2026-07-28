"""BL-07 timing attribution: deterministic math and honest unavailable states."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from scripts.analysis import profitability_adapters
from scripts.analysis.profitability_types import Sleeve
from scripts.analysis.timing_ratio_oneoff import (
    analyse_sleeve,
    block_bootstrap_ci,
    build_report,
)


def _sleeve(
    strategy_id: str,
    *,
    invalid_baselines: tuple[str, ...] = (),
    n_trades: int = 20,
) -> SimpleNamespace:
    return SimpleNamespace(
        strategy_id=strategy_id,
        asset="test_asset",
        position=np.asarray([0.0, 0.5, 1.0, 0.25, 0.75], dtype=float),
        asset_ret=np.asarray([0.01, -0.02, 0.03, 0.01, -0.01], dtype=float),
        index=np.asarray(
            [
                "2026-01-01",
                "2026-01-02",
                "2026-01-03",
                "2026-01-04",
                "2026-01-05",
            ]
        ),
        clock_label="daily/252",
        n_trades=n_trades,
        invalid_baselines=invalid_baselines,
    )


def _analyse(sleeve: SimpleNamespace) -> dict:
    return analyse_sleeve(
        sleeve,
        expected_strategy_id=sleeve.strategy_id,
        adapter_key="test",
        role="champion",
        block_size=2,
        bootstrap_samples=50,
        seed=42,
    )


def test_invalid_underlying_return_is_unavailable_not_a_numeric_proxy() -> None:
    row = _analyse(
        _sleeve(
            "smart_simple_v11",
            invalid_baselines=("B1_buy_and_hold", "B1_prime_exposure_matched"),
        )
    )

    assert row["attribution_status"] == "UNAVAILABLE"
    assert "timing_ratio" not in row
    assert "timing_ratio_ci" not in row
    assert "underlying" in row["unavailable_reason"].lower()


def test_available_attribution_satisfies_both_exact_identities() -> None:
    row = _analyse(_sleeve("gold_trend_simple"))

    assert row["attribution_status"] == "AVAILABLE"
    assert abs(row["gross_equals_beta_plus_timing_residual"]) < 1e-12
    assert abs(row["timing_sum_equals_n_covariance_residual"]) < 1e-12
    assert row["timing_ratio_ci"]["valid_samples"] == 50


def test_small_trade_sample_exposes_only_counts_and_pnl() -> None:
    row = _analyse(_sleeve("btc_hodl_b1", n_trades=1))

    assert row["attribution_status"] == "INSUFFICIENT_TRADES"
    assert row["n_trades"] == 1
    assert row["min_trades_for_stats"] == 20
    assert "sum_gross_pnl" in row
    assert "sum_pnl_beta" in row
    assert "sum_pnl_timing" in row
    assert "timing_ratio" not in row
    assert "timing_ratio_ci" not in row
    assert "population_covariance_weight_return" not in row


def test_block_bootstrap_is_reproducible() -> None:
    sleeve = _sleeve("gold_trend_simple")
    first = block_bootstrap_ci(
        sleeve.position,
        sleeve.asset_ret,
        block_size=2,
        bootstrap_samples=100,
        seed=42,
    )
    second = block_bootstrap_ci(
        sleeve.position,
        sleeve.asset_ret,
        block_size=2,
        bootstrap_samples=100,
        seed=42,
    )

    assert first == second


def test_report_keeps_four_champions_and_is_strict_json() -> None:
    adapters = {
        "usdcop": lambda: _sleeve(
            "smart_simple_v11",
            invalid_baselines=("B1_buy_and_hold",),
        ),
        "gold_trend_simple": lambda: _sleeve("gold_trend_simple"),
        "btc_hodl_b1": lambda: _sleeve("btc_hodl_b1"),
        "spx500_regime_gated_v1": lambda: _sleeve(
            "spx500_regime_gated_v1"
        ),
    }

    report = build_report(
        adapters,
        include_gold_control=False,
        block_size=2,
        bootstrap_samples=25,
        seed=42,
    )

    rows = report["results"]
    assert [row["strategy_id"] for row in rows] == report["champions_required"]
    assert rows[0]["attribution_status"] == "UNAVAILABLE"
    assert report["persisted"] is False
    json.dumps(report, allow_nan=False)


def test_gold_control_is_appended_and_zero_is_inside_its_interval() -> None:
    control = _sleeve("gold_dynamic_exit")
    control.position = np.full_like(control.position, 0.5)
    adapters = {
        "gold_trend_simple": lambda: _sleeve("gold_trend_simple"),
        "btc_hodl_b1": lambda: _sleeve("btc_hodl_b1"),
        "spx500_regime_gated_v1": lambda: _sleeve(
            "spx500_regime_gated_v1"
        ),
        "xauusd": lambda: control,
    }

    report = build_report(
        adapters,
        include_gold_control=True,
        block_size=2,
        bootstrap_samples=25,
        seed=42,
    )

    assert [row["strategy_id"] for row in report["results"][:4]] == (
        report["champions_required"]
    )
    gold = report["results"][-1]
    assert gold["strategy_id"] == "gold_dynamic_exit"
    assert gold["role"] == "manual_control"
    assert gold["timing_ratio_ci"]["lower"] <= 0.0
    assert gold["timing_ratio_ci"]["upper"] >= 0.0


def test_dependency_light_sleeve_keeps_adapter_identity_and_net_return() -> None:
    sleeve = Sleeve(
        asset="test",
        strategy_id="test",
        index=[0, 1],
        position=[1.0, 2.0],
        asset_ret=[0.1, -0.2],
        cost=[0.01, 0.02],
        swap=None,
        n_trades=2,
        clock=252,
        clock_label="daily/252",
        dumb_name="buy_and_hold",
        dumb_position=[1.0, 1.0],
    )

    assert profitability_adapters.Sleeve is Sleeve
    np.testing.assert_allclose(sleeve.strat_ret, [0.09, -0.42])
