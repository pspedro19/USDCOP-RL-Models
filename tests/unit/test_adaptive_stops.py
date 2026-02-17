"""Tests for adaptive_stops — volatility-scaled TP and hard-stop levels."""

import math
import pytest
from src.forecasting.adaptive_stops import (
    AdaptiveStops,
    AdaptiveStopsConfig,
    compute_adaptive_stops,
    check_hard_stop,
    check_take_profit,
    get_exit_price,
)


@pytest.fixture
def config():
    return AdaptiveStopsConfig()


class TestAdaptiveStopsComputation:
    def test_moderate_vol(self, config):
        """vol_ann=10% → daily ~0.63% → weekly ~1.41% → HS=2.11%, TP=1.06%."""
        stops = compute_adaptive_stops(0.10, config)
        expected_daily = 0.10 / math.sqrt(252)
        expected_weekly = expected_daily * math.sqrt(5)
        expected_hs = expected_weekly * 1.5

        assert abs(stops.realized_vol_daily - expected_daily) < 1e-8
        assert abs(stops.realized_vol_weekly - expected_weekly) < 1e-8
        assert abs(stops.hard_stop_pct - expected_hs) < 1e-8
        assert abs(stops.take_profit_pct - expected_hs * 0.5) < 1e-8

    def test_low_vol_clamp(self, config):
        """vol_ann=5% → raw HS ~1.06% → clamped to 1.06% (above 1% floor)."""
        stops = compute_adaptive_stops(0.05, config)
        assert stops.hard_stop_pct >= config.hard_stop_min_pct
        # 5% ann → daily 0.315% → weekly 0.704% → HS = 1.056%
        assert stops.hard_stop_pct > 0.01  # Above minimum

    def test_very_low_vol_hits_floor(self, config):
        """vol_ann=2% → raw HS ~0.42% → clamped to 1% floor."""
        stops = compute_adaptive_stops(0.02, config)
        assert abs(stops.hard_stop_pct - config.hard_stop_min_pct) < 1e-8
        assert abs(stops.take_profit_pct - config.hard_stop_min_pct * 0.5) < 1e-8

    def test_high_vol_clamp(self, config):
        """vol_ann=20% → raw HS ~4.22% → clamped to 3% cap."""
        stops = compute_adaptive_stops(0.20, config)
        assert abs(stops.hard_stop_pct - config.hard_stop_max_pct) < 1e-8
        assert abs(stops.take_profit_pct - config.hard_stop_max_pct * 0.5) < 1e-8

    def test_very_high_vol_capped(self, config):
        """vol_ann=30% → clamped to 3%."""
        stops = compute_adaptive_stops(0.30, config)
        assert abs(stops.hard_stop_pct - 0.03) < 1e-8

    def test_tp_always_half_of_hs(self, config):
        """TP = HS * 0.5 for any volatility level."""
        for vol in [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
            stops = compute_adaptive_stops(vol, config)
            assert abs(stops.take_profit_pct - stops.hard_stop_pct * 0.5) < 1e-10


class TestHardStopCheck:
    def test_long_hard_stop_triggered(self):
        """LONG: bar_low below stop level."""
        triggered = check_hard_stop(
            direction=1, entry_price=4200.0,
            bar_high=4210.0, bar_low=4110.0,  # low = 4110, stop at 4200*(1-0.02) = 4116
            hard_stop_pct=0.02,
        )
        assert triggered is True

    def test_long_hard_stop_not_triggered(self):
        """LONG: bar_low above stop level."""
        triggered = check_hard_stop(
            direction=1, entry_price=4200.0,
            bar_high=4210.0, bar_low=4150.0,  # low = 4150 > 4116
            hard_stop_pct=0.02,
        )
        assert triggered is False

    def test_short_hard_stop_triggered(self):
        """SHORT: bar_high above stop level."""
        triggered = check_hard_stop(
            direction=-1, entry_price=4200.0,
            bar_high=4290.0, bar_low=4190.0,  # high = 4290, stop at 4200*(1+0.02) = 4284
            hard_stop_pct=0.02,
        )
        assert triggered is True

    def test_short_hard_stop_not_triggered(self):
        """SHORT: bar_high below stop level."""
        triggered = check_hard_stop(
            direction=-1, entry_price=4200.0,
            bar_high=4250.0, bar_low=4190.0,  # high = 4250 < 4284
            hard_stop_pct=0.02,
        )
        assert triggered is False

    def test_exact_stop_level(self):
        """Exactly at stop → triggered."""
        triggered = check_hard_stop(
            direction=1, entry_price=4200.0,
            bar_high=4210.0, bar_low=4116.0,  # 4200*(1-0.02) = 4116
            hard_stop_pct=0.02,
        )
        assert triggered is True


class TestTakeProfitCheck:
    def test_long_tp_triggered(self):
        """LONG: bar_high above TP level."""
        triggered = check_take_profit(
            direction=1, entry_price=4200.0,
            bar_high=4250.0, bar_low=4195.0,  # high = 4250, TP at 4200*(1+0.01) = 4242
            take_profit_pct=0.01,
        )
        assert triggered is True

    def test_long_tp_not_triggered(self):
        """LONG: bar_high below TP level."""
        triggered = check_take_profit(
            direction=1, entry_price=4200.0,
            bar_high=4230.0, bar_low=4195.0,  # high = 4230 < 4242
            take_profit_pct=0.01,
        )
        assert triggered is False

    def test_short_tp_triggered(self):
        """SHORT: bar_low below TP level."""
        triggered = check_take_profit(
            direction=-1, entry_price=4200.0,
            bar_high=4205.0, bar_low=4140.0,  # low = 4140, TP at 4200*(1-0.01) = 4158
            take_profit_pct=0.01,
        )
        assert triggered is True

    def test_short_tp_not_triggered(self):
        """SHORT: bar_low above TP level."""
        triggered = check_take_profit(
            direction=-1, entry_price=4200.0,
            bar_high=4205.0, bar_low=4170.0,  # low = 4170 > 4158
            take_profit_pct=0.01,
        )
        assert triggered is False


class TestExitPrice:
    def test_tp_long_exit_price(self):
        """LONG TP: exact limit order price."""
        price = get_exit_price(1, 4200.0, "take_profit", 0.02, 0.01, 4250.0)
        assert abs(price - 4200.0 * 1.01) < 1e-6

    def test_tp_short_exit_price(self):
        """SHORT TP: exact limit order price."""
        price = get_exit_price(-1, 4200.0, "take_profit", 0.02, 0.01, 4150.0)
        assert abs(price - 4200.0 * 0.99) < 1e-6

    def test_hs_long_exit_price(self):
        """LONG HS: exact limit order price."""
        price = get_exit_price(1, 4200.0, "hard_stop", 0.02, 0.01, 4100.0)
        assert abs(price - 4200.0 * 0.98) < 1e-6

    def test_hs_short_exit_price(self):
        """SHORT HS: exact limit order price."""
        price = get_exit_price(-1, 4200.0, "hard_stop", 0.02, 0.01, 4300.0)
        assert abs(price - 4200.0 * 1.02) < 1e-6

    def test_week_end_exit_price(self):
        """Week end: market order at bar_close."""
        price = get_exit_price(1, 4200.0, "week_end", 0.02, 0.01, 4215.0)
        assert abs(price - 4215.0) < 1e-6


class TestCustomConfig:
    def test_aggressive_stops(self):
        """Wider stop range: 2-5%."""
        config = AdaptiveStopsConfig(
            vol_multiplier=2.0,
            hard_stop_min_pct=0.02,
            hard_stop_max_pct=0.05,
            tp_ratio=0.3,
        )
        stops = compute_adaptive_stops(0.15, config)
        assert stops.hard_stop_pct >= 0.02
        assert stops.hard_stop_pct <= 0.05
        assert abs(stops.take_profit_pct - stops.hard_stop_pct * 0.3) < 1e-10
