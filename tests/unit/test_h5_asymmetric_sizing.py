"""
Tests for asymmetric sizing helper in vol_targeting.py.

H=5 Track B uses:
    SHORT: full leverage (1.0x multiplier)
    LONG:  half leverage (0.5x multiplier)

Rationale: 2023 walk-forward showed LONG fragility when model
generates >50% LONG signals.
"""

import pytest
from src.forecasting.vol_targeting import (
    apply_asymmetric_sizing,
    compute_vol_target_signal,
    VolTargetConfig,
)


class TestAsymmetricSizing:
    def test_short_gets_full_leverage(self):
        """SHORT direction should get 1.0x multiplier."""
        result = apply_asymmetric_sizing(leverage=1.5, direction=-1)
        assert result == 1.5

    def test_long_gets_half_leverage(self):
        """LONG direction should get 0.5x multiplier."""
        result = apply_asymmetric_sizing(leverage=1.5, direction=1)
        assert result == 0.75

    def test_custom_multipliers(self):
        """Custom multipliers should override defaults."""
        result = apply_asymmetric_sizing(
            leverage=2.0, direction=1, long_mult=0.3, short_mult=0.8
        )
        assert abs(result - 0.6) < 1e-10  # 2.0 * 0.3

        result = apply_asymmetric_sizing(
            leverage=2.0, direction=-1, long_mult=0.3, short_mult=0.8
        )
        assert abs(result - 1.6) < 1e-10  # 2.0 * 0.8

    def test_zero_leverage(self):
        """Zero leverage should return zero."""
        assert apply_asymmetric_sizing(0.0, direction=1) == 0.0
        assert apply_asymmetric_sizing(0.0, direction=-1) == 0.0

    def test_min_leverage(self):
        """Min leverage (0.5) with LONG should give 0.25."""
        result = apply_asymmetric_sizing(0.5, direction=1)
        assert abs(result - 0.25) < 1e-10

    def test_max_leverage(self):
        """Max leverage (2.0) with SHORT should stay 2.0."""
        result = apply_asymmetric_sizing(2.0, direction=-1)
        assert result == 2.0

    def test_max_leverage_long(self):
        """Max leverage (2.0) with LONG should give 1.0."""
        result = apply_asymmetric_sizing(2.0, direction=1)
        assert abs(result - 1.0) < 1e-10

    def test_invalid_direction_raises(self):
        """Direction must be +1 or -1."""
        with pytest.raises(AssertionError):
            apply_asymmetric_sizing(1.0, direction=0)
        with pytest.raises(AssertionError):
            apply_asymmetric_sizing(1.0, direction=2)

    def test_negative_leverage_raises(self):
        """Leverage must be non-negative."""
        with pytest.raises(AssertionError):
            apply_asymmetric_sizing(-1.0, direction=1)


class TestVolTargetWithAsymmetric:
    """Integration: vol-targeting followed by asymmetric sizing."""

    def test_full_pipeline_short(self):
        """SHORT signal: vol-target -> asymmetric -> should be unchanged."""
        config = VolTargetConfig(target_vol=0.15, max_leverage=2.0, min_leverage=0.5)
        signal = compute_vol_target_signal(
            forecast_direction=-1,
            forecast_return=-0.005,
            realized_vol_21d=0.10,
            config=config,
        )
        # raw_leverage = 0.15 / 0.10 = 1.5
        assert abs(signal.clipped_leverage - 1.5) < 1e-6

        asymmetric = apply_asymmetric_sizing(signal.clipped_leverage, signal.forecast_direction)
        assert abs(asymmetric - 1.5) < 1e-6  # SHORT: full

    def test_full_pipeline_long(self):
        """LONG signal: vol-target -> asymmetric -> should be halved."""
        config = VolTargetConfig(target_vol=0.15, max_leverage=2.0, min_leverage=0.5)
        signal = compute_vol_target_signal(
            forecast_direction=1,
            forecast_return=0.005,
            realized_vol_21d=0.10,
            config=config,
        )
        # raw_leverage = 0.15 / 0.10 = 1.5
        assert abs(signal.clipped_leverage - 1.5) < 1e-6

        asymmetric = apply_asymmetric_sizing(signal.clipped_leverage, signal.forecast_direction)
        assert abs(asymmetric - 0.75) < 1e-6  # LONG: half

    def test_low_vol_short(self):
        """Low vol -> high leverage, clipped to 2.0, SHORT keeps full."""
        config = VolTargetConfig(target_vol=0.15, max_leverage=2.0, min_leverage=0.5)
        signal = compute_vol_target_signal(
            forecast_direction=-1,
            forecast_return=-0.01,
            realized_vol_21d=0.05,
            config=config,
        )
        # raw = 0.15/0.05 = 3.0, clipped to 2.0
        assert abs(signal.clipped_leverage - 2.0) < 1e-6

        asymmetric = apply_asymmetric_sizing(signal.clipped_leverage, -1)
        assert abs(asymmetric - 2.0) < 1e-6

    def test_low_vol_long(self):
        """Low vol -> high leverage clipped to 2.0, LONG halved to 1.0."""
        config = VolTargetConfig(target_vol=0.15, max_leverage=2.0, min_leverage=0.5)
        signal = compute_vol_target_signal(
            forecast_direction=1,
            forecast_return=0.01,
            realized_vol_21d=0.05,
            config=config,
        )
        assert abs(signal.clipped_leverage - 2.0) < 1e-6

        asymmetric = apply_asymmetric_sizing(signal.clipped_leverage, 1)
        assert abs(asymmetric - 1.0) < 1e-6
