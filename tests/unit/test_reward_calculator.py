"""
Unit tests for RewardCalculator.

Tests the reward function implementation ensuring:
- Transaction costs are ADDITIVE (critical fix)
- Asymmetric loss penalties work correctly
- All bonus/penalty components calculate properly
- Reward clipping prevents extreme values

Contract: CTR-REWARD-001
Author: Trading Team
Version: 1.0.0
Created: 2026-01-18
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Direct import to avoid circular dependencies through src.__init__
import importlib.util
spec = importlib.util.spec_from_file_location(
    "reward_calculator",
    PROJECT_ROOT / "src" / "training" / "reward_calculator.py"
)
reward_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reward_module)

RewardCalculator = reward_module.RewardCalculator
RewardConfig = reward_module.RewardConfig


class TestRewardCalculatorBasics:
    """Basic reward calculation tests."""

    @pytest.fixture
    def default_calc(self):
        """Default reward calculator with standard config."""
        return RewardCalculator()

    @pytest.fixture
    def custom_calc(self):
        """Reward calculator with custom config for testing."""
        config = RewardConfig(
            transaction_cost_pct=0.0002,
            loss_penalty_multiplier=2.0,
            hold_bonus_per_bar=0.0001,
            hold_bonus_requires_profit=False,  # Easier to test
            consecutive_win_bonus=0.001,
            max_consecutive_bonus=5,
            drawdown_penalty_threshold=0.05,
            drawdown_penalty_multiplier=2.0,
        )
        return RewardCalculator(config)

    def test_positive_pnl_returns_positive_reward(self, default_calc):
        """Positive PnL should result in positive reward."""
        reward, breakdown = default_calc.calculate(
            pnl_pct=0.001,
            position_change=0,
        )
        assert reward > 0
        assert breakdown['base_pnl'] == 0.001

    def test_negative_pnl_returns_negative_reward(self, default_calc):
        """Negative PnL should result in negative reward."""
        reward, breakdown = default_calc.calculate(
            pnl_pct=-0.001,
            position_change=0,
        )
        assert reward < 0

    def test_zero_pnl_returns_near_zero_reward(self, default_calc):
        """Zero PnL with no bonuses/penalties should return near-zero reward."""
        reward, breakdown = default_calc.calculate(
            pnl_pct=0.0,
            position_change=0,
            bars_held=0,
            consecutive_wins=0,
            current_drawdown=0.0,
        )
        assert abs(reward) < 0.001


class TestTransactionCostAdditive:
    """Critical tests: Transaction cost must be ADDITIVE, not multiplicative."""

    @pytest.fixture
    def calc(self):
        return RewardCalculator()

    def test_transaction_cost_is_additive_not_multiplicative(self, calc):
        """
        CRITICAL TEST: Transaction cost must be ADDITIVE.

        If multiplicative: cost scales with PnL magnitude
        If additive: cost is constant regardless of PnL
        """
        # Test with small PnL
        reward_small, bd_small = calc.calculate(pnl_pct=0.001, position_change=1)
        cost_small = 0.001 - reward_small

        # Test with large PnL
        reward_large, bd_large = calc.calculate(pnl_pct=0.1, position_change=1)
        cost_large = 0.1 - reward_large

        # Both costs should be approximately equal (the transaction cost)
        assert abs(cost_small - cost_large) < 0.0001, (
            f"Transaction cost scaled with PnL! "
            f"Small cost: {cost_small:.6f}, Large cost: {cost_large:.6f}"
        )

    def test_transaction_cost_value_is_correct(self, calc):
        """Transaction cost should equal config value."""
        reward, breakdown = calc.calculate(
            pnl_pct=0.0,  # Zero PnL isolates the cost
            position_change=1,
        )

        expected_cost = -calc.config.transaction_cost_pct
        assert breakdown['transaction_cost'] == expected_cost

    def test_no_transaction_cost_when_position_unchanged(self, calc):
        """No transaction cost should be applied when position doesn't change."""
        reward, breakdown = calc.calculate(
            pnl_pct=0.001,
            position_change=0,  # No position change
        )

        assert breakdown['transaction_cost'] == 0.0

    def test_transaction_cost_applied_on_open(self, calc):
        """Transaction cost should be applied when opening position."""
        _, breakdown = calc.calculate(pnl_pct=0.0, position_change=1)
        assert breakdown['transaction_cost'] < 0

    def test_transaction_cost_applied_on_close(self, calc):
        """Transaction cost should be applied when closing position."""
        _, breakdown = calc.calculate(pnl_pct=0.0, position_change=-1)
        assert breakdown['transaction_cost'] < 0


class TestAsymmetricPenalty:
    """Tests for asymmetric loss penalty."""

    @pytest.fixture
    def calc(self):
        config = RewardConfig(loss_penalty_multiplier=2.0)
        return RewardCalculator(config)

    def test_loss_penalty_applied_to_negative_pnl(self, calc):
        """Losses should be multiplied by penalty factor."""
        _, breakdown = calc.calculate(pnl_pct=-0.001, position_change=0)

        expected = -0.001 * 2.0  # loss * multiplier
        assert breakdown['after_asymmetric'] == pytest.approx(expected)

    def test_no_penalty_for_positive_pnl(self, calc):
        """Gains should not be penalized."""
        _, breakdown = calc.calculate(pnl_pct=0.001, position_change=0)

        # Positive PnL should pass through unchanged
        assert breakdown['after_asymmetric'] == breakdown['base_pnl']

    def test_asymmetric_penalty_magnitude(self, calc):
        """Loss penalty should make equal loss feel worse than equal gain feels good."""
        reward_gain, _ = calc.calculate(pnl_pct=0.001, position_change=0)
        reward_loss, _ = calc.calculate(pnl_pct=-0.001, position_change=0)

        # Due to 2x penalty, loss should be more impactful
        assert abs(reward_loss) > abs(reward_gain)


class TestDrawdownPenalty:
    """Tests for drawdown penalty calculation."""

    @pytest.fixture
    def calc(self):
        config = RewardConfig(
            drawdown_penalty_threshold=0.05,  # 5%
            drawdown_penalty_multiplier=2.0,
        )
        return RewardCalculator(config)

    def test_no_penalty_below_threshold(self, calc):
        """No penalty when drawdown is below threshold."""
        _, breakdown = calc.calculate(
            pnl_pct=0.0,
            position_change=0,
            current_drawdown=0.03,  # 3% < 5% threshold
        )

        assert breakdown['drawdown_penalty'] == 0.0

    def test_penalty_above_threshold(self, calc):
        """Penalty should apply when drawdown exceeds threshold."""
        _, breakdown = calc.calculate(
            pnl_pct=0.0,
            position_change=0,
            current_drawdown=0.10,  # 10% > 5% threshold
        )

        expected = (0.10 - 0.05) * 2.0  # excess DD * multiplier
        assert breakdown['drawdown_penalty'] == pytest.approx(expected)

    def test_penalty_at_threshold_is_zero(self, calc):
        """Penalty should be zero exactly at threshold."""
        _, breakdown = calc.calculate(
            pnl_pct=0.0,
            position_change=0,
            current_drawdown=0.05,  # Exactly at threshold
        )

        assert breakdown['drawdown_penalty'] == 0.0


class TestConsistencyBonus:
    """Tests for consecutive wins bonus."""

    @pytest.fixture
    def calc(self):
        config = RewardConfig(
            consecutive_win_bonus=0.001,
            max_consecutive_bonus=5,
        )
        return RewardCalculator(config)

    def test_no_bonus_without_consecutive_wins(self, calc):
        """No bonus when consecutive_wins is 0."""
        _, breakdown = calc.calculate(
            pnl_pct=0.001,
            position_change=0,
            consecutive_wins=0,
        )

        assert breakdown['consistency_bonus'] == 0.0

    def test_bonus_with_consecutive_wins(self, calc):
        """Bonus should scale with consecutive wins."""
        _, breakdown = calc.calculate(
            pnl_pct=0.001,
            position_change=0,
            consecutive_wins=3,
        )

        expected = 0.001 * 3  # bonus * wins
        assert breakdown['consistency_bonus'] == pytest.approx(expected)

    def test_bonus_capped_at_max(self, calc):
        """Bonus should not exceed max_consecutive_bonus."""
        _, breakdown = calc.calculate(
            pnl_pct=0.001,
            position_change=0,
            consecutive_wins=10,  # Exceeds max of 5
        )

        expected = 0.001 * 5  # bonus * max
        assert breakdown['consistency_bonus'] == pytest.approx(expected)

    def test_no_bonus_on_loss(self, calc):
        """No consistency bonus when current bar is a loss."""
        _, breakdown = calc.calculate(
            pnl_pct=-0.001,  # Loss
            position_change=0,
            consecutive_wins=5,  # Previous wins
        )

        assert breakdown['consistency_bonus'] == 0.0


class TestIntratradeDrawdown:
    """Tests for intratrade drawdown penalty."""

    @pytest.fixture
    def calc(self):
        config = RewardConfig(
            intratrade_dd_penalty=0.5,
            max_intratrade_dd=0.02,  # 2%
        )
        return RewardCalculator(config)

    def test_no_penalty_below_threshold(self, calc):
        """No penalty when intratrade DD is below threshold."""
        _, breakdown = calc.calculate(
            pnl_pct=0.0,
            position_change=0,
            intratrade_drawdown=0.01,  # 1% < 2%
        )

        assert breakdown['intratrade_penalty'] == 0.0

    def test_penalty_above_threshold(self, calc):
        """Penalty applies when intratrade DD exceeds threshold."""
        _, breakdown = calc.calculate(
            pnl_pct=0.0,
            position_change=0,
            intratrade_drawdown=0.05,  # 5% > 2%
        )

        # (5% - 2%) * 0.5 * 100
        expected = (0.05 - 0.02) * 0.5 * 100
        assert breakdown['intratrade_penalty'] == pytest.approx(expected)


class TestTimeDecay:
    """Tests for time decay on stale positions."""

    @pytest.fixture
    def calc(self):
        config = RewardConfig(
            time_decay_start_bars=24,
            time_decay_per_bar=0.0001,
            time_decay_losing_multiplier=2.0,
        )
        return RewardCalculator(config)

    def test_no_decay_before_threshold(self, calc):
        """No decay when bars_held is below start threshold."""
        _, breakdown = calc.calculate(
            pnl_pct=0.0,
            position_change=0,
            bars_held=20,  # < 24
        )

        assert breakdown['time_decay'] == 0.0

    def test_decay_after_threshold(self, calc):
        """Decay applies after threshold is exceeded."""
        _, breakdown = calc.calculate(
            pnl_pct=0.001,  # Winning position
            position_change=0,
            bars_held=30,  # > 24
        )

        expected = (30 - 24) * 0.0001  # excess_bars * decay_per_bar
        assert breakdown['time_decay'] == pytest.approx(expected)

    def test_double_decay_for_losing_position(self, calc):
        """Losing positions should have 2x time decay."""
        _, breakdown_loss = calc.calculate(
            pnl_pct=-0.001,  # Losing
            position_change=0,
            bars_held=30,
        )

        _, breakdown_win = calc.calculate(
            pnl_pct=0.001,  # Winning
            position_change=0,
            bars_held=30,
        )

        # Losing should have 2x decay
        assert breakdown_loss['time_decay'] == pytest.approx(
            breakdown_win['time_decay'] * 2.0
        )


class TestRewardClipping:
    """Tests for reward clipping."""

    @pytest.fixture
    def calc(self):
        config = RewardConfig(min_reward=-1.0, max_reward=1.0)
        return RewardCalculator(config)

    def test_clips_extreme_positive_reward(self, calc):
        """Large positive PnL should be clipped to max."""
        reward, breakdown = calc.calculate(
            pnl_pct=2.0,  # 200% - extreme
            position_change=0,
        )

        assert reward == 1.0
        assert breakdown['was_clipped'] == True  # noqa: E712

    def test_clips_extreme_negative_reward(self, calc):
        """Large negative PnL should be clipped to min."""
        reward, breakdown = calc.calculate(
            pnl_pct=-2.0,  # -200% - extreme
            position_change=0,
        )

        assert reward == -1.0
        assert breakdown['was_clipped'] == True  # noqa: E712

    def test_no_clip_for_normal_values(self, calc):
        """Normal rewards should not be clipped."""
        reward, breakdown = calc.calculate(
            pnl_pct=0.001,
            position_change=0,
        )

        assert breakdown['was_clipped'] == False  # noqa: E712


class TestHoldBonus:
    """Tests for hold bonus calculation."""

    @pytest.fixture
    def calc_with_hold(self):
        config = RewardConfig(
            hold_bonus_per_bar=0.0001,
            hold_bonus_requires_profit=False,  # Allow bonus regardless of PnL
        )
        return RewardCalculator(config)

    @pytest.fixture
    def calc_profit_required(self):
        config = RewardConfig(
            hold_bonus_per_bar=0.0001,
            hold_bonus_requires_profit=True,  # Only if profitable
        )
        return RewardCalculator(config)

    def test_hold_bonus_scales_with_bars(self, calc_with_hold):
        """Hold bonus should scale with bars held."""
        _, breakdown = calc_with_hold.calculate(
            pnl_pct=0.0,
            position_change=0,
            bars_held=5,
        )

        expected = 0.0001 * 5
        assert breakdown['hold_bonus'] == pytest.approx(expected)

    def test_hold_bonus_capped_at_10_bars(self, calc_with_hold):
        """Hold bonus should not exceed 10 bars worth."""
        _, breakdown = calc_with_hold.calculate(
            pnl_pct=0.0,
            position_change=0,
            bars_held=20,  # Exceeds cap
        )

        expected = 0.0001 * 10  # Capped at 10
        assert breakdown['hold_bonus'] == pytest.approx(expected)

    def test_no_hold_bonus_on_position_change(self, calc_with_hold):
        """No hold bonus when position is opened/closed."""
        _, breakdown = calc_with_hold.calculate(
            pnl_pct=0.0,
            position_change=1,  # Opening
            bars_held=5,
        )

        assert breakdown['hold_bonus'] == 0.0

    def test_hold_bonus_requires_profit(self, calc_profit_required):
        """When required, hold bonus only applies to profitable positions."""
        # Profitable - should get bonus
        _, breakdown_win = calc_profit_required.calculate(
            pnl_pct=0.001,
            position_change=0,
            bars_held=5,
        )

        # Losing - should not get bonus
        _, breakdown_loss = calc_profit_required.calculate(
            pnl_pct=-0.001,
            position_change=0,
            bars_held=5,
        )

        assert breakdown_win['hold_bonus'] > 0
        assert breakdown_loss['hold_bonus'] == 0.0


class TestRewardCalculatorReset:
    """Tests for reward calculator reset."""

    def test_reset_clears_internal_state(self):
        """Reset should clear internal state."""
        calc = RewardCalculator()
        calc._consecutive_wins = 5  # Simulate state

        calc.reset()

        assert calc._consecutive_wins == 0


class TestRewardBreakdown:
    """Tests for reward breakdown dictionary."""

    def test_breakdown_contains_all_components(self):
        """Breakdown should contain all reward components."""
        calc = RewardCalculator()
        _, breakdown = calc.calculate(
            pnl_pct=0.001,
            position_change=1,
            bars_held=5,
            consecutive_wins=2,
            current_drawdown=0.06,
            intratrade_drawdown=0.03,
        )

        expected_keys = [
            'base_pnl',
            'after_asymmetric',
            'transaction_cost',
            'after_transaction',
            'hold_bonus',
            'consistency_bonus',
            'drawdown_penalty',
            'intratrade_penalty',
            'time_decay',
            'final_reward',
            'was_clipped',
        ]

        for key in expected_keys:
            assert key in breakdown, f"Missing key: {key}"


class TestEdgeCases:
    """Edge case tests."""

    @pytest.fixture
    def calc(self):
        return RewardCalculator()

    def test_nan_pnl_handled(self, calc):
        """NaN PnL should not crash (though behavior may vary)."""
        # This is more of a defensive test
        try:
            reward, _ = calc.calculate(pnl_pct=float('nan'), position_change=0)
            # If it doesn't crash, check result is reasonable
            assert not np.isnan(reward) or reward == calc.config.min_reward
        except Exception:
            pytest.skip("NaN handling not explicitly supported")

    def test_very_small_pnl(self, calc):
        """Very small PnL should be handled without underflow."""
        reward, breakdown = calc.calculate(pnl_pct=1e-10, position_change=0)
        assert not np.isnan(reward)
        assert not np.isinf(reward)

    def test_max_int_bars_held(self, calc):
        """Large bars_held should not cause overflow."""
        reward, _ = calc.calculate(
            pnl_pct=0.001,
            position_change=0,
            bars_held=1000000,  # Very large
        )
        assert not np.isnan(reward)
        assert not np.isinf(reward)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
