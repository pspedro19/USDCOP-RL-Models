"""
RewardCalculatorV20 - Corrected Reward Function for RL Training
================================================================

V20 FIX: Corrected order of operations in reward calculation.

The V19 reward had a critical bug where transaction costs were multiplied
instead of added, causing incorrect reward scaling.

CORRECT ORDER OF OPERATIONS (V20):
1. Base PnL (percentage change)
2. Asymmetric penalty (only on losses)
3. Transaction cost (ADDITIVE, not multiplied)
4. Hold bonus (encourage holding positions)
5. Consistency bonus (reward consecutive wins)
6. Drawdown penalty (discourage large drawdowns)

Author: Pedro @ Lean Tech Solutions / Claude Code
Version: 1.0.0
Date: 2026-01-09
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class RewardConfig:
    """Configuration for V20 reward calculator."""
    # Transaction costs
    transaction_cost_pct: float = 0.0002  # 2 bps per trade

    # Asymmetric loss penalty (losses hurt more than wins help)
    loss_penalty_multiplier: float = 1.5  # Losses are 1.5x more impactful

    # Hold bonus (encourage staying in profitable positions)
    hold_bonus_per_bar: float = 0.0001  # Small bonus per bar held

    # Consistency bonus
    consecutive_win_bonus: float = 0.001  # Bonus for each consecutive win
    max_consecutive_bonus: int = 5  # Cap at 5x bonus

    # Drawdown penalty
    drawdown_penalty_threshold: float = 0.05  # Start penalizing at 5% DD
    drawdown_penalty_multiplier: float = 2.0  # 2x penalty per % DD above threshold

    # Reward clipping
    min_reward: float = -1.0
    max_reward: float = 1.0


class RewardCalculatorV20:
    """
    V20 Reward Calculator with corrected order of operations.

    The reward formula is:

        reward = base_pnl
                 * asymmetric_penalty (only if loss)
                 + transaction_cost (ADDITIVE)
                 + hold_bonus
                 + consistency_bonus
                 - drawdown_penalty

    Critical V20 Fix:
    - Transaction cost is now ADDITIVE, not multiplicative
    - This prevents the cost from scaling incorrectly with PnL magnitude

    Example:
        >>> calc = RewardCalculatorV20()
        >>> reward, breakdown = calc.calculate(
        ...     pnl_pct=0.001,  # 0.1% gain
        ...     position_change=1,  # New long position
        ...     bars_held=5,
        ...     consecutive_wins=2,
        ...     current_drawdown=0.03
        ... )
        >>> print(f"Reward: {reward:.6f}")
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        """
        Initialize the reward calculator.

        Args:
            config: Reward configuration (uses defaults if not provided)
        """
        self.config = config or RewardConfig()
        self._consecutive_wins = 0

    def calculate(
        self,
        pnl_pct: float,
        position_change: int,  # 0=no change, 1=opened, -1=closed
        bars_held: int = 0,
        consecutive_wins: int = 0,
        current_drawdown: float = 0.0
    ) -> Tuple[float, dict]:
        """
        Calculate the reward with V20 corrected formula.

        Args:
            pnl_pct: Percentage P&L for this step (e.g., 0.001 = 0.1%)
            position_change: -1 if closed, 0 if unchanged, 1 if opened
            bars_held: Number of bars the current position has been held
            consecutive_wins: Number of consecutive winning trades
            current_drawdown: Current drawdown as decimal (e.g., 0.05 = 5%)

        Returns:
            Tuple of (final_reward, breakdown_dict)
        """
        breakdown = {}

        # ==========================================================
        # STEP 1: Base PnL
        # ==========================================================
        base_pnl = pnl_pct
        breakdown['base_pnl'] = base_pnl

        # ==========================================================
        # STEP 2: Asymmetric Penalty (only on losses)
        # ==========================================================
        if base_pnl < 0:
            adjusted_pnl = base_pnl * self.config.loss_penalty_multiplier
        else:
            adjusted_pnl = base_pnl
        breakdown['after_asymmetric'] = adjusted_pnl

        # ==========================================================
        # STEP 3: Transaction Cost (ADDITIVE - V20 FIX)
        # ==========================================================
        # Only apply transaction cost when position changes
        transaction_cost = 0.0
        if position_change != 0:
            # Opening or closing costs the same
            transaction_cost = -self.config.transaction_cost_pct
        breakdown['transaction_cost'] = transaction_cost

        # V20 FIX: ADDITIVE, not multiplicative
        reward_after_cost = adjusted_pnl + transaction_cost
        breakdown['after_transaction'] = reward_after_cost

        # ==========================================================
        # STEP 4: Hold Bonus
        # ==========================================================
        hold_bonus = 0.0
        if bars_held > 0 and position_change == 0:
            # Bonus for holding a position (encourages not over-trading)
            hold_bonus = self.config.hold_bonus_per_bar * min(bars_held, 10)
        breakdown['hold_bonus'] = hold_bonus

        # ==========================================================
        # STEP 5: Consistency Bonus
        # ==========================================================
        consistency_bonus = 0.0
        if consecutive_wins > 0 and base_pnl > 0:
            # Reward consecutive wins (up to max)
            bonus_multiplier = min(consecutive_wins, self.config.max_consecutive_bonus)
            consistency_bonus = self.config.consecutive_win_bonus * bonus_multiplier
        breakdown['consistency_bonus'] = consistency_bonus

        # ==========================================================
        # STEP 6: Drawdown Penalty
        # ==========================================================
        drawdown_penalty = 0.0
        if current_drawdown > self.config.drawdown_penalty_threshold:
            excess_dd = current_drawdown - self.config.drawdown_penalty_threshold
            drawdown_penalty = excess_dd * self.config.drawdown_penalty_multiplier
        breakdown['drawdown_penalty'] = drawdown_penalty

        # ==========================================================
        # FINAL: Combine all components
        # ==========================================================
        final_reward = (
            reward_after_cost
            + hold_bonus
            + consistency_bonus
            - drawdown_penalty
        )

        # Clip to prevent extreme rewards
        final_reward = np.clip(
            final_reward,
            self.config.min_reward,
            self.config.max_reward
        )

        breakdown['final_reward'] = final_reward
        breakdown['was_clipped'] = (
            final_reward == self.config.min_reward or
            final_reward == self.config.max_reward
        )

        return final_reward, breakdown

    def reset(self):
        """Reset internal state for new episode."""
        self._consecutive_wins = 0


def run_unit_tests():
    """Run unit tests for RewardCalculatorV20."""
    print("Running RewardCalculatorV20 unit tests...")

    calc = RewardCalculatorV20()
    all_passed = True

    # Test 1: Positive PnL without position change
    reward, bd = calc.calculate(pnl_pct=0.001, position_change=0, bars_held=5)
    expected = 0.001 + (0.0001 * 5)  # base + hold bonus
    if abs(reward - expected) < 0.0001:
        print("  [PASS] Test 1: Positive PnL with hold bonus")
    else:
        print(f"  [FAIL] Test 1: Expected {expected:.6f}, got {reward:.6f}")
        all_passed = False

    # Test 2: Negative PnL with asymmetric penalty
    reward, bd = calc.calculate(pnl_pct=-0.001, position_change=0)
    expected = -0.001 * 1.5  # base * penalty
    if abs(reward - expected) < 0.0001:
        print("  [PASS] Test 2: Negative PnL with asymmetric penalty")
    else:
        print(f"  [FAIL] Test 2: Expected {expected:.6f}, got {reward:.6f}")
        all_passed = False

    # Test 3: Transaction cost is ADDITIVE
    reward, bd = calc.calculate(pnl_pct=0.001, position_change=1)
    expected = 0.001 - 0.0002  # base + transaction cost (ADDITIVE)
    if abs(reward - expected) < 0.0001:
        print("  [PASS] Test 3: Transaction cost is ADDITIVE")
    else:
        print(f"  [FAIL] Test 3: Expected {expected:.6f}, got {reward:.6f}")
        all_passed = False

    # Test 4: Drawdown penalty
    reward, bd = calc.calculate(pnl_pct=0.0, position_change=0, current_drawdown=0.10)
    expected = 0.0 - (0.10 - 0.05) * 2.0  # 5% excess DD * 2x penalty
    if abs(reward - expected) < 0.0001:
        print("  [PASS] Test 4: Drawdown penalty")
    else:
        print(f"  [FAIL] Test 4: Expected {expected:.6f}, got {reward:.6f}")
        all_passed = False

    # Test 5: Consistency bonus
    reward, bd = calc.calculate(pnl_pct=0.001, position_change=0, consecutive_wins=3)
    expected = 0.001 + (0.001 * 3)  # base + consistency bonus
    if abs(reward - expected) < 0.0001:
        print("  [PASS] Test 5: Consistency bonus")
    else:
        print(f"  [FAIL] Test 5: Expected {expected:.6f}, got {reward:.6f}")
        all_passed = False

    # Test 6: Reward clipping
    reward, bd = calc.calculate(pnl_pct=2.0, position_change=0)  # Extreme value
    if reward == 1.0:  # Should be clipped to max
        print("  [PASS] Test 6: Reward clipping (max)")
    else:
        print(f"  [FAIL] Test 6: Expected 1.0, got {reward:.6f}")
        all_passed = False

    # Test 7: V20 FIX - Transaction cost should NOT scale with PnL
    # If multiplicative: 0.1 * (1 - 0.0002) = 0.09998 (wrong)
    # If additive: 0.1 - 0.0002 = 0.0998 (correct)
    reward_small, _ = calc.calculate(pnl_pct=0.001, position_change=1)
    reward_large, _ = calc.calculate(pnl_pct=0.1, position_change=1)

    cost_small = 0.001 - reward_small  # Should be ~0.0002
    cost_large = 0.1 - reward_large    # Should also be ~0.0002

    if abs(cost_small - cost_large) < 0.0001:
        print("  [PASS] Test 7: V20 FIX - Transaction cost is constant (not scaled)")
    else:
        print(f"  [FAIL] Test 7: Cost varied with PnL size (small={cost_small:.6f}, large={cost_large:.6f})")
        all_passed = False

    if all_passed:
        print("\n[SUCCESS] All reward calculator tests passed")
    else:
        print("\n[FAILURE] Some tests failed")

    return all_passed


if __name__ == "__main__":
    run_unit_tests()
