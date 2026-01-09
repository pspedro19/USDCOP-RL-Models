"""
Risk Calculator Module
=======================
Calculate stop loss, take profit, position sizing, and risk metrics.

Author: Pedro @ Lean Tech Solutions
Created: 2025-12-17
"""

import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class RiskCalculator:
    """Calculate risk management parameters"""

    @staticmethod
    def calculate_position_size(
        account_balance: float,
        risk_per_trade_pct: float,
        entry_price: float,
        stop_loss_price: float,
        max_position_size_pct: float = 0.1
    ) -> Tuple[float, float]:
        """
        Calculate position size based on risk management rules.

        Args:
            account_balance: Total account balance
            risk_per_trade_pct: Risk per trade as percentage (e.g., 0.02 for 2%)
            entry_price: Entry price
            stop_loss_price: Stop loss price
            max_position_size_pct: Maximum position size as percentage (default: 0.1 for 10%)

        Returns:
            Tuple of (position_size_shares, position_size_pct)
        """
        try:
            # Calculate risk amount in currency
            risk_amount = account_balance * risk_per_trade_pct

            # Calculate risk per share
            risk_per_share = abs(entry_price - stop_loss_price)

            if risk_per_share == 0:
                logger.warning("Risk per share is zero, using minimum position size")
                return 1.0, 0.01

            # Calculate position size in shares
            position_shares = risk_amount / risk_per_share

            # Calculate position value
            position_value = position_shares * entry_price

            # Calculate position size as percentage of account
            position_size_pct = position_value / account_balance

            # Cap at maximum position size
            if position_size_pct > max_position_size_pct:
                position_size_pct = max_position_size_pct
                position_shares = (account_balance * position_size_pct) / entry_price

            return float(position_shares), float(position_size_pct)

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 1.0, 0.01

    @staticmethod
    def calculate_atr_based_levels(
        current_price: float,
        atr: float,
        is_long: bool,
        sl_multiplier: float = 2.0,
        tp_multiplier: float = 3.0
    ) -> Dict[str, float]:
        """
        Calculate stop loss and take profit based on ATR.

        Args:
            current_price: Current market price
            atr: Average True Range
            is_long: True for long position, False for short
            sl_multiplier: ATR multiplier for stop loss (default: 2.0)
            tp_multiplier: ATR multiplier for take profit (default: 3.0)

        Returns:
            Dictionary with 'stop_loss', 'take_profit', 'risk_reward_ratio'
        """
        try:
            if is_long:
                stop_loss = current_price - (atr * sl_multiplier)
                take_profit = current_price + (atr * tp_multiplier)
            else:
                stop_loss = current_price + (atr * sl_multiplier)
                take_profit = current_price - (atr * tp_multiplier)

            # Calculate risk/reward ratio
            risk = abs(current_price - stop_loss)
            reward = abs(take_profit - current_price)
            risk_reward_ratio = reward / risk if risk > 0 else 0.0

            return {
                'stop_loss': max(stop_loss, 0.01),  # Ensure positive
                'take_profit': max(take_profit, 0.01),
                'risk_reward_ratio': risk_reward_ratio
            }

        except Exception as e:
            logger.error(f"Error calculating ATR-based levels: {e}")
            # Return conservative defaults
            return {
                'stop_loss': current_price * 0.98,
                'take_profit': current_price * 1.02,
                'risk_reward_ratio': 1.0
            }

    @staticmethod
    def calculate_percentage_based_levels(
        current_price: float,
        is_long: bool,
        sl_pct: float = 0.02,
        tp_pct: float = 0.03
    ) -> Dict[str, float]:
        """
        Calculate stop loss and take profit based on percentage.

        Args:
            current_price: Current market price
            is_long: True for long position, False for short
            sl_pct: Stop loss percentage (default: 0.02 for 2%)
            tp_pct: Take profit percentage (default: 0.03 for 3%)

        Returns:
            Dictionary with 'stop_loss', 'take_profit', 'risk_reward_ratio'
        """
        try:
            if is_long:
                stop_loss = current_price * (1 - sl_pct)
                take_profit = current_price * (1 + tp_pct)
            else:
                stop_loss = current_price * (1 + sl_pct)
                take_profit = current_price * (1 - tp_pct)

            risk_reward_ratio = tp_pct / sl_pct

            return {
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward_ratio': risk_reward_ratio
            }

        except Exception as e:
            logger.error(f"Error calculating percentage-based levels: {e}")
            return {
                'stop_loss': current_price * 0.98,
                'take_profit': current_price * 1.02,
                'risk_reward_ratio': 1.0
            }

    @staticmethod
    def calculate_risk_metrics(
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        position_size: float,
        account_balance: float
    ) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics for a trade.

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            position_size: Position size as percentage of account
            account_balance: Total account balance

        Returns:
            Dictionary with risk metrics
        """
        try:
            # Calculate position value
            position_value = account_balance * position_size

            # Calculate risk amount
            risk_per_share = abs(entry_price - stop_loss)
            risk_amount = (position_value / entry_price) * risk_per_share
            risk_pct = (risk_amount / account_balance) * 100

            # Calculate reward amount
            reward_per_share = abs(take_profit - entry_price)
            reward_amount = (position_value / entry_price) * reward_per_share
            reward_pct = (reward_amount / account_balance) * 100

            # Calculate risk/reward ratio
            risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0.0

            return {
                'position_value': position_value,
                'risk_amount': risk_amount,
                'risk_pct': risk_pct,
                'reward_amount': reward_amount,
                'reward_pct': reward_pct,
                'risk_reward_ratio': risk_reward_ratio
            }

        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {
                'position_value': 0.0,
                'risk_amount': 0.0,
                'risk_pct': 0.0,
                'reward_amount': 0.0,
                'reward_pct': 0.0,
                'risk_reward_ratio': 0.0
            }

    @staticmethod
    def validate_trade(
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        is_long: bool,
        min_risk_reward: float = 1.5
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate if a trade setup meets risk management criteria.

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            is_long: True for long position, False for short
            min_risk_reward: Minimum acceptable risk/reward ratio

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check for valid prices
            if any(p <= 0 for p in [entry_price, stop_loss, take_profit]):
                return False, "All prices must be positive"

            # Check stop loss placement
            if is_long:
                if stop_loss >= entry_price:
                    return False, "Stop loss must be below entry for long position"
                if take_profit <= entry_price:
                    return False, "Take profit must be above entry for long position"
            else:
                if stop_loss <= entry_price:
                    return False, "Stop loss must be above entry for short position"
                if take_profit >= entry_price:
                    return False, "Take profit must be below entry for short position"

            # Calculate risk/reward ratio
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            risk_reward_ratio = reward / risk if risk > 0 else 0.0

            # Check minimum risk/reward
            if risk_reward_ratio < min_risk_reward:
                return False, f"Risk/reward ratio {risk_reward_ratio:.2f} below minimum {min_risk_reward}"

            return True, None

        except Exception as e:
            logger.error(f"Error validating trade: {e}")
            return False, f"Validation error: {str(e)}"

    @staticmethod
    def calculate_kelly_criterion(
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate optimal position size using Kelly Criterion.

        Args:
            win_rate: Win rate (0-1)
            avg_win: Average winning trade amount
            avg_loss: Average losing trade amount (positive value)

        Returns:
            Optimal position size as decimal (e.g., 0.05 for 5%)
        """
        try:
            if avg_loss == 0:
                return 0.02  # Default 2%

            # Kelly formula: f = (p * b - q) / b
            # where p = win rate, q = loss rate, b = win/loss ratio
            win_loss_ratio = avg_win / avg_loss
            kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio

            # Apply fractional Kelly (0.25x) for safety
            fractional_kelly = kelly * 0.25

            # Cap between 1% and 5%
            return float(np.clip(fractional_kelly, 0.01, 0.05))

        except Exception as e:
            logger.error(f"Error calculating Kelly Criterion: {e}")
            return 0.02  # Default 2%
