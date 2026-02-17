"""
Backtest Configuration - Single Source of Truth (SSOT)
======================================================

This file defines THE canonical backtest configuration used across:
- L4 Backtest Validation DAG
- Dashboard Backtest Replay
- Model Evaluation Engine

CRITICAL: Any changes here affect ALL backtest calculations.
         Ensure L4 and Dashboard use these exact values.

Author: USDCOP Trading Team
Version: 1.0.0
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class BacktestConfigSSOT:
    """
    Immutable backtest configuration.

    All values are calibrated for USDCOP forex trading:
    - Spread based on MEXC institutional pricing
    - Thresholds optimized during model training
    - Capital aligned with position sizing rules
    """

    # === TRANSACTION COSTS ===
    # Based on MEXC forex pricing for USDCOP
    spread_bps: float = 2.5          # Bid-ask spread in basis points
    slippage_bps: float = 1.0        # Execution slippage

    @property
    def total_cost_bps(self) -> float:
        """Total round-trip transaction cost."""
        return (self.spread_bps + self.slippage_bps) * 2  # Entry + Exit

    # === ENTRY/EXIT THRESHOLDS ===
    # Must match training environment for consistency
    threshold_long: float = 0.50     # Model confidence for LONG entry
    threshold_short: float = -0.50   # Model confidence for SHORT entry
    exit_threshold: float = 0.0      # Neutral zone for exit consideration

    # === CAPITAL & POSITION ===
    initial_capital: float = 10_000.0    # Starting capital in USD
    position_size_pct: float = 1.0       # 100% of capital per trade
    max_position_bars: int = 576         # Max bars to hold (2 days @ 5min)

    # === RISK MANAGEMENT ===
    stop_loss_pct: float = 0.025         # 2.5% stop loss
    take_profit_pct: float = 0.030       # 3.0% take profit
    trailing_stop_enabled: bool = True
    trailing_stop_activation_pct: float = 0.015  # Activate at 1.5% profit
    trailing_stop_trail_factor: float = 0.5      # Trail at 50% of max gain

    # === MARKET ASSUMPTIONS ===
    bars_per_trading_day: int = 144      # 12 hours * 12 bars/hour (5min bars)
    trading_days_per_year: int = 252     # Standard forex trading days

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "spread_bps": self.spread_bps,
            "slippage_bps": self.slippage_bps,
            "total_cost_bps": self.total_cost_bps,
            "threshold_long": self.threshold_long,
            "threshold_short": self.threshold_short,
            "exit_threshold": self.exit_threshold,
            "initial_capital": self.initial_capital,
            "position_size_pct": self.position_size_pct,
            "max_position_bars": self.max_position_bars,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "trailing_stop_enabled": self.trailing_stop_enabled,
            "trailing_stop_activation_pct": self.trailing_stop_activation_pct,
            "trailing_stop_trail_factor": self.trailing_stop_trail_factor,
            "bars_per_trading_day": self.bars_per_trading_day,
            "trading_days_per_year": self.trading_days_per_year,
        }


# === SINGLETON INSTANCE ===
# Use this throughout the codebase
BACKTEST_SSOT = BacktestConfigSSOT()


# === CONVENIENCE EXPORTS ===
def get_backtest_config() -> BacktestConfigSSOT:
    """Get the canonical backtest configuration."""
    return BACKTEST_SSOT


def get_backtest_config_dict() -> Dict[str, Any]:
    """Get backtest config as dictionary."""
    return BACKTEST_SSOT.to_dict()


# === VALIDATION ===
def validate_config_consistency(external_config: Dict[str, Any]) -> bool:
    """
    Validate that an external config matches SSOT.
    Used to verify dashboard/API configs align with L4.

    Returns True if configs match, False otherwise.
    """
    ssot = BACKTEST_SSOT.to_dict()

    # Check critical parameters
    critical_keys = ["spread_bps", "threshold_long", "threshold_short", "initial_capital"]

    for key in critical_keys:
        if key in external_config:
            if abs(external_config[key] - ssot[key]) > 1e-6:
                return False

    return True


# === L4 VALIDATION THRESHOLDS ===
# These thresholds determine if a model passes L4 validation
L4_VALIDATION_THRESHOLDS = {
    "min_sharpe_ratio": 0.5,
    "max_drawdown_pct": 0.15,      # 15% max drawdown
    "min_win_rate": 0.45,          # 45% minimum win rate
    "min_trades": 30,              # At least 30 trades in test period
    "min_profit_factor": 1.2,
}


if __name__ == "__main__":
    # Print config for verification
    print("=== BACKTEST CONFIG SSOT ===")
    for k, v in BACKTEST_SSOT.to_dict().items():
        print(f"  {k}: {v}")

    print("\n=== L4 VALIDATION THRESHOLDS ===")
    for k, v in L4_VALIDATION_THRESHOLDS.items():
        print(f"  {k}: {v}")
