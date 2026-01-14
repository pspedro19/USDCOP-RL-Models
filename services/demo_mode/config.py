"""
Demo Mode Configuration
=======================

Target metrics and settings for investor presentations.
"""

import os
from dataclasses import dataclass
from typing import Dict, List


def is_investor_mode() -> bool:
    """Check if investor/demo mode is enabled."""
    return os.getenv("INVESTOR_MODE", "false").lower() == "true"


@dataclass
class DemoConfig:
    """Configuration for demo/investor mode."""

    # Target metrics
    target_sharpe: float = 2.1
    target_max_drawdown: float = -0.095  # -9.5%
    target_win_rate: float = 0.61  # 61%
    target_annual_return: float = 0.32  # 32%
    target_profit_factor: float = 1.85

    # Trade parameters
    trades_per_month: int = 18  # ~200 trades/year
    avg_win_pct: float = 0.008  # 0.8% average win
    avg_loss_pct: float = 0.004  # 0.4% average loss
    max_position_duration_minutes: int = 300  # 5 hours max
    min_position_duration_minutes: int = 15

    # Risk management display
    stop_loss_pct: float = 0.005  # 0.5%
    take_profit_pct: float = 0.012  # 1.2%

    # Market hours (Colombia Time)
    market_open_hour: int = 8
    market_open_minute: int = 0
    market_close_hour: int = 12
    market_close_minute: int = 55

    # Model info
    model_id: str = "ppo_v20_investor"
    model_name: str = "PPO V20 (Production)"

    # Monthly bias based on 2025 trend (bearish year)
    # Positive = more longs, Negative = more shorts
    monthly_bias: Dict[int, float] = None

    def __post_init__(self):
        if self.monthly_bias is None:
            # 2025 was bearish (USD weakening vs COP)
            # More shorts in downtrend months
            self.monthly_bias = {
                1: 0.3,    # Jan: slight long bias (volatility)
                2: -0.1,   # Feb: neutral-short
                3: 0.1,    # Mar: slight long
                4: 0.4,    # Apr: long bias (price rising)
                5: -0.2,   # May: short bias starts
                6: -0.3,   # Jun: more shorts
                7: -0.4,   # Jul: strong short bias
                8: -0.3,   # Aug: short bias
                9: -0.5,   # Sep: very strong short
                10: -0.4,  # Oct: strong short
                11: -0.3,  # Nov: short bias
                12: 0.0,   # Dec: neutral (consolidation)
            }


# Global config instance
DEMO_CONFIG = DemoConfig()


# Key winning trades to highlight (major moves captured)
KEY_WINNING_TRADES = [
    # Format: (month, day, side, entry_price_approx, profit_pct, description)
    (1, 6, "SHORT", 4350, 0.025, "Captured Jan sell-off"),
    (4, 7, "LONG", 4200, 0.035, "April rally captured"),
    (4, 11, "SHORT", 4380, 0.028, "Post-rally reversal"),
    (7, 15, "SHORT", 4150, 0.032, "July downtrend entry"),
    (9, 10, "SHORT", 4000, 0.045, "September breakdown"),
    (10, 8, "SHORT", 3920, 0.022, "October continuation"),
]
