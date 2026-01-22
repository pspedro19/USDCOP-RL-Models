"""
Demo Mode Configuration
=======================

SSOT: config.models table (model_id='investor_demo')

The demo mode is now a selectable model in the UI.
To enable demo mode:
1. Select "investor_demo" model in the backtest/replay UI
2. The InferenceEngine will detect algorithm='SYNTHETIC' and use synthetic trade generation

This file provides helper functions to check if a model_id is demo mode
and to load demo config from the database.
"""

import os
import logging
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# SSOT: model_id for demo mode
DEMO_MODEL_ID = "investor_demo"


def is_demo_model(model_id: str) -> bool:
    """
    Check if the given model_id is the demo/investor mode model.

    This is the DRY way to check for demo mode - just compare model_id.
    No separate flags or environment variables needed.

    Args:
        model_id: The model identifier from the request

    Returns:
        True if this is the demo model
    """
    return model_id == DEMO_MODEL_ID


def is_investor_mode() -> bool:
    """
    Check if INVESTOR_MODE environment variable is enabled.

    This is for backwards compatibility with the environment variable approach.
    The preferred approach is to use is_demo_model(model_id) which checks by model_id.

    Returns:
        True if INVESTOR_MODE=true in environment
    """
    return os.getenv("INVESTOR_MODE", "false").lower() in ("true", "1", "yes")


@dataclass
class DemoConfig:
    """
    Configuration for demo/investor mode.

    SSOT: These values come from config.models.hyperparameters
    and config.models.policy_config for model_id='investor_demo'

    This dataclass is a local cache - the true source is the database.
    """
    # Target metrics (from hyperparameters)
    target_sharpe: float = 2.1
    target_max_drawdown: float = -0.095
    target_win_rate: float = 0.61
    target_annual_return: float = 0.32
    target_profit_factor: float = 1.85
    trades_per_month: int = 18

    # Trade parameters (from policy_config)
    avg_win_pct: float = 0.005
    avg_loss_pct: float = 0.004
    max_position_duration_minutes: int = 300
    min_position_duration_minutes: int = 15

    # Risk management (from environment_config)
    stop_loss_pct: float = 0.005
    take_profit_pct: float = 0.012
    market_open_hour: int = 8
    market_close_hour: int = 12

    # Model info
    model_id: str = DEMO_MODEL_ID
    model_name: str = "Demo Mode (Investor Presentation)"

    # Monthly bias for trade direction (positive = more longs, negative = more shorts)
    monthly_bias: Dict[int, float] = None

    def __post_init__(self):
        if self.monthly_bias is None:
            # Default monthly bias for 2025 (bearish year)
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

    @classmethod
    def from_db_model(cls, model_record: Dict) -> "DemoConfig":
        """
        Create DemoConfig from a database model record.

        Args:
            model_record: Row from config.models table

        Returns:
            DemoConfig instance with values from database
        """
        hp = model_record.get("hyperparameters", {})
        pc = model_record.get("policy_config", {})
        ec = model_record.get("environment_config", {})

        return cls(
            target_sharpe=hp.get("target_sharpe", 2.1),
            target_max_drawdown=hp.get("target_max_drawdown", -0.095),
            target_win_rate=hp.get("target_win_rate", 0.61),
            target_annual_return=hp.get("target_annual_return", 0.32),
            target_profit_factor=hp.get("target_profit_factor", 1.85),
            trades_per_month=hp.get("trades_per_month", 18),
            avg_win_pct=pc.get("avg_win_pct", 0.005),
            avg_loss_pct=pc.get("avg_loss_pct", 0.004),
            max_position_duration_minutes=pc.get("max_position_duration_minutes", 300),
            min_position_duration_minutes=pc.get("min_position_duration_minutes", 15),
            stop_loss_pct=ec.get("stop_loss_pct", 0.005),
            take_profit_pct=ec.get("take_profit_pct", 0.012),
            market_open_hour=ec.get("market_open_hour", 8),
            market_close_hour=ec.get("market_close_hour", 12),
            model_id=model_record.get("model_id", DEMO_MODEL_ID),
            model_name=model_record.get("name", "Demo Mode"),
        )


# Default config instance (used when DB not available)
DEMO_CONFIG = DemoConfig()


# Key winning trades to highlight (major moves captured)
# These are synthetic examples for the demo presentation
KEY_WINNING_TRADES = [
    # Format: (month, day, side, entry_price_approx, profit_pct, description)
    (1, 6, "SHORT", 4350, 0.025, "Captured Jan sell-off"),
    (4, 7, "LONG", 4200, 0.035, "April rally captured"),
    (4, 11, "SHORT", 4380, 0.028, "Post-rally reversal"),
    (7, 15, "SHORT", 4150, 0.032, "July downtrend entry"),
    (9, 10, "SHORT", 4000, 0.045, "September breakdown"),
    (10, 8, "SHORT", 3920, 0.022, "October continuation"),
]


# Monthly bias for 2025 (bearish year)
# Positive = more longs, Negative = more shorts
MONTHLY_BIAS_2025 = {
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
