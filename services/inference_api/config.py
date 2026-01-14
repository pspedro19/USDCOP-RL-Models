"""
Configuration for Inference API Service
Uses Feature Contract as SSOT for norm_stats path
"""

import os
import sys
from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache

# Add project root to path for imports
_project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_project_root / "src"))

try:
    from features.contract import FEATURE_CONTRACT
    _contract = FEATURE_CONTRACT
except ImportError:
    # Fallback if contract not available yet
    _contract = None


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Database - P0-4: Use env vars only, no hardcoded defaults
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_db: str = os.getenv("POSTGRES_DB", "usdcop_trading")
    postgres_user: str = os.getenv("POSTGRES_USER", "")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "")

    # Model paths (relative to project root)
    project_root: Path = Path(__file__).parent.parent.parent  # Go up to USDCOP-RL-Models
    model_path: str = "models/ppo_production/final_model.zip"  # 15-dim observation model

    # Use norm_stats from Feature Contract (SSOT)
    norm_stats_path: str = _contract.norm_stats_path if _contract else "config/norm_stats.json"

    # Trading parameters - from config/trading_config.yaml SSOT
    initial_capital: float = 10000.0
    transaction_cost_bps: float = 75.0  # From SSOT: costs.transaction_cost_bps
    slippage_bps: float = 15.0  # From SSOT: costs.slippage_bps

    # Action thresholds - from config/trading_config.yaml SSOT
    threshold_long: float = 0.33  # From SSOT: thresholds.long
    threshold_short: float = -0.33  # From SSOT: thresholds.short

    # Risk Management
    stop_loss_pct: float = 1.5  # 1.5% stop-loss
    take_profit_pct: float = 3.0  # 3% take-profit (2:1 reward/risk)

    # Position Sizing based on confidence
    min_position_size: float = 0.5  # 50% of capital minimum
    max_position_size: float = 1.0  # 100% of capital maximum

    # === POSITION BIAS FIX ===
    # Max position duration (force exit after N bars to prevent year-long holds)
    max_position_duration_bars: int = 60  # 60 bars × 5min = 5 hours max hold
    min_bars_between_trades: int = 6  # 30 min cooldown between trades

    # Dynamic thresholds - from config/trading_config.yaml SSOT
    threshold_long_entry: float = 0.33   # From SSOT: thresholds.long
    threshold_short_entry: float = -0.33  # From SSOT: thresholds.short
    threshold_exit: float = 0.10  # Exit threshold within HOLD zone

    # Feature configuration - P0-1: Use contract as source of truth
    observation_dim: int = _contract.observation_dim if _contract else 15  # 13 core + 2 state features

    @property
    def database_url(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @property
    def full_model_path(self) -> Path:
        return self.project_root / self.model_path

    @property
    def full_norm_stats_path(self) -> Path:
        return self.project_root / self.norm_stats_path

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Feature order for observation builder (must match training)
FEATURE_ORDER = [
    # Core market features (13)
    "log_ret_5m",       # 0: 5-min log return
    "log_ret_1h",       # 1: 1-hour log return
    "log_ret_4h",       # 2: 4-hour log return
    "rsi_9",            # 3: RSI period 9
    "atr_pct",          # 4: ATR percentage
    "adx_14",           # 5: ADX period 14
    "dxy_z",            # 6: DXY z-score
    "dxy_change_1d",    # 7: DXY daily % change
    "vix_z",            # 8: VIX z-score
    "embi_z",           # 9: EMBI z-score
    "brent_change_1d",  # 10: Brent daily % change
    "rate_spread",      # 11: UST 10Y spread
    "usdmxn_change_1d", # 12: USDMXN hourly return
    # State features (2)
    "position",         # 13: Current position (-1 to 1)
    "time_normalized",  # 14: Normalized session time (0 to 1)
]

# Macro feature mappings from database columns
MACRO_COLUMN_MAP = {
    "dxy": "fxrt_index_dxy_usa_d_dxy",
    "vix": "volt_vix_usa_d_vix",
    "embi": "crsk_spread_embi_col_d_embi",
    "brent": "comm_oil_brent_glb_d_brent",
    "treasury_10y": "finc_bond_yield10y_usa_d_ust10y",
    "usdmxn": "fxrt_spot_usdmxn_mex_d_usdmxn",
}
