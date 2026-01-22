"""
Configuration for Inference API Service
Uses Feature Contract as SSOT for norm_stats path and FEATURE_ORDER
"""

import os
import sys
from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache

# Add project root to path for imports
_project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "src"))

# Import FEATURE_ORDER and OBSERVATION_DIM from SSOT (src/core/constants.py)
try:
    from src.core.constants import FEATURE_ORDER, OBSERVATION_DIM
    _ssot_available = True
except ImportError:
    # Fallback to contracts if constants not available
    try:
        from src.core.contracts import FEATURE_ORDER, OBSERVATION_DIM
        _ssot_available = True
    except ImportError:
        _ssot_available = False
        FEATURE_ORDER = None
        OBSERVATION_DIM = 15

try:
    from features.contract import FEATURE_CONTRACT, NORM_STATS_PATH
    _contract = FEATURE_CONTRACT
    _norm_stats_path = NORM_STATS_PATH
except ImportError:
    # Fallback if contract not available yet
    _contract = None
    _norm_stats_path = "config/norm_stats.json"


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Database - P0-4: Use env vars only, no hardcoded defaults
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_db: str = os.getenv("POSTGRES_DB", "usdcop_trading")
    postgres_user: str = os.getenv("POSTGRES_USER", "")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "")

    # Model paths - use MODEL_PATH env var (Docker) or relative path (local dev)
    project_root: Path = Path(__file__).parent.parent.parent  # Go up to USDCOP-RL-Models
    model_base_path: str = os.getenv("MODEL_PATH", "models")  # Docker: /models, Local: models
    model_subpath: str = "ppo_v20_production/final_model.zip"  # Model file within base path

    # Use norm_stats from Feature Contract constant (SSOT)
    norm_stats_path: str = _norm_stats_path

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

    # Feature configuration - P0-1: Use SSOT from src.core.contracts
    observation_dim: int = OBSERVATION_DIM  # 13 core + 2 state features from SSOT

    @property
    def database_url(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @property
    def full_model_path(self) -> Path:
        """Get full path to model file. Uses MODEL_PATH env var in Docker."""
        base = Path(self.model_base_path)
        if base.is_absolute():
            # Docker: MODEL_PATH=/models (absolute)
            return base / self.model_subpath
        else:
            # Local dev: relative to project root
            return self.project_root / base / self.model_subpath

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


# =============================================================================
# FEATURE_ORDER - SSOT Import (DO NOT DEFINE LOCALLY)
# =============================================================================
# FEATURE_ORDER is imported from src.core.contracts (SSOT) at module level.
# See src/core/contracts/feature_contract.py for the canonical definition.
# Contains 15 features:
#   0-5: Technical features (log_ret_5m, log_ret_1h, log_ret_4h, rsi_9, atr_pct, adx_14)
#   6-12: Macro features (dxy_z, dxy_change_1d, vix_z, embi_z, brent_change_1d, rate_spread, usdmxn_change_1d)
#   13-14: State features (position, time_normalized)
#
# If SSOT import fails, FEATURE_ORDER will be None - caller must handle this.
# This ensures no duplicate definitions exist outside the SSOT.

# Macro feature mappings from database columns
MACRO_COLUMN_MAP = {
    "dxy": "fxrt_index_dxy_usa_d_dxy",
    "vix": "volt_vix_usa_d_vix",
    "embi": "crsk_spread_embi_col_d_embi",
    "brent": "comm_oil_brent_glb_d_brent",
    "treasury_10y": "finc_bond_yield10y_usa_d_ust10y",
    "usdmxn": "fxrt_spot_usdmxn_mex_d_usdmxn",
}
