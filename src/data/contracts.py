"""
Data Contracts - SSOT for column mappings between DB and pipelines.

Contract: CTR-DATA-002
Version: 3.0.0

ARCHITECTURE 10/10 - CLEAR SEPARATION:
======================================

┌─────────────────────────────────────────────────────────────────────────┐
│                          DATA SOURCES                                    │
├───────────────────────────────┬─────────────────────────────────────────┤
│  RL PIPELINE (5-min)          │  FORECASTING PIPELINE (Daily)           │
├───────────────────────────────┼─────────────────────────────────────────┤
│  Source: TwelveData API       │  Source: Investing.com (OFFICIAL)       │
│  Table: usdcop_m5_ohlcv       │  Table: bi.dim_daily_usdcop             │
│  Loader: load_5min()          │  Loader: load_daily()                   │
│  Frequency: 5-minute bars     │  Frequency: Daily OHLCV                 │
│  Features: 15 (5-min based)   │  Features: 19 (daily based)             │
│  RSI Period: 9 bars           │  RSI Period: 14 days                    │
└───────────────────────────────┴─────────────────────────────────────────┘

CRITICAL:
- RL uses 5-min data from TwelveData (for intraday patterns)
- Forecasting uses OFFICIAL daily close from Investing.com (for accuracy)
- NEVER use resampled 5-min data for forecasting production

This module defines the Single Source of Truth (SSOT) for:
- Column name mappings between PostgreSQL and friendly names
- Feature lists for RL and Forecasting pipelines
- Target definitions for forecasting
- Table names and schema definitions
"""

from typing import Dict, Tuple, List

# =============================================================================
# MACRO COLUMN MAPPINGS (DB → Friendly Name)
# =============================================================================

MACRO_DB_TO_FRIENDLY: Dict[str, str] = {
    # FOREX
    "fxrt_index_dxy_usa_d_dxy": "dxy",
    "fxrt_spot_usdmxn_mex_d_usdmxn": "usdmxn",
    "fxrt_spot_usdclp_chl_d_usdclp": "usdclp",
    "fxrt_reer_bilateral_col_m_itcr": "itcr",

    # COMMODITIES
    "comm_oil_wti_glb_d_wti": "wti",
    "comm_oil_brent_glb_d_brent": "brent",
    "comm_metal_gold_glb_d_gold": "gold",
    "comm_agri_coffee_glb_d_coffee": "coffee",

    # VOLATILITY
    "volt_vix_usa_d_vix": "vix",

    # CREDIT RISK
    "crsk_spread_embi_col_d_embi": "embi",

    # INTEREST RATES
    "finc_bond_yield10y_usa_d_ust10y": "ust10y",
    "finc_bond_yield2y_usa_d_dgs2": "ust2y",
    "finc_bond_yield10y_col_d_col10y": "col10y",
    "finc_bond_yield5y_col_d_col5y": "col5y",
    "finc_rate_ibr_overnight_col_d_ibr": "ibr",

    # POLICY RATES
    "polr_policy_rate_col_d_tpm": "tpm",
    "polr_policy_rate_col_m_tpm": "tpm_m",
    "polr_fed_funds_usa_m_fedfunds": "fedfunds",
    "polr_prime_rate_usa_d_prime": "prime",

    # EQUITY
    "eqty_index_colcap_col_d_colcap": "colcap",

    # GDP & PRODUCTION
    "gdpp_real_gdp_usa_q_gdp_q": "gdp_usa",
    "prod_industrial_usa_m_indpro": "indpro",

    # INFLATION
    "infl_cpi_total_col_m_ipccol": "cpi_col",
    "infl_cpi_all_usa_m_cpiaucsl": "cpi_usa",
    "infl_cpi_core_usa_m_cpilfesl": "core_cpi_usa",
    "infl_pce_usa_m_pcepi": "pce",

    # LABOR
    "labr_unemployment_usa_m_unrate": "unemployment",

    # MONEY SUPPLY
    "mnys_m2_supply_usa_m_m2sl": "m2",

    # BALANCE OF PAYMENTS
    "rsbp_current_account_col_q_cacct_q": "current_account",
    "rsbp_fdi_inflow_col_q_fdiin_q": "fdi_inflow",
    "rsbp_fdi_outflow_col_q_fdiout_q": "fdi_outflow",
    "rsbp_reserves_international_col_m_resint": "reserves",

    # FOREIGN TRADE
    "ftrd_exports_total_col_m_expusd": "exports",
    "ftrd_imports_total_col_m_impusd": "imports",
    "ftrd_terms_trade_col_m_tot": "terms_of_trade",

    # SENTIMENT
    "sent_consumer_usa_m_umcsent": "consumer_sentiment",
    "crsk_sentiment_cci_col_m_cci": "cci_col",
    "crsk_sentiment_ici_col_m_ici": "ici_col",
}

# Reverse mapping (Friendly → DB)
MACRO_FRIENDLY_TO_DB: Dict[str, str] = {v: k for k, v in MACRO_DB_TO_FRIENDLY.items()}


# =============================================================================
# RL PIPELINE COLUMNS
# =============================================================================

# Macro columns used by RL training (subset of all macro)
RL_MACRO_COLUMNS: Tuple[str, ...] = (
    "dxy",
    "vix",
    "embi",
    "brent",
    "ust10y",
    "usdmxn",
    "col10y",
    "ibr",
    "tpm",
    "fedfunds",
    "colcap",
    "gold",
    "coffee",
)

# Calculated columns for RL
RL_CALCULATED_COLUMNS: Tuple[str, ...] = (
    "rate_spread",      # col10y - ust10y
    "dxy_change_1d",    # dxy.pct_change(1)
    "vix_regime",       # 1 if vix < 20, 2 if 20-30, 3 if > 30
)


# =============================================================================
# FORECASTING PIPELINE COLUMNS
# =============================================================================

# Macro columns used by Forecasting (subset of all macro)
FORECASTING_MACRO_COLUMNS: Tuple[str, ...] = (
    "dxy",
    "wti",
)

# SSOT: 19 features in exact order for Forecasting
FORECASTING_FEATURES: Tuple[str, ...] = (
    # OHLC (4)
    "close",
    "open",
    "high",
    "low",
    # Returns (4)
    "return_1d",
    "return_5d",
    "return_10d",
    "return_20d",
    # Volatility (3)
    "volatility_5d",
    "volatility_10d",
    "volatility_20d",
    # Technical (3)
    "rsi_14d",
    "ma_ratio_20d",
    "ma_ratio_50d",
    # Calendar (3)
    "day_of_week",
    "month",
    "is_month_end",
    # Macro (2)
    "dxy_close_lag1",
    "oil_close_lag1",
)

NUM_FORECASTING_FEATURES: int = 19

# Target horizons for forecasting (days)
FORECASTING_HORIZONS: Tuple[int, ...] = (1, 5, 10, 15, 20, 25, 30)

# Target column names
FORECASTING_TARGETS: Tuple[str, ...] = tuple(
    f"target_{h}d" for h in FORECASTING_HORIZONS
)

FORECASTING_RETURN_TARGETS: Tuple[str, ...] = tuple(
    f"target_return_{h}d" for h in FORECASTING_HORIZONS
)


# =============================================================================
# TABLE NAMES (PostgreSQL)
# =============================================================================

# RL Pipeline (5-minute)
RL_OHLCV_TABLE = "usdcop_m5_ohlcv"           # TimescaleDB hypertable
RL_MACRO_TABLE = "macro_indicators_daily"    # Daily macro, ffill to 5-min

# Forecasting Pipeline (Daily) - bi schema
FORECASTING_OHLCV_TABLE = "bi.dim_daily_usdcop"    # OFFICIAL daily close
FORECASTING_FEATURES_VIEW = "bi.v_forecasting_features"
FORECASTING_RESULTS_TABLE = "bi.fact_forecasts"
FORECASTING_CONSENSUS_TABLE = "bi.fact_consensus"


# =============================================================================
# OHLCV SCHEMA (5-minute for RL)
# =============================================================================

OHLCV_COLUMNS: Tuple[str, ...] = (
    "time",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

OHLCV_REQUIRED: Tuple[str, ...] = (
    "time",
    "open",
    "high",
    "low",
    "close",
)


# =============================================================================
# DAILY OHLCV SCHEMA (for Forecasting)
# =============================================================================

DAILY_OHLCV_COLUMNS: Tuple[str, ...] = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "source",  # 'investing', 'twelvedata', 'manual'
)

DAILY_OHLCV_REQUIRED: Tuple[str, ...] = (
    "date",
    "open",
    "high",
    "low",
    "close",
)

# Valid sources for daily data
DAILY_OHLCV_SOURCES: Tuple[str, ...] = (
    "investing",      # Primary: Investing.com official values
    "twelvedata",     # Secondary: TwelveData API
    "manual",         # Manual entry/corrections
    "resampled_5min", # Fallback only (NOT recommended)
)


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def validate_forecasting_features(df_columns: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate that DataFrame has all required forecasting features.

    Args:
        df_columns: List of column names from DataFrame

    Returns:
        Tuple of (is_valid, missing_columns)
    """
    missing = [col for col in FORECASTING_FEATURES if col not in df_columns]
    return len(missing) == 0, missing


def validate_ohlcv_columns(df_columns: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate that DataFrame has all required OHLCV columns.

    Args:
        df_columns: List of column names from DataFrame

    Returns:
        Tuple of (is_valid, missing_columns)
    """
    missing = [col for col in OHLCV_REQUIRED if col not in df_columns]
    return len(missing) == 0, missing


def get_db_columns_for_rl() -> List[str]:
    """Get list of DB column names needed for RL pipeline."""
    return [MACRO_FRIENDLY_TO_DB.get(col, col) for col in RL_MACRO_COLUMNS]


def get_db_columns_for_forecasting() -> List[str]:
    """Get list of DB column names needed for Forecasting pipeline."""
    return [MACRO_FRIENDLY_TO_DB.get(col, col) for col in FORECASTING_MACRO_COLUMNS]


def validate_daily_ohlcv_columns(df_columns: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate that DataFrame has all required daily OHLCV columns.

    Args:
        df_columns: List of column names from DataFrame

    Returns:
        Tuple of (is_valid, missing_columns)
    """
    missing = [col for col in DAILY_OHLCV_REQUIRED if col not in df_columns]
    return len(missing) == 0, missing


def get_data_lineage_info() -> dict:
    """
    Get documentation of data lineage for auditing.

    Returns:
        Dict with data lineage information
    """
    return {
        "rl_pipeline": {
            "ohlcv_table": RL_OHLCV_TABLE,
            "ohlcv_source": "TwelveData API (5-min)",
            "macro_table": RL_MACRO_TABLE,
            "frequency": "5-minute",
            "loader_method": "load_5min()",
            "features_count": len(RL_MACRO_COLUMNS) + len(RL_CALCULATED_COLUMNS),
        },
        "forecasting_pipeline": {
            "ohlcv_table": FORECASTING_OHLCV_TABLE,
            "ohlcv_source": "Investing.com (Official Daily)",
            "features_view": FORECASTING_FEATURES_VIEW,
            "frequency": "daily",
            "loader_method": "load_daily()",
            "features_count": NUM_FORECASTING_FEATURES,
        },
    }
