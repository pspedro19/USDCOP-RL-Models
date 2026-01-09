"""
Dataset Configuration V20 - USDCOP Trading
==========================================

Dataset specification for V20 model training.
Defines date ranges, features, normalization, and filters.

From: 09_Documento Maestro Completo.md Section 6.8

Author: Pedro @ Lean Tech Solutions / Claude Code
Date: 2026-01-09
"""

# Dataset V20 Configuration
DATASET_CONFIG_V20 = {
    # Source data
    "source_table": "usdcop_m5_ohlcv",
    "macro_table": "macro_indicators_daily",

    # Date ranges for train/validation/test splits
    "date_ranges": {
        "train": {
            "start": "2020-01-01",
            "end": "2024-12-31",
            "description": "Training data: 5 years of historical data"
        },
        "validation": {
            "start": "2025-01-01",
            "end": "2025-06-30",
            "description": "Validation data: First half of 2025"
        },
        "test": {
            "start": "2025-07-01",
            "end": "2026-01-08",
            "description": "Test data: Second half of 2025 to present (OOS)"
        },
    },

    # V19 Feature list (15 dimensions: 13 core + 2 state)
    "features": {
        "core_market": [
            "log_ret_5m",       # 5-minute log return
            "log_ret_1h",       # 1-hour log return
            "log_ret_4h",       # 4-hour log return
            "rsi_9",            # RSI with 9 period
            "atr_pct",          # ATR as percentage of price
            "adx_14",           # ADX trend strength
        ],
        "macro": [
            "dxy_z",            # DXY z-score normalized
            "dxy_change_1d",    # DXY 1-day change
            "vix_z",            # VIX z-score normalized
            "embi_z",           # EMBI Colombia z-score
            "brent_change_1d",  # Brent oil 1-day change
            "rate_spread",      # Interest rate spread
            "usdmxn_change_1d", # USD/MXN correlation proxy
        ],
        "state": [
            "position",         # Current position (-1, 0, 1)
            "time_normalized",  # Time in session (0 to 1)
        ],
        "total_dim": 15,
    },

    # Normalization configuration
    "normalization": {
        "method": "z_score",
        "clip_range": [-5.0, 5.0],      # Clip extreme values
        "use_train_stats": True,         # Use training set statistics for all
        "stats_file": "config/v19_norm_stats.json",
    },

    # Market hours filter (Colombian trading hours)
    "filters": {
        "market_hours_only": True,
        "market_open": "08:00",          # 8:00 AM COT
        "market_close": "12:55",         # 12:55 PM COT
        "timezone": "America/Bogota",
        "exclude_holidays": True,
        "exclude_us_holidays": True,     # V20: Include US holidays (affects USD)
        "min_volume": 0,                 # No volume filter for FX
    },

    # Episode configuration
    "episode": {
        "bars_per_session": 60,          # 5 hours * 12 bars/hour
        "overlap_sessions": False,       # No overlapping episodes
        "random_start": True,            # Random start within session
    },
}

# Split percentages (alternative to date ranges)
SPLIT_CONFIG = {
    "train_pct": 0.70,
    "val_pct": 0.15,
    "test_pct": 0.15,
    "shuffle": False,    # Time series, no shuffle
    "random_state": 42,
}


def get_dataset_config():
    """Get V20 dataset configuration."""
    return DATASET_CONFIG_V20.copy()


def get_feature_list():
    """Get flat list of all features."""
    config = DATASET_CONFIG_V20["features"]
    return (
        config["core_market"] +
        config["macro"] +
        config["state"]
    )


def get_date_range(split: str):
    """Get date range for a specific split.

    Args:
        split: One of 'train', 'validation', 'test'

    Returns:
        Tuple of (start_date, end_date)
    """
    ranges = DATASET_CONFIG_V20["date_ranges"]
    if split not in ranges:
        raise ValueError(f"Invalid split: {split}. Must be one of {list(ranges.keys())}")

    return ranges[split]["start"], ranges[split]["end"]


if __name__ == "__main__":
    print("Dataset V20 Configuration:")
    print(f"  Total features: {DATASET_CONFIG_V20['features']['total_dim']}")
    print(f"  Feature list: {get_feature_list()}")
    print("\nDate Ranges:")
    for split, config in DATASET_CONFIG_V20["date_ranges"].items():
        print(f"  {split}: {config['start']} to {config['end']}")
    print(f"\nFilters: Market hours {DATASET_CONFIG_V20['filters']['market_open']} - {DATASET_CONFIG_V20['filters']['market_close']} COT")
