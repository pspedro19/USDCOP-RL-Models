"""
Centralized Timeframe Management
Consolidates all timeframe mappings from scattered locations
"""

from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class TimeframeConfig:
    name: str
    minutes: int
    pandas_freq: str
    description: str
    mt5_constant: Optional[int] = None

# MT5 timeframe constants
MT5_TIMEFRAMES = {
    "M1": 1,      # PERIOD_M1
    "M2": 2,      # PERIOD_M2
    "M3": 3,      # PERIOD_M3
    "M4": 4,      # PERIOD_M4
    "M5": 5,      # PERIOD_M5
    "M6": 6,      # PERIOD_M6
    "M10": 10,    # PERIOD_M10
    "M12": 12,    # PERIOD_M12
    "M15": 15,    # PERIOD_M15
    "M20": 20,    # PERIOD_M20
    "M30": 30,    # PERIOD_M30
    "H1": 16385,  # PERIOD_H1
    "H2": 16386,  # PERIOD_H2
    "H3": 16387,  # PERIOD_H3
    "H4": 16388,  # PERIOD_H4
    "H6": 16390,  # PERIOD_H6
    "H8": 16392,  # PERIOD_H8
    "H12": 16396, # PERIOD_H12
    "D1": 16408,  # PERIOD_D1
    "W1": 32769,  # PERIOD_W1
    "MN1": 49153  # PERIOD_MN1
}

# Single source of truth for all timeframes
TIMEFRAMES = {
    "M1": TimeframeConfig("M1", 1, "1T", "1 Minute", MT5_TIMEFRAMES.get("M1")),
    "M2": TimeframeConfig("M2", 2, "2T", "2 Minutes", MT5_TIMEFRAMES.get("M2")),
    "M3": TimeframeConfig("M3", 3, "3T", "3 Minutes", MT5_TIMEFRAMES.get("M3")),
    "M5": TimeframeConfig("M5", 5, "5T", "5 Minutes", MT5_TIMEFRAMES.get("M5")),
    "M10": TimeframeConfig("M10", 10, "10T", "10 Minutes", MT5_TIMEFRAMES.get("M10")),
    "M15": TimeframeConfig("M15", 15, "15T", "15 Minutes", MT5_TIMEFRAMES.get("M15")),
    "M30": TimeframeConfig("M30", 30, "30T", "30 Minutes", MT5_TIMEFRAMES.get("M30")),
    "H1": TimeframeConfig("H1", 60, "1H", "1 Hour", MT5_TIMEFRAMES.get("H1")),
    "H2": TimeframeConfig("H2", 120, "2H", "2 Hours", MT5_TIMEFRAMES.get("H2")),
    "H4": TimeframeConfig("H4", 240, "4H", "4 Hours", MT5_TIMEFRAMES.get("H4")),
    "D1": TimeframeConfig("D1", 1440, "1D", "1 Day", MT5_TIMEFRAMES.get("D1")),
}

def get_timeframe_minutes(tf: str) -> int:
    """Get minutes for timeframe"""
    return TIMEFRAMES.get(tf, TIMEFRAMES["M5"]).minutes

def get_timeframe_pandas(tf: str) -> str:
    """Get pandas frequency for timeframe"""
    return TIMEFRAMES.get(tf, TIMEFRAMES["M5"]).pandas_freq

def get_default_timeframe() -> str:
    """Get default timeframe"""
    return "M5"

def get_all_timeframes() -> Dict[str, TimeframeConfig]:
    """Get all available timeframes"""
    return TIMEFRAMES.copy()

def is_valid_timeframe(tf: str) -> bool:
    """Check if timeframe is valid"""
    return tf in TIMEFRAMES

def get_timeframe_description(tf: str) -> str:
    """Get description for timeframe"""
    return TIMEFRAMES.get(tf, TIMEFRAMES["M5"]).description

def convert_timeframe(tf: str, to_format: str = "minutes") -> Optional:
    """Convert timeframe to different formats
    
    Args:
        tf: Timeframe string (e.g., "M5", "H1")
        to_format: Target format ("minutes", "pandas", "description", "mt5")
    
    Returns:
        Converted value or None if invalid
    """
    if to_format == "minutes":
        return get_timeframe_minutes(tf)
    elif to_format == "pandas":
        return get_timeframe_pandas(tf)
    elif to_format == "description":
        return get_timeframe_description(tf)
    elif to_format == "mt5":
        config = TIMEFRAMES.get(tf)
        return config.mt5_constant if config else None
    return None
