"""
V3 DAG Architecture - Consolidated USD/COP Trading System

This module contains the v3 consolidated architecture with 5 clean DAGs:
1. l0_ohlcv_realtime.py - OHLCV data acquisition every 5min
2. l0_macro_daily.py - Macro data scraping 3x/day
3. l1_feature_refresh.py - Feature refresh and Python feature calculation
4. l5_realtime_inference.py - Real-time RL inference (CRITICAL - reads from config)
5. alert_monitor.py - System monitoring and alerts

Key improvements over v2:
- Reads from feature_config.json (Single Source of Truth)
- Correct observation space: 15 dimensions (13 features + position + time_normalized)
- Uses FeatureBuilder service for Python feature calculations
- No hardcoded features or norm_stats
"""

__version__ = "3.0.0"
__author__ = "Pedro @ Lean Tech Solutions"
