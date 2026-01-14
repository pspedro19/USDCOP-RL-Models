"""
USDCOP Inference API Service
============================

FastAPI service for on-demand backtesting and trade generation.
Generates trades by running PPO model inference on historical data
when trades don't exist for a given date range.
"""

__version__ = "1.0.0"
# Trigger reload
