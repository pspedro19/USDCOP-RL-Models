"""
Forecasting Module
==================

USD/COP Exchange Rate Forecasting using ML models.

This module provides:
- Multiple ML models (Ridge, XGBoost, LightGBM, CatBoost, Hybrids)
- Backtesting and evaluation metrics
- Feature engineering pipeline
- Inference service

Architecture:
- Models: Factory pattern for model creation
- Evaluation: Strategy pattern for metrics
- Repository: Repository pattern for data access

Usage:
    from src.forecasting import ModelFactory, BacktestEngine

    # Create a model
    model = ModelFactory.create('xgboost_pure', horizon=15)

    # Run backtest
    engine = BacktestEngine()
    results = engine.run(model, data)

@version 1.0.0
@lastSync 2026-01-22
"""

from src.forecasting.models import ModelFactory
from src.forecasting.evaluation import BacktestEngine, WalkForwardValidator, Metrics
from src.forecasting.config import get_config, ForecastingConfig
from src.forecasting.data_contracts import (
    FEATURE_COLUMNS,
    NUM_FEATURES,
    TARGET_HORIZONS,
    DATA_CONTRACT_VERSION,
)
from src.forecasting.engine import ForecastingEngine

__all__ = [
    # Models
    'ModelFactory',
    # Evaluation
    'BacktestEngine',
    'WalkForwardValidator',
    'Metrics',
    # Engine
    'ForecastingEngine',
    # Configuration
    'get_config',
    'ForecastingConfig',
    # Data Contracts
    'FEATURE_COLUMNS',
    'NUM_FEATURES',
    'TARGET_HORIZONS',
    'DATA_CONTRACT_VERSION',
]

__version__ = '1.0.0'
