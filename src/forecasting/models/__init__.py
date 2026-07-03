"""
Forecasting Models
==================

ML models for USD/COP exchange rate prediction.

Available Models:
- Linear: Ridge, Bayesian Ridge, ARD
- Boosting: XGBoost, LightGBM, CatBoost
- Hybrid: XGBoost+Ridge, LightGBM+Ridge, CatBoost+Ridge

Usage:
    from src.forecasting.models import ModelFactory

    # Create model instance
    model = ModelFactory.create('xgboost_pure')

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

@version 1.0.0
"""

from src.forecasting.models.factory import ModelFactory
from src.forecasting.models.base import BaseModel

__all__ = [
    'ModelFactory',
    'BaseModel',
]
