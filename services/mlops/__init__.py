"""
USDCOP Trading MLOps Module
===========================

Industry-grade MLOps components for production inference:
- ONNX-based inference engine
- Risk management and circuit breakers
- Data drift monitoring
- Feature caching
- Health monitoring

Usage:
    from services.mlops import (
        InferenceEngine,
        RiskManager,
        DriftMonitor,
        FeatureCache,
        MLOpsConfig
    )
"""

from .config import MLOpsConfig, RiskLimits, TradingHours
from .inference_engine import InferenceEngine, get_inference_engine
from .risk_manager import RiskManager, get_risk_manager
from .drift_monitor import DriftMonitor
from .feature_cache import FeatureCache

__all__ = [
    'MLOpsConfig',
    'RiskLimits',
    'TradingHours',
    'InferenceEngine',
    'get_inference_engine',
    'RiskManager',
    'get_risk_manager',
    'DriftMonitor',
    'FeatureCache',
]

__version__ = '1.0.0'
