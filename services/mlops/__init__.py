"""
USDCOP Trading MLOps Module
===========================

Industry-grade MLOps components for production inference:
- ONNX-based inference engine
- Risk management and circuit breakers
- Data drift monitoring
- Feature caching
- Health monitoring

Architecture:
- Legacy: Direct class imports (backward compatible)
- Modern: Via bridge module using SOLID refactored code

Usage (Legacy):
    from services.mlops import (
        InferenceEngine,
        RiskManager,
        DriftMonitor,
        FeatureCache,
        MLOpsConfig
    )

Usage (Modern - SOLID Architecture):
    from services.mlops.bridge import (
        get_inference_engine_v2,
        get_risk_check_chain,
        get_ensemble_strategy_registry,
    )

Author: Trading Team
Version: 2.0.0
Date: 2025-01-14
"""

# Legacy imports (backward compatible)
from mlops.config import MLOpsConfig, RiskLimits, TradingHours
from mlops.inference_engine import InferenceEngine, get_inference_engine
from mlops.risk_manager import RiskManager, get_risk_manager
from mlops.drift_monitor import DriftMonitor
from mlops.feature_cache import FeatureCache

# Modern bridge imports (SOLID architecture)
from mlops.bridge import (
    get_service_container,
    get_inference_engine_v2,
    get_risk_check_chain,
    get_ensemble_strategy_registry,
    get_repository_factory,
    get_daily_stats_repository,
    get_trade_log_repository,
    health_check as bridge_health_check,
)

__all__ = [
    # Legacy (backward compatible)
    'MLOpsConfig',
    'RiskLimits',
    'TradingHours',
    'InferenceEngine',
    'get_inference_engine',
    'RiskManager',
    'get_risk_manager',
    'DriftMonitor',
    'FeatureCache',
    # Modern (SOLID architecture)
    'get_service_container',
    'get_inference_engine_v2',
    'get_risk_check_chain',
    'get_ensemble_strategy_registry',
    'get_repository_factory',
    'get_daily_stats_repository',
    'get_trade_log_repository',
    'bridge_health_check',
]

__version__ = '2.0.0'
