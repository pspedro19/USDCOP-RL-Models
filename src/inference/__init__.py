"""
Inference Module - Single Source of Truth (SSOT)
=================================================

This module is the CANONICAL location for all inference-related components.
Do NOT import InferenceEngine from src.models - use this module instead.

Components:
- InferenceEngine: Unified facade for all inference operations (SSOT)
- ONNXModelLoader: Load and warm up ONNX models
- ONNXPredictor: Single model ONNX inference
- EnsemblePredictor: Coordinate multiple models with strategies
- ModelRouter: Shadow mode execution for A/B testing (MLOps-3)
- ShadowPnLTracker: Virtual PnL tracking for shadow mode
- ValidatedPredictor: Contract-validated model wrapper

Usage:
    >>> from src.inference import InferenceEngine
    >>> engine = InferenceEngine(config)
    >>> engine.load_models()
    >>> result = engine.predict(observation)

For architecture details, see ARCHITECTURE.md in this directory.

Author: Trading Team
Version: 2.4.0
Date: 2026-01-17
"""

from .model_loader import ONNXModelLoader
from .predictor import ONNXPredictor
from .ensemble_predictor import EnsemblePredictor, load_ensemble_from_multi_seed
from .inference_engine import InferenceEngine
from .model_router import (
    ModelRouter,
    ModelWrapper,
    PredictionResult,
    RouterPrediction,
    create_model_router,
)
from .shadow_pnl import (
    VirtualTrade,
    ShadowMetrics,
    ShadowPnLTracker,
)
from .validated_predictor import (
    ValidatedPredictor,
    PredictionStats,
    PredictableModel,
)

__all__ = [
    # Core inference
    'ONNXModelLoader',
    'ONNXPredictor',
    'EnsemblePredictor',
    'load_ensemble_from_multi_seed',
    'InferenceEngine',
    # Validated inference
    'ValidatedPredictor',
    'PredictionStats',
    'PredictableModel',
    # Shadow mode (MLOps-3)
    'ModelRouter',
    'ModelWrapper',
    'PredictionResult',
    'RouterPrediction',
    'create_model_router',
    # Shadow PnL tracking
    'VirtualTrade',
    'ShadowMetrics',
    'ShadowPnLTracker',
]
