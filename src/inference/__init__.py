"""
Inference Module
================

Refactored inference components following SRP.

Components:
- ModelLoader: Load and warm up models
- Predictor: Single model inference
- EnsemblePredictor: Coordinate multiple models
- InferenceEngine: Facade for backward compatibility

Author: Trading Team
Version: 2.0.0
Date: 2025-01-14
"""

from .model_loader import ONNXModelLoader
from .predictor import ONNXPredictor
from .ensemble_predictor import EnsemblePredictor
from .inference_engine import InferenceEngine

__all__ = [
    'ONNXModelLoader',
    'ONNXPredictor',
    'EnsemblePredictor',
    'InferenceEngine',
]
