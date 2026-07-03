"""
Model management module for USD/COP RL Trading System.

This module provides components for managing, loading, and running
inference on reinforcement learning models.

Classes:
    ModelRegistry: Central registry for managing multiple RL models
    ModelLoader: Loads models from various sources (file, MinIO, ONNX)
    ModelConfig: Configuration dataclass for model metadata

ORM Models (from src.models.orm):
    Base: SQLAlchemy declarative base
    Trade: Trading action records with execution details
    Prediction: Model inference results with observation context
    RiskEvent: Risk management events and limit breaches
    FeatureSnapshot: Point-in-time feature snapshots for reproducibility

Submodules:
    validation: Model validation utilities including smoke tests
    orm: SQLAlchemy ORM models for database persistence

Note:
    InferenceEngine has been consolidated to src/inference/inference_engine.py
    For inference operations, use:
        >>> from src.inference import InferenceEngine

Example usage:
    >>> from src.models import ModelRegistry, ModelConfig
    >>> from src.inference import InferenceEngine  # Canonical location
    >>>
    >>> # Initialize registry
    >>> registry = ModelRegistry(
    ...     db_connection="postgresql://...",
    ...     model_storage_path="/models"
    ... )
    >>>
    >>> # Get production model
    >>> prod_model = registry.get_production_model()
    >>> print(f"Production: {prod_model.model_id} v{prod_model.version}")
    >>>
    >>> # Run smoke test before deployment
    >>> from src.models.validation import run_smoke_test
    >>> result = run_smoke_test("models:/my_model/Production")
    >>> if result.passed:
    ...     print("Model ready for deployment!")
    >>>
    >>> # Use ORM models for database operations
    >>> from src.models.orm import Trade, Prediction, RiskEvent, FeatureSnapshot
"""

# Import registry which doesn't depend on ONNX
from .model_registry import ModelRegistry, ModelConfig

# Lazy imports for ONNX-dependent modules to avoid crash on Windows
# with onnxruntime 1.22.x + Python 3.12 access violation bug
_ModelLoader = None


def _lazy_import_loader():
    """Lazy import for ModelLoader to avoid ONNX crash on module load."""
    global _ModelLoader
    if _ModelLoader is None:
        try:
            from .model_loader import ModelLoader
            _ModelLoader = ModelLoader
        except Exception as e:
            import logging
            logging.warning(f"ModelLoader import failed: {e}")
            _ModelLoader = None
    return _ModelLoader


# For direct attribute access, use __getattr__
def __getattr__(name):
    if name == 'ModelLoader':
        return _lazy_import_loader()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'ModelRegistry',
    'ModelConfig',
    'ModelLoader',
    # Note: InferenceEngine moved to src/inference/
    # Import from there: from src.inference import InferenceEngine
]

__version__ = '1.1.0'
