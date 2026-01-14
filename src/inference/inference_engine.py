"""
Inference Engine (Facade)
=========================

Facade Pattern: Provides unified interface for backward compatibility.
Delegates to ModelLoader, Predictor, and EnsemblePredictor.

Author: Trading Team
Version: 2.0.0
Date: 2025-01-14
"""

import os
import logging
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

import numpy as np

from src.core.interfaces.inference import (
    IInferenceEngine,
    IHealthChecker,
    InferenceResult,
    EnsembleResult,
    SignalType,
)
from src.core.strategies.ensemble_strategies import EnsembleStrategyRegistry
from .model_loader import ONNXModelLoader
from .predictor import ONNXPredictor
from .ensemble_predictor import EnsemblePredictor

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    onnx_path: str
    observation_dim: int = 45
    weight: float = 1.0
    enabled: bool = True
    algorithm: str = "PPO"


class InferenceEngine(IInferenceEngine):
    """
    Inference Engine Facade.

    Facade Pattern: Unified interface that delegates to:
    - ONNXModelLoader: Model loading
    - ONNXPredictor: Single model inference
    - EnsemblePredictor: Multi-model ensemble

    Maintains backward compatibility with existing API.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        ensemble_strategy: str = "weighted_average",
    ):
        """
        Args:
            config: MLOpsConfig or similar with models list
            ensemble_strategy: Strategy name for ensemble
        """
        self._config = config
        self._ensemble_strategy = ensemble_strategy

        # Internal components
        self._loaders: Dict[str, ONNXModelLoader] = {}
        self._predictors: Dict[str, ONNXPredictor] = {}
        self._ensemble: Optional[EnsemblePredictor] = None
        self._loaded = False

    @property
    def model_name(self) -> str:
        """Get primary model name."""
        if self._predictors:
            return next(iter(self._predictors.keys()))
        return "none"

    def load_models(self, providers: Optional[List[str]] = None) -> bool:
        """
        Load all configured models.

        Args:
            providers: Execution providers

        Returns:
            True if at least one model loaded
        """
        providers = providers or ['CPUExecutionProvider']

        if not self._config or not hasattr(self._config, 'models'):
            logger.warning("No models configured")
            return False

        for model_config in self._config.models:
            if not model_config.enabled:
                logger.info(f"Skipping disabled model: {model_config.name}")
                continue

            try:
                # Create loader
                loader = ONNXModelLoader(name=model_config.name)

                if loader.load(model_config.onnx_path, providers):
                    loader.warmup()

                    # Create predictor
                    predictor = ONNXPredictor(
                        loader=loader,
                        weight=getattr(model_config, 'weight', 1.0)
                    )

                    self._loaders[model_config.name] = loader
                    self._predictors[model_config.name] = predictor

                    logger.info(f"Loaded model: {model_config.name}")

            except Exception as e:
                logger.error(f"Failed to load {model_config.name}: {e}")

        self._loaded = len(self._predictors) > 0

        if self._loaded:
            # Create ensemble predictor
            self._ensemble = EnsemblePredictor(
                predictors=self._predictors,
                strategy_name=self._ensemble_strategy,
            )

        return self._loaded

    def load_single_model(
        self,
        name: str,
        onnx_path: str,
        observation_dim: int = 45,
        weight: float = 1.0,
        providers: Optional[List[str]] = None,
    ) -> bool:
        """
        Load a single model manually.

        Args:
            name: Model identifier
            onnx_path: Path to ONNX file
            observation_dim: Input dimension
            weight: Model weight for ensemble
            providers: Execution providers

        Returns:
            True if loaded successfully
        """
        providers = providers or ['CPUExecutionProvider']

        try:
            loader = ONNXModelLoader(name=name)

            if loader.load(onnx_path, providers):
                loader.warmup()

                predictor = ONNXPredictor(loader=loader, weight=weight)

                self._loaders[name] = loader
                self._predictors[name] = predictor

                # Update ensemble
                if self._ensemble:
                    self._ensemble.add_predictor(predictor)
                else:
                    self._ensemble = EnsemblePredictor(
                        predictors=self._predictors,
                        strategy_name=self._ensemble_strategy,
                    )

                self._loaded = True
                logger.info(f"Loaded model: {name}")
                return True

        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")

        return False

    def predict(
        self,
        observation: np.ndarray,
        model_name: Optional[str] = None
    ) -> InferenceResult:
        """
        Run inference with a specific model.

        Args:
            observation: Feature vector
            model_name: Specific model to use (uses first if None)

        Returns:
            InferenceResult with signal and confidence
        """
        if not self._loaded:
            raise RuntimeError("No models loaded. Call load_models() first.")

        if model_name:
            if model_name not in self._predictors:
                raise ValueError(f"Model not found: {model_name}")
            predictor = self._predictors[model_name]
        else:
            predictor = next(iter(self._predictors.values()))

        return predictor.predict(observation)

    def predict_ensemble(self, observation: np.ndarray) -> EnsembleResult:
        """
        Run ensemble inference with all models.

        Args:
            observation: Feature vector

        Returns:
            EnsembleResult with combined signal
        """
        if not self._loaded or not self._ensemble:
            raise RuntimeError("No models loaded. Call load_models() first.")

        return self._ensemble.predict(observation)

    @property
    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._loaded

    @property
    def is_healthy(self) -> bool:
        """Quick health check."""
        return self._loaded and len(self._predictors) > 0

    @property
    def model_names(self) -> List[str]:
        """Get list of loaded model names."""
        return list(self._predictors.keys())

    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        return {
            "loaded": self._loaded,
            "model_count": len(self._predictors),
            "ensemble_strategy": self._ensemble_strategy,
            "models": {
                name: pred.stats
                for name, pred in self._predictors.items()
            },
        }

    def health_check(self) -> Dict[str, Any]:
        """
        Run health check on inference engine.

        Returns:
            Health status dictionary
        """
        health = {
            "status": "healthy" if self._loaded else "unhealthy",
            "loaded_models": len(self._predictors),
            "ensemble_strategy": self._ensemble_strategy,
            "models": {}
        }

        # Test each predictor
        for name, predictor in self._predictors.items():
            try:
                loader = self._loaders[name]
                obs_dim = loader.input_shape[-1] if loader.input_shape else 45
                dummy = np.random.randn(obs_dim).astype(np.float32)

                result = predictor.predict(dummy)

                health["models"][name] = {
                    "status": "healthy",
                    "latency_ms": result.latency_ms,
                    "inference_count": predictor.inference_count,
                }
            except Exception as e:
                health["models"][name] = {
                    "status": "unhealthy",
                    "error": str(e),
                }
                health["status"] = "degraded"

        return health

    def set_ensemble_strategy(self, strategy_name: str) -> None:
        """
        Change ensemble strategy.

        Args:
            strategy_name: Strategy name from registry
        """
        self._ensemble_strategy = strategy_name

        if self._ensemble:
            self._ensemble.set_strategy_by_name(strategy_name)

        logger.info(f"Changed ensemble strategy to: {strategy_name}")

    def get_available_strategies(self) -> List[str]:
        """Get list of available ensemble strategies."""
        return EnsembleStrategyRegistry.list_strategies()
