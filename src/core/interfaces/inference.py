"""
Inference Interfaces
====================

Defines abstract interfaces for the inference subsystem following ISP.
Splits InferenceEngine responsibilities into focused interfaces.

Author: Trading Team
Version: 2.0.0
Date: 2025-01-14
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum

import numpy as np


class SignalType(str, Enum):
    """Trading signal types."""
    HOLD = "HOLD"
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class InferenceResult:
    """Result of model inference."""
    signal: SignalType
    confidence: float
    action_probs: Dict[str, float]
    model_name: str
    latency_ms: float
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal": self.signal.value,
            "confidence": self.confidence,
            "action_probs": self.action_probs,
            "model_name": self.model_name,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
        }


# =============================================================================
# Interface Segregation: Split InferenceEngine into focused interfaces
# =============================================================================

class IModelLoader(ABC):
    """
    Interface for model loading operations.

    Single Responsibility: Load and initialize model artifacts.
    """

    @abstractmethod
    def load(self, path: str, providers: Optional[List[str]] = None) -> bool:
        """
        Load model from path.

        Args:
            path: Path to model file (ONNX, etc.)
            providers: Execution providers (e.g., CPUExecutionProvider)

        Returns:
            True if loaded successfully
        """
        pass

    @abstractmethod
    def warmup(self, iterations: int = 10) -> None:
        """
        Warm up model with dummy inference.

        Args:
            iterations: Number of warmup iterations
        """
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        pass

    @property
    @abstractmethod
    def input_shape(self) -> Tuple[int, ...]:
        """Get expected input shape."""
        pass


class IPredictor(ABC):
    """
    Interface for running model predictions.

    Single Responsibility: Execute inference on observations.
    """

    @abstractmethod
    def predict(self, observation: np.ndarray) -> InferenceResult:
        """
        Run inference on observation.

        Args:
            observation: Feature vector (1D or 2D array)

        Returns:
            InferenceResult with signal, confidence, and probabilities
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get model name identifier."""
        pass


class IEnsembleStrategy(ABC):
    """
    Interface for ensemble combination strategies.

    Strategy Pattern: Allows different ensemble methods to be plugged in.
    Open/Closed: Add new strategies without modifying existing code.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get strategy name identifier."""
        pass

    @abstractmethod
    def combine(
        self,
        results: List[InferenceResult],
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[Dict[str, float], SignalType, float]:
        """
        Combine multiple inference results.

        Args:
            results: List of individual model results
            weights: Optional model weights (model_name -> weight)

        Returns:
            Tuple of (combined_probs, signal, confidence)
        """
        pass


class IHealthChecker(ABC):
    """
    Interface for health check operations.

    Single Responsibility: System health monitoring.
    """

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        Run health check.

        Returns:
            Dictionary with health status and details
        """
        pass

    @property
    @abstractmethod
    def is_healthy(self) -> bool:
        """Quick health status check."""
        pass


class IInferenceEngine(IPredictor, IHealthChecker, ABC):
    """
    Combined interface for inference engine.

    Facade Pattern: Provides unified interface for inference operations.
    Implements IPredictor and IHealthChecker through composition.
    """

    @abstractmethod
    def load_models(self, providers: Optional[List[str]] = None) -> bool:
        """Load all configured models."""
        pass

    @abstractmethod
    def predict_ensemble(self, observation: np.ndarray) -> 'EnsembleResult':
        """Run ensemble inference."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        pass

    @property
    @abstractmethod
    def model_names(self) -> List[str]:
        """Get list of loaded model names."""
        pass


@dataclass
class EnsembleResult:
    """Result of ensemble inference."""
    signal: SignalType
    confidence: float
    action_probs: Dict[str, float]
    individual_results: List[InferenceResult]
    ensemble_strategy: str
    total_latency_ms: float
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal": self.signal.value,
            "confidence": self.confidence,
            "action_probs": self.action_probs,
            "individual_results": [r.to_dict() for r in self.individual_results],
            "ensemble_strategy": self.ensemble_strategy,
            "total_latency_ms": self.total_latency_ms,
            "timestamp": self.timestamp,
        }
