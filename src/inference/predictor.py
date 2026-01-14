"""
Predictor
=========

Single Responsibility: Execute inference on observations.

Author: Trading Team
Version: 1.0.0
Date: 2025-01-14
"""

import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from threading import Lock

import numpy as np

from src.core.interfaces.inference import (
    IPredictor,
    InferenceResult,
    SignalType,
)
from .model_loader import ONNXModelLoader

logger = logging.getLogger(__name__)


class ONNXPredictor(IPredictor):
    """
    ONNX model predictor.

    Single Responsibility: Execute inference and return results.
    Thread-safe for concurrent access.
    """

    def __init__(
        self,
        loader: ONNXModelLoader,
        weight: float = 1.0
    ):
        """
        Args:
            loader: Model loader with loaded model
            weight: Model weight for ensemble (default: 1.0)
        """
        self._loader = loader
        self._weight = weight
        self._lock = Lock()
        self._inference_count = 0
        self._total_latency_ms = 0.0

    @property
    def model_name(self) -> str:
        return self._loader.name

    @property
    def weight(self) -> float:
        return self._weight

    def predict(self, observation: np.ndarray) -> InferenceResult:
        """
        Run inference on observation.

        Args:
            observation: Feature vector (1D or 2D array)

        Returns:
            InferenceResult with signal, confidence, and probabilities
        """
        if not self._loader.is_loaded():
            raise RuntimeError(f"Model '{self.model_name}' not loaded")

        with self._lock:
            start_time = time.perf_counter()

            # Ensure correct shape and dtype
            if observation.ndim == 1:
                observation = observation.reshape(1, -1)
            observation = observation.astype(np.float32)

            # Run inference
            session = self._loader.session
            outputs = session.run(
                None,
                {self._loader.input_name: observation}
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Parse outputs
            action_probs = outputs[0][0]  # First output, first batch

            # Handle different output formats
            if len(outputs) > 1:
                confidence = float(outputs[1][0][0])
            else:
                confidence = float(np.max(action_probs))

            # Map action to signal
            action_idx = int(np.argmax(action_probs))
            signal_map = {0: SignalType.HOLD, 1: SignalType.BUY, 2: SignalType.SELL}
            signal = signal_map.get(action_idx, SignalType.HOLD)

            # Track metrics
            self._inference_count += 1
            self._total_latency_ms += latency_ms

            return InferenceResult(
                signal=signal,
                confidence=confidence,
                action_probs={
                    "HOLD": float(action_probs[0]),
                    "BUY": float(action_probs[1]) if len(action_probs) > 1 else 0.0,
                    "SELL": float(action_probs[2]) if len(action_probs) > 2 else 0.0,
                },
                model_name=self.model_name,
                latency_ms=latency_ms,
                timestamp=datetime.now().isoformat(),
            )

    @property
    def avg_latency_ms(self) -> float:
        """Get average inference latency."""
        if self._inference_count == 0:
            return 0.0
        return self._total_latency_ms / self._inference_count

    @property
    def inference_count(self) -> int:
        """Get total inference count."""
        return self._inference_count

    @property
    def stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        return {
            "model_name": self.model_name,
            "weight": self._weight,
            "inference_count": self._inference_count,
            "avg_latency_ms": self.avg_latency_ms,
            "total_latency_ms": self._total_latency_ms,
        }

    def reset_stats(self) -> None:
        """Reset inference statistics."""
        with self._lock:
            self._inference_count = 0
            self._total_latency_ms = 0.0
