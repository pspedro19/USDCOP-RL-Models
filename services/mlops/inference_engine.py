"""
ONNX Inference Engine
=====================

High-performance inference engine using ONNX Runtime.
Optimized for low-latency trading signal generation.

Features:
- Sub-5ms inference latency
- Multi-model ensemble support
- Automatic model warming
- Thread-safe inference
- Performance metrics tracking
"""

import os
import time
import logging
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from threading import Lock
import json

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None
    logging.warning("onnxruntime not installed. Install with: pip install onnxruntime")

from .config import MLOpsConfig, get_config, SignalType, ModelConfig

logger = logging.getLogger(__name__)


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


class ModelSession:
    """Wrapper for an ONNX Runtime session with metadata."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: str = ""
        self.output_names: List[str] = []
        self._lock = Lock()
        self._inference_count = 0
        self._total_latency_ms = 0.0

    def load(self, providers: List[str] = None):
        """Load the ONNX model."""
        if ort is None:
            raise ImportError("onnxruntime not installed")

        if not os.path.exists(self.config.onnx_path):
            raise FileNotFoundError(f"Model not found: {self.config.onnx_path}")

        providers = providers or ['CPUExecutionProvider']

        # Session options for optimization
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 2

        self.session = ort.InferenceSession(
            self.config.onnx_path,
            sess_options=sess_options,
            providers=providers
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        logger.info(f"Loaded model: {self.config.name} from {self.config.onnx_path}")
        logger.info(f"  Input: {self.input_name}, Outputs: {self.output_names}")

    def warmup(self, num_iterations: int = 10):
        """Warm up the model with dummy inference."""
        if self.session is None:
            raise RuntimeError("Model not loaded")

        logger.info(f"Warming up {self.config.name}...")

        dummy_input = np.random.randn(1, self.config.observation_dim).astype(np.float32)

        for _ in range(num_iterations):
            self.session.run(None, {self.input_name: dummy_input})

        logger.info(f"Warmup complete for {self.config.name}")

    def predict(self, observation: np.ndarray) -> InferenceResult:
        """Run inference on observation."""
        if self.session is None:
            raise RuntimeError("Model not loaded")

        with self._lock:
            start_time = time.perf_counter()

            # Ensure correct shape and dtype
            if observation.ndim == 1:
                observation = observation.reshape(1, -1)
            observation = observation.astype(np.float32)

            # Run inference
            outputs = self.session.run(None, {self.input_name: observation})

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
                model_name=self.config.name,
                latency_ms=latency_ms,
                timestamp=datetime.now().isoformat(),
            )

    @property
    def avg_latency_ms(self) -> float:
        if self._inference_count == 0:
            return 0.0
        return self._total_latency_ms / self._inference_count

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "model_name": self.config.name,
            "inference_count": self._inference_count,
            "avg_latency_ms": self.avg_latency_ms,
            "total_latency_ms": self._total_latency_ms,
        }


class InferenceEngine:
    """
    High-performance inference engine with ensemble support.

    Usage:
        engine = InferenceEngine(config)
        engine.load_models()

        result = engine.predict(observation)
        print(f"Signal: {result.signal}, Confidence: {result.confidence}")
    """

    def __init__(self, config: Optional[MLOpsConfig] = None):
        self.config = config or get_config()
        self.models: Dict[str, ModelSession] = {}
        self._loaded = False
        self._lock = Lock()

    def load_models(self, providers: List[str] = None):
        """Load all configured models."""
        providers = providers or ['CPUExecutionProvider']

        for model_config in self.config.models:
            if not model_config.enabled:
                logger.info(f"Skipping disabled model: {model_config.name}")
                continue

            try:
                session = ModelSession(model_config)
                session.load(providers)
                session.warmup()
                self.models[model_config.name] = session
                logger.info(f"✅ Loaded model: {model_config.name}")

            except Exception as e:
                logger.error(f"❌ Failed to load {model_config.name}: {e}")

        self._loaded = len(self.models) > 0

        if not self._loaded:
            logger.warning("No models loaded!")

        return self._loaded

    def load_single_model(
        self,
        name: str,
        onnx_path: str,
        observation_dim: int = 45,
        algorithm: str = "PPO"
    ):
        """Load a single model manually."""
        config = ModelConfig(
            name=name,
            algorithm=algorithm,
            onnx_path=onnx_path,
            observation_dim=observation_dim,
        )

        session = ModelSession(config)
        session.load()
        session.warmup()
        self.models[name] = session
        self._loaded = True

        logger.info(f"✅ Loaded single model: {name}")

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
            if model_name not in self.models:
                raise ValueError(f"Model not found: {model_name}")
            model = self.models[model_name]
        else:
            model = next(iter(self.models.values()))

        return model.predict(observation)

    def predict_ensemble(self, observation: np.ndarray) -> EnsembleResult:
        """
        Run inference with all models and combine results.

        Args:
            observation: Feature vector

        Returns:
            EnsembleResult with combined signal
        """
        if not self._loaded:
            raise RuntimeError("No models loaded. Call load_models() first.")

        start_time = time.perf_counter()
        individual_results = []

        # Get predictions from all models
        for name, model in self.models.items():
            try:
                result = model.predict(observation)
                individual_results.append(result)
            except Exception as e:
                logger.error(f"Inference failed for {name}: {e}")

        if not individual_results:
            raise RuntimeError("All model inferences failed")

        # Combine results based on strategy
        if self.config.ensemble_strategy == "weighted_average":
            combined_probs, signal, confidence = self._weighted_average(individual_results)
        else:
            combined_probs, signal, confidence = self._majority_vote(individual_results)

        total_latency = (time.perf_counter() - start_time) * 1000

        return EnsembleResult(
            signal=signal,
            confidence=confidence,
            action_probs=combined_probs,
            individual_results=individual_results,
            ensemble_strategy=self.config.ensemble_strategy,
            total_latency_ms=total_latency,
            timestamp=datetime.now().isoformat(),
        )

    def _weighted_average(
        self,
        results: List[InferenceResult]
    ) -> Tuple[Dict[str, float], SignalType, float]:
        """Combine results using weighted average of probabilities."""
        total_weight = 0.0
        combined = {"HOLD": 0.0, "BUY": 0.0, "SELL": 0.0}

        for result in results:
            model = self.models[result.model_name]
            weight = model.config.weight * result.confidence

            for action in combined:
                combined[action] += result.action_probs.get(action, 0.0) * weight

            total_weight += weight

        # Normalize
        if total_weight > 0:
            for action in combined:
                combined[action] /= total_weight

        # Get final signal
        signal_map = {"HOLD": SignalType.HOLD, "BUY": SignalType.BUY, "SELL": SignalType.SELL}
        best_action = max(combined, key=combined.get)
        signal = signal_map[best_action]
        confidence = combined[best_action]

        return combined, signal, confidence

    def _majority_vote(
        self,
        results: List[InferenceResult]
    ) -> Tuple[Dict[str, float], SignalType, float]:
        """Combine results using majority vote."""
        votes = {"HOLD": 0, "BUY": 0, "SELL": 0}

        for result in results:
            votes[result.signal.value] += 1

        # Get majority
        best_action = max(votes, key=votes.get)
        signal_map = {"HOLD": SignalType.HOLD, "BUY": SignalType.BUY, "SELL": SignalType.SELL}
        signal = signal_map[best_action]

        # Confidence is proportion of votes
        total_votes = sum(votes.values())
        confidence = votes[best_action] / total_votes if total_votes > 0 else 0.0

        # Normalized votes as probs
        probs = {k: v / total_votes for k, v in votes.items()}

        return probs, signal, confidence

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_names(self) -> List[str]:
        return list(self.models.keys())

    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics for all models."""
        return {
            "loaded": self._loaded,
            "model_count": len(self.models),
            "models": {name: model.stats for name, model in self.models.items()},
        }

    def health_check(self) -> Dict[str, Any]:
        """Run health check on inference engine."""
        health = {
            "status": "healthy" if self._loaded else "unhealthy",
            "loaded_models": len(self.models),
            "models": {}
        }

        # Test each model
        for name, model in self.models.items():
            try:
                dummy = np.random.randn(model.config.observation_dim).astype(np.float32)
                result = model.predict(dummy)
                health["models"][name] = {
                    "status": "healthy",
                    "latency_ms": result.latency_ms,
                }
            except Exception as e:
                health["models"][name] = {
                    "status": "unhealthy",
                    "error": str(e),
                }
                health["status"] = "degraded"

        return health


# Global engine instance
_engine: Optional[InferenceEngine] = None


def get_inference_engine() -> InferenceEngine:
    """Get or create global inference engine."""
    global _engine
    if _engine is None:
        _engine = InferenceEngine()
    return _engine


def initialize_engine(config: Optional[MLOpsConfig] = None) -> InferenceEngine:
    """Initialize and load the global inference engine."""
    global _engine
    _engine = InferenceEngine(config)
    _engine.load_models()
    return _engine
