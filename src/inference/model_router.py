"""
Model Router - Shadow Mode Implementation
==========================================

Router that executes champion and shadow models in parallel for A/B testing
and gradual model rollout. The champion model's prediction is used for actual
trading decisions while shadow model predictions are logged for comparison.

MLOps-3: Shadow Mode for Models

Features:
- Parallel execution of champion and shadow models
- Hot reloading without service restart
- Agreement tracking between models
- Comprehensive metrics for monitoring

Author: Trading Team
Version: 1.0.0
Date: 2025-01-14
"""

import time
import logging
import threading
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None
    MlflowClient = None

try:
    from prometheus_client import Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# Prometheus Metrics
# =============================================================================

if PROMETHEUS_AVAILABLE:
    PREDICTIONS_TOTAL = Counter(
        'model_predictions_total',
        'Total predictions by model type and action',
        ['model_type', 'action']
    )
    SHADOW_AGREEMENT = Counter(
        'shadow_agreement_total',
        'Agreement between champion and shadow models',
        ['agree']
    )
    PREDICTION_LATENCY = Histogram(
        'prediction_latency_seconds',
        'Prediction latency by model type',
        ['model_type'],
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0)
    )
    MODEL_LOADED = Gauge(
        'model_loaded',
        'Indicates if a model is loaded (1) or not (0)',
        ['model_type', 'model_version']
    )
    SHADOW_DIVERGENCE_RATE = Gauge(
        'shadow_divergence_rate',
        'Rate of divergence between champion and shadow over last N predictions'
    )
else:
    # Mock metrics for when Prometheus is not available
    class MockMetric:
        def labels(self, *args, **kwargs):
            return self
        def inc(self, *args, **kwargs):
            pass
        def observe(self, *args, **kwargs):
            pass
        def set(self, *args, **kwargs):
            pass

    PREDICTIONS_TOTAL = MockMetric()
    SHADOW_AGREEMENT = MockMetric()
    PREDICTION_LATENCY = MockMetric()
    MODEL_LOADED = MockMetric()
    SHADOW_DIVERGENCE_RATE = MockMetric()


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PredictionResult:
    """Result from a single model prediction."""
    action: float
    confidence: float
    signal: str  # "LONG", "SHORT", "HOLD"
    latency_ms: float
    model_name: str
    model_version: str
    timestamp: str


@dataclass
class RouterPrediction:
    """Complete prediction result from the router."""
    champion: PredictionResult
    shadow: Optional[PredictionResult]
    agree: bool
    divergence: float  # Absolute difference in actions
    champion_used: bool  # Always True in normal operation


# =============================================================================
# Model Wrapper
# =============================================================================

class ModelWrapper:
    """
    Wraps a model loaded from MLflow or local storage.
    Provides consistent interface for prediction regardless of source.
    """

    def __init__(
        self,
        model: Any,
        name: str,
        version: str,
        stage: str,
        threshold_long: float = 0.33,
        threshold_short: float = -0.33
    ):
        self.model = model
        self.name = name
        self.version = version
        self.stage = stage
        self.threshold_long = threshold_long
        self.threshold_short = threshold_short
        self.loaded_at = datetime.now(timezone.utc)
        self._lock = threading.Lock()

    def predict(self, observation: np.ndarray) -> Tuple[float, float]:
        """
        Run prediction on observation.

        Args:
            observation: Feature vector

        Returns:
            Tuple of (action, confidence)
        """
        with self._lock:
            # Ensure correct shape
            if observation.ndim == 1:
                observation = observation.reshape(1, -1)

            # Handle different model types
            if hasattr(self.model, 'predict'):
                # Stable Baselines / sklearn style
                if hasattr(self.model, 'policy'):
                    # PPO model
                    action, _ = self.model.predict(observation, deterministic=True)
                else:
                    # sklearn style
                    action = self.model.predict(observation)
            else:
                # Direct callable
                action = self.model(observation)

            # Extract scalar
            action_value = float(action[0]) if hasattr(action, '__len__') else float(action)

            # Confidence as magnitude
            confidence = min(abs(action_value), 1.0)

            return action_value, confidence

    def get_signal(self, action: float) -> str:
        """Convert continuous action to discrete signal."""
        if action > self.threshold_long:
            return "LONG"
        elif action < self.threshold_short:
            return "SHORT"
        return "HOLD"

    @property
    def info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": self.name,
            "version": self.version,
            "stage": self.stage,
            "loaded_at": self.loaded_at.isoformat(),
            "threshold_long": self.threshold_long,
            "threshold_short": self.threshold_short,
        }


# =============================================================================
# Model Router
# =============================================================================

class ModelRouter:
    """
    Routes inference to champion + shadow models.

    The router executes both models in parallel but only returns the
    champion model's prediction for actual use. Shadow predictions are
    logged for comparison and monitoring purposes.

    Features:
    - Hot reload models without restart
    - Track agreement between champion and shadow
    - Prometheus metrics for monitoring
    - Thread-safe model access

    Usage:
        router = ModelRouter(mlflow_uri="http://localhost:5000")
        result = router.predict(observation)

        if not result.agree:
            logger.warning(f"Models diverged: {result.divergence}")
    """

    # Agreement window size for calculating divergence rate
    AGREEMENT_WINDOW = 1000

    def __init__(
        self,
        mlflow_uri: str = "http://localhost:5000",
        champion_stage: str = "Production",
        shadow_stage: str = "Staging",
        enable_shadow: bool = True,
        model_name: str = "ppo_usdcop"
    ):
        """
        Initialize the model router.

        Args:
            mlflow_uri: MLflow tracking server URI
            shadow_stage: Stage to load shadow model from
            enable_shadow: Whether to run shadow predictions
            model_name: Registered model name in MLflow
        """
        self.mlflow_uri = mlflow_uri
        self.champion_stage = champion_stage
        self.shadow_stage = shadow_stage
        self.enable_shadow = enable_shadow
        self.model_name = model_name

        self._champion: Optional[ModelWrapper] = None
        self._shadow: Optional[ModelWrapper] = None
        self._client: Optional[MlflowClient] = None
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="model_router")

        # Agreement tracking
        self._agreement_history: list = []
        self._total_predictions = 0
        self._total_agreements = 0

        # Initialize MLflow client
        if MLFLOW_AVAILABLE:
            try:
                mlflow.set_tracking_uri(mlflow_uri)
                self._client = MlflowClient(mlflow_uri)
                logger.info(f"MLflow client initialized: {mlflow_uri}")
            except Exception as e:
                logger.warning(f"Could not initialize MLflow client: {e}")

        # Load models
        self._load_models()

    def _load_model(self, stage: str) -> Optional[ModelWrapper]:
        """
        Load a model from MLflow by stage.

        Args:
            stage: Model stage ("Production" or "Staging")

        Returns:
            ModelWrapper or None if loading fails
        """
        if not MLFLOW_AVAILABLE or self._client is None:
            logger.warning(f"MLflow not available, cannot load {stage} model")
            return None

        try:
            # Get latest version for stage
            versions = self._client.get_latest_versions(
                self.model_name,
                stages=[stage]
            )

            if not versions:
                logger.warning(f"No model found for stage: {stage}")
                return None

            version_info = versions[0]
            model_version = version_info.version
            run_id = version_info.run_id

            # Load model
            model_uri = f"models:/{self.model_name}/{stage}"
            model = mlflow.pyfunc.load_model(model_uri)

            wrapper = ModelWrapper(
                model=model,
                name=self.model_name,
                version=model_version,
                stage=stage,
            )

            # Update metrics
            MODEL_LOADED.labels(
                model_type=stage.lower(),
                model_version=model_version
            ).set(1)

            logger.info(
                f"Loaded {stage} model: {self.model_name} v{model_version}"
            )

            return wrapper

        except Exception as e:
            logger.error(f"Failed to load {stage} model: {e}")
            return None

    def _load_models(self) -> None:
        """Load champion and shadow models."""
        with self._lock:
            self._champion = self._load_model(self.champion_stage)

            if self.enable_shadow:
                self._shadow = self._load_model(self.shadow_stage)

    def reload_models(self) -> Dict[str, Any]:
        """
        Hot reload both models.

        Returns:
            Dict with reload status for each model
        """
        result = {
            "champion": {"success": False, "version": None},
            "shadow": {"success": False, "version": None},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with self._lock:
            # Clear old metrics
            if self._champion:
                MODEL_LOADED.labels(
                    model_type="production",
                    model_version=self._champion.version
                ).set(0)
            if self._shadow:
                MODEL_LOADED.labels(
                    model_type="staging",
                    model_version=self._shadow.version
                ).set(0)

            # Reload champion
            old_champion = self._champion
            new_champion = self._load_model(self.champion_stage)

            if new_champion:
                self._champion = new_champion
                result["champion"] = {
                    "success": True,
                    "version": new_champion.version,
                    "previous_version": old_champion.version if old_champion else None,
                }

            # Reload shadow
            if self.enable_shadow:
                old_shadow = self._shadow
                new_shadow = self._load_model(self.shadow_stage)

                if new_shadow:
                    self._shadow = new_shadow
                    result["shadow"] = {
                        "success": True,
                        "version": new_shadow.version,
                        "previous_version": old_shadow.version if old_shadow else None,
                    }

        logger.info(f"Model reload completed: {result}")
        return result

    def load_local_models(
        self,
        champion_model: Any,
        champion_version: str = "local",
        shadow_model: Optional[Any] = None,
        shadow_version: str = "local-shadow"
    ) -> None:
        """
        Load models from local Python objects (for testing or non-MLflow usage).

        Args:
            champion_model: Champion model object
            champion_version: Version string for champion
            shadow_model: Optional shadow model object
            shadow_version: Version string for shadow
        """
        with self._lock:
            self._champion = ModelWrapper(
                model=champion_model,
                name="local_model",
                version=champion_version,
                stage="Production",
            )
            MODEL_LOADED.labels(model_type="production", model_version=champion_version).set(1)

            if shadow_model is not None:
                self._shadow = ModelWrapper(
                    model=shadow_model,
                    name="local_model",
                    version=shadow_version,
                    stage="Staging",
                )
                MODEL_LOADED.labels(model_type="staging", model_version=shadow_version).set(1)

            logger.info(f"Loaded local models: champion={champion_version}, shadow={shadow_version}")

    def _predict_single(
        self,
        model: ModelWrapper,
        observation: np.ndarray,
        model_type: str
    ) -> PredictionResult:
        """Run prediction on a single model with timing."""
        start_time = time.perf_counter()

        action, confidence = model.predict(observation)
        signal = model.get_signal(action)

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Record metrics
        PREDICTIONS_TOTAL.labels(model_type=model_type, action=signal).inc()
        PREDICTION_LATENCY.labels(model_type=model_type).observe(latency_ms / 1000)

        return PredictionResult(
            action=action,
            confidence=confidence,
            signal=signal,
            latency_ms=latency_ms,
            model_name=model.name,
            model_version=model.version,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def predict(self, observation: np.ndarray) -> RouterPrediction:
        """
        Run inference on champion (and optionally shadow) model.

        The champion model's result is always returned for actual use.
        Shadow predictions are logged for comparison only.

        Args:
            observation: Feature vector (1D or 2D numpy array)

        Returns:
            RouterPrediction with champion and optional shadow results

        Raises:
            RuntimeError: If champion model is not loaded
        """
        with self._lock:
            if self._champion is None:
                raise RuntimeError("Champion model not loaded")

            champion_result: Optional[PredictionResult] = None
            shadow_result: Optional[PredictionResult] = None

            # Execute in parallel if shadow is enabled
            if self.enable_shadow and self._shadow is not None:
                futures = {}

                # Submit both predictions
                futures["champion"] = self._executor.submit(
                    self._predict_single,
                    self._champion,
                    observation.copy(),  # Copy to avoid race conditions
                    "champion"
                )
                futures["shadow"] = self._executor.submit(
                    self._predict_single,
                    self._shadow,
                    observation.copy(),
                    "shadow"
                )

                # Collect results
                for name, future in futures.items():
                    try:
                        result = future.result(timeout=1.0)  # 1 second timeout
                        if name == "champion":
                            champion_result = result
                        else:
                            shadow_result = result
                    except Exception as e:
                        logger.error(f"Error in {name} prediction: {e}")
                        if name == "champion":
                            # Champion failed, try synchronously
                            champion_result = self._predict_single(
                                self._champion, observation, "champion"
                            )
            else:
                # Just run champion
                champion_result = self._predict_single(
                    self._champion, observation, "champion"
                )

            # Calculate agreement
            agree = True
            divergence = 0.0

            if shadow_result is not None:
                agree = champion_result.signal == shadow_result.signal
                divergence = abs(champion_result.action - shadow_result.action)

                # Track agreement
                SHADOW_AGREEMENT.labels(agree=str(agree).lower()).inc()

                self._agreement_history.append(agree)
                self._total_predictions += 1
                if agree:
                    self._total_agreements += 1

                # Maintain window size
                if len(self._agreement_history) > self.AGREEMENT_WINDOW:
                    self._agreement_history.pop(0)

                # Update divergence rate gauge
                recent_agreements = sum(self._agreement_history)
                divergence_rate = 1.0 - (recent_agreements / len(self._agreement_history))
                SHADOW_DIVERGENCE_RATE.set(divergence_rate)

                # Log divergence for analysis
                if not agree:
                    logger.info(
                        f"Model divergence: champion={champion_result.signal} "
                        f"({champion_result.action:.4f}), shadow={shadow_result.signal} "
                        f"({shadow_result.action:.4f}), diff={divergence:.4f}"
                    )

            return RouterPrediction(
                champion=champion_result,
                shadow=shadow_result,
                agree=agree,
                divergence=divergence,
                champion_used=True,
            )

    @property
    def champion(self) -> Optional[ModelWrapper]:
        """Get champion model wrapper."""
        return self._champion

    @property
    def shadow(self) -> Optional[ModelWrapper]:
        """Get shadow model wrapper."""
        return self._shadow

    @property
    def is_ready(self) -> bool:
        """Check if router is ready for predictions."""
        return self._champion is not None

    @property
    def agreement_rate(self) -> float:
        """Get overall agreement rate between champion and shadow."""
        if self._total_predictions == 0:
            return 1.0
        return self._total_agreements / self._total_predictions

    @property
    def recent_agreement_rate(self) -> float:
        """Get recent agreement rate (last AGREEMENT_WINDOW predictions)."""
        if not self._agreement_history:
            return 1.0
        return sum(self._agreement_history) / len(self._agreement_history)

    def get_status(self) -> Dict[str, Any]:
        """Get router status and statistics."""
        status = {
            "ready": self.is_ready,
            "champion": self._champion.info if self._champion else None,
            "shadow": self._shadow.info if self._shadow else None,
            "shadow_enabled": self.enable_shadow,
            "statistics": {
                "total_predictions": self._total_predictions,
                "agreement_rate": round(self.agreement_rate, 4),
                "recent_agreement_rate": round(self.recent_agreement_rate, 4),
                "agreement_window": self.AGREEMENT_WINDOW,
            },
            "mlflow_uri": self.mlflow_uri,
            "model_name": self.model_name,
        }
        return status

    def shutdown(self) -> None:
        """Shutdown the router and cleanup resources."""
        self._executor.shutdown(wait=True)
        logger.info("ModelRouter shutdown complete")


# =============================================================================
# Factory Function
# =============================================================================

def create_model_router(
    mlflow_uri: str = "http://localhost:5000",
    enable_shadow: bool = True,
    model_name: str = "ppo_usdcop"
) -> ModelRouter:
    """
    Factory function to create a configured ModelRouter.

    Args:
        mlflow_uri: MLflow tracking URI
        enable_shadow: Enable shadow model execution
        model_name: MLflow registered model name

    Returns:
        Configured ModelRouter instance
    """
    return ModelRouter(
        mlflow_uri=mlflow_uri,
        enable_shadow=enable_shadow,
        model_name=model_name,
    )
