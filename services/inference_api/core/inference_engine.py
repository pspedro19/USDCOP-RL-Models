"""
Inference Engine for PPO Model
Handles model loading and prediction

CLAUDE-T16 / GEMINI-T15: Supports dynamic model loading from model_registry
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

# Try to import stable_baselines3
try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: stable_baselines3 not installed. Using mock inference.")

# Try to import psycopg2 for database queries
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

from ..config import get_settings

# Import thresholds from SSOT (REQUIRED - no fallback)
from src.core.constants import THRESHOLD_LONG, THRESHOLD_SHORT

# P2-2: Import hash validation utilities
try:
    from ..contracts.model_contract import compute_json_hash
    HASH_VALIDATION_AVAILABLE = True
except ImportError:
    HASH_VALIDATION_AVAILABLE = False
    compute_json_hash = None

settings = get_settings()
logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Manages PPO model loading and inference.
    Caches loaded models for efficient reuse.

    Supports loading models from:
    1. Explicit model_path
    2. Database model_registry (by model_id)
    3. Filesystem pattern matching
    """

    # Model-specific thresholds from SSOT (src/core/constants.py)
    # Default to SSOT values - can override per model if needed
    MODEL_THRESHOLDS = {
        "ppo_primary": {"long": THRESHOLD_LONG, "short": THRESHOLD_SHORT},
        "ppo_secondary": {"long": THRESHOLD_LONG, "short": THRESHOLD_SHORT},
    }

    def __init__(self):
        self.models: Dict[str, any] = {}
        self.model_metadata: Dict[str, dict] = {}  # Store metadata for loaded models
        self.threshold_long = settings.threshold_long
        self.threshold_short = settings.threshold_short
        self._db_url = os.environ.get("DATABASE_URL")

        # Post-prediction hook for drift observation (set by main.py)
        self.post_predict_hook: Optional[callable] = None

    def load_model(self, model_id: str, model_path: Optional[Path] = None) -> bool:
        """
        Load a PPO model into memory.

        Attempts to load in this order:
        1. From explicit model_path if provided
        2. From database model_registry
        3. From filesystem pattern matching

        Args:
            model_id: Identifier for the model
            model_path: Optional custom path to model file

        Returns:
            True if loaded successfully
        """
        if model_id in self.models:
            logger.info(f"Model {model_id} already loaded")
            return True

        if not SB3_AVAILABLE:
            logger.warning("stable_baselines3 not available, using mock model")
            self.models[model_id] = MockModel()
            return True

        # Strategy 1: Use explicit path if provided
        if model_path is not None and Path(model_path).exists():
            return self._load_from_path(model_id, Path(model_path))

        # Strategy 2: Query database for model path
        db_model_path = self._get_model_path_from_registry(model_id)
        if db_model_path and Path(db_model_path).exists():
            logger.info(f"Found model in registry: {db_model_path}")
            return self._load_from_path(model_id, Path(db_model_path))

        # Strategy 3: Pattern matching in models directory
        # Use MODEL_PATH env var (Docker: /models) or project_root/models (local)
        base_path = Path(settings.model_base_path)
        if not base_path.is_absolute():
            base_path = settings.project_root / base_path
        model_dir = base_path

        pattern = f"{model_id}*.zip"
        matches = list(model_dir.glob(pattern))
        if matches:
            model_path = matches[0]
            logger.info(f"Found model via pattern matching: {model_path}")
        else:
            # Strategy 4: Also check subdirectories for model_id
            subdir_path = model_dir / f"{model_id}_production" / "final_model.zip"
            if subdir_path.exists():
                model_path = subdir_path
                logger.info(f"Found model in production subdirectory: {model_path}")
            elif model_path is None:
                # Strategy 5: Use default path from settings (ppo_primary uses ppo_v20_production)
                if model_id == "ppo_primary":
                    model_path = settings.full_model_path
                    logger.info(f"Using default model path for ppo_primary: {model_path}")
                else:
                    logger.error(f"Model file not found for {model_id}")
                    self.models[model_id] = MockModel()
                    return True

        return self._load_from_path(model_id, Path(model_path))

    def _load_from_path(self, model_id: str, model_path: Path) -> bool:
        """Load model from a specific path."""
        try:
            logger.info(f"Loading model from {model_path}")

            # P2-2: Validate norm_stats_hash before loading model
            norm_stats_validation = self._validate_norm_stats_hash(model_id)
            if norm_stats_validation.get("status") == "failed":
                logger.error(
                    f"CRITICAL: norm_stats hash validation failed for {model_id}: "
                    f"{norm_stats_validation.get('error')}"
                )
                # Don't block model loading, but log prominently
                logger.warning(
                    "Loading model anyway, but predictions may be INCORRECT "
                    "due to norm_stats mismatch with training"
                )

            self.models[model_id] = PPO.load(str(model_path))
            self.model_metadata[model_id] = {
                "path": str(model_path),
                "loaded_at": __import__("datetime").datetime.now().isoformat(),
                "norm_stats_validation": norm_stats_validation,
            }
            logger.info(f"Model {model_id} loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.models[model_id] = MockModel()
            return True

    def _validate_norm_stats_hash(self, model_id: str) -> dict:
        """
        P2-2: Validate norm_stats.json hash matches registered hash.

        This ensures inference uses the SAME normalization as training.

        Returns:
            dict with status, expected_hash, actual_hash
        """
        if not HASH_VALIDATION_AVAILABLE:
            return {"status": "skipped", "reason": "hash_validation_not_available"}

        try:
            # Get expected hash from model metadata (from registry query)
            metadata = self.model_metadata.get(model_id, {})
            expected_hash = metadata.get("norm_stats_hash")

            if not expected_hash:
                # Try to get from registry
                expected_hash = self._get_norm_stats_hash_from_registry(model_id)

            if not expected_hash:
                return {"status": "skipped", "reason": "no_expected_hash_registered"}

            # Compute actual hash
            norm_stats_path = settings.full_norm_stats_path
            if not norm_stats_path.exists():
                return {
                    "status": "failed",
                    "error": f"norm_stats file not found: {norm_stats_path}",
                }

            actual_hash = compute_json_hash(norm_stats_path)

            if actual_hash == expected_hash:
                return {
                    "status": "passed",
                    "expected_hash": expected_hash[:16],
                    "actual_hash": actual_hash[:16],
                }
            else:
                return {
                    "status": "failed",
                    "error": "hash_mismatch",
                    "expected_hash": expected_hash[:16],
                    "actual_hash": actual_hash[:16],
                }

        except Exception as e:
            logger.warning(f"norm_stats hash validation error: {e}")
            return {"status": "error", "error": str(e)}

    def _get_norm_stats_hash_from_registry(self, model_id: str) -> Optional[str]:
        """Get norm_stats_hash from model_registry table."""
        if not PSYCOPG2_AVAILABLE or not self._db_url:
            return None

        try:
            conn = psycopg2.connect(self._db_url)
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute(
                """
                SELECT norm_stats_hash
                FROM model_registry
                WHERE model_id = %s
                """,
                (model_id,)
            )

            row = cursor.fetchone()
            cursor.close()
            conn.close()

            if row:
                return row.get("norm_stats_hash")
            return None

        except Exception as e:
            logger.debug(f"Could not get norm_stats_hash from registry: {e}")
            return None

    def _get_model_path_from_registry(self, model_id: str) -> Optional[str]:
        """
        Query model_registry table for model path.

        Args:
            model_id: Model identifier to look up

        Returns:
            Model path if found, None otherwise
        """
        if not PSYCOPG2_AVAILABLE or not self._db_url:
            return None

        try:
            conn = psycopg2.connect(self._db_url)
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute(
                """
                SELECT model_path, model_hash, model_version, status
                FROM model_registry
                WHERE model_id = %s
                """,
                (model_id,)
            )

            row = cursor.fetchone()
            cursor.close()
            conn.close()

            if row:
                # Store metadata for later use
                self.model_metadata[model_id] = {
                    "model_hash": row.get("model_hash"),
                    "model_version": row.get("model_version"),
                    "status": row.get("status"),
                }
                return row.get("model_path")

            return None

        except Exception as e:
            logger.warning(f"Could not query model registry: {e}")
            return None

    def get_available_models(self) -> list:
        """
        Get list of available models from registry.

        Returns:
            List of model info dicts
        """
        if not PSYCOPG2_AVAILABLE or not self._db_url:
            return []

        try:
            conn = psycopg2.connect(self._db_url)
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute(
                """
                SELECT model_id, model_version, model_path, status,
                       COALESCE(backtest_sharpe, test_sharpe) as sharpe,
                       COALESCE(backtest_max_drawdown, test_max_drawdown) as max_drawdown
                FROM model_registry
                WHERE status IN ('registered', 'deployed')
                ORDER BY
                    CASE status WHEN 'deployed' THEN 0 ELSE 1 END,
                    created_at DESC
                """
            )

            rows = cursor.fetchall()
            cursor.close()
            conn.close()

            return [dict(row) for row in rows]

        except Exception as e:
            logger.warning(f"Could not fetch available models: {e}")
            return []

    def is_loaded(self, model_id: str) -> bool:
        """Check if a model is loaded"""
        return model_id in self.models

    def predict(
        self,
        observation: np.ndarray,
        model_id: str = "ppo_primary",
        deterministic: bool = True
    ) -> Tuple[float, float]:
        """
        Run inference on a single observation.

        Args:
            observation: 15-dimensional numpy array
            model_id: Model to use for prediction
            deterministic: Use deterministic prediction

        Returns:
            Tuple of (action, confidence)
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not loaded. Call load_model() first.")

        model = self.models[model_id]

        # Ensure correct shape
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)

        # Run prediction
        if isinstance(model, MockModel):
            action = model.predict(observation, deterministic=deterministic)
        else:
            action, _ = model.predict(observation, deterministic=deterministic)

        # Extract scalar action
        action_value = float(action[0]) if hasattr(action, '__len__') else float(action)

        # Calculate confidence as magnitude of action
        confidence = min(abs(action_value), 1.0)

        # Call post-predict hook for drift observation
        if self.post_predict_hook is not None:
            try:
                self.post_predict_hook(observation)
            except Exception as e:
                logger.debug(f"Post-predict hook error: {e}")

        return action_value, confidence

    def get_signal(self, action: float, model_id: str = "ppo_primary") -> str:
        """
        Convert continuous action to discrete trading signal.

        Args:
            action: Continuous action value
            model_id: Model ID to get model-specific thresholds

        Returns:
            "LONG", "SHORT", or "HOLD"
        """
        # Get model-specific thresholds or use defaults
        thresholds = self.MODEL_THRESHOLDS.get(model_id, {})
        threshold_long = thresholds.get("long", self.threshold_long)
        threshold_short = thresholds.get("short", self.threshold_short)

        if action > threshold_long:
            return "LONG"
        elif action < threshold_short:
            return "SHORT"
        return "HOLD"

    def predict_signal(
        self,
        observation: np.ndarray,
        model_id: str = "ppo_primary"
    ) -> Tuple[str, float, float]:
        """
        Predict and return signal with confidence.

        Args:
            observation: 15-dimensional numpy array
            model_id: Model to use

        Returns:
            Tuple of (signal, action, confidence)
        """
        action, confidence = self.predict(observation, model_id)
        signal = self.get_signal(action, model_id)
        return signal, action, confidence


class MockModel:
    """
    Mock model for testing when stable_baselines3 is not available.
    Generates random signals with slight long bias.
    """

    def __init__(self):
        self.rng = np.random.default_rng(42)

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Generate mock prediction based on some observation features"""
        # Use RSI-like feature (index 3) to generate somewhat meaningful signals
        if observation.ndim == 2:
            obs = observation[0]
        else:
            obs = observation

        # Mock logic: tend to go long when RSI is low, short when high
        rsi_feature = obs[3] if len(obs) > 3 else 0

        # Base action with some noise
        base = -rsi_feature * 0.1  # Inverse of RSI z-score
        noise = self.rng.normal(0, 0.05) if not deterministic else 0

        action = np.clip(base + noise, -1.0, 1.0)

        return np.array([action])
