"""
Drift Observation Service
=========================

Service that feeds observations to drift detectors after inference.
Integrates with the inference pipeline to track feature distributions.

Usage:
    # Initialize during startup
    drift_service = DriftObservationService(
        drift_detector=app.state.drift_detector,
        multivariate_detector=app.state.multivariate_drift_detector,
    )
    app.state.drift_observation_service = drift_service

    # After each inference
    drift_service.observe(observation_dict, observation_array)

Author: Trading Team
Date: 2026-01-17
"""

import logging
from typing import Dict, List, Optional, TYPE_CHECKING
import numpy as np
from collections import deque
from datetime import datetime
import asyncio

if TYPE_CHECKING:
    from src.monitoring.drift_detector import (
        FeatureDriftDetector,
        MultivariateDriftDetector,
    )

logger = logging.getLogger(__name__)


class DriftObservationService:
    """
    Service for feeding observations to drift detectors.

    Features:
    - Async-safe observation buffering
    - Batch processing for efficiency
    - Automatic feature name mapping
    - Statistics tracking
    """

    def __init__(
        self,
        drift_detector: Optional['FeatureDriftDetector'] = None,
        multivariate_detector: Optional['MultivariateDriftDetector'] = None,
        feature_order: Optional[List[str]] = None,
        buffer_size: int = 100,
        auto_flush: bool = True,
    ):
        """
        Initialize the drift observation service.

        Args:
            drift_detector: Univariate drift detector instance
            multivariate_detector: Multivariate drift detector instance
            feature_order: List of feature names in order (for array conversion)
            buffer_size: Number of observations to buffer before flushing
            auto_flush: Automatically flush buffer when full
        """
        self.drift_detector = drift_detector
        self.multivariate_detector = multivariate_detector
        self.buffer_size = buffer_size
        self.auto_flush = auto_flush

        # Load feature order from SSOT
        if feature_order is None:
            try:
                from src.core.contracts import FEATURE_ORDER
                self.feature_order = list(FEATURE_ORDER)
            except ImportError:
                self.feature_order = []
                logger.warning("Could not load FEATURE_ORDER from SSOT")
        else:
            self.feature_order = feature_order

        # Observation buffer for batch processing
        self._buffer: deque = deque(maxlen=buffer_size)
        self._lock = asyncio.Lock()

        # Statistics
        self._observations_total = 0
        self._observations_buffered = 0
        self._flushes = 0
        self._errors = 0
        self._last_observation_time: Optional[datetime] = None

        logger.info(
            f"DriftObservationService initialized: "
            f"univariate={drift_detector is not None}, "
            f"multivariate={multivariate_detector is not None}, "
            f"features={len(self.feature_order)}"
        )

    def observe(
        self,
        observation_dict: Optional[Dict[str, float]] = None,
        observation_array: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Add an observation to the drift detectors.

        Either observation_dict or observation_array should be provided.
        If both are provided, dict is used for univariate and array for multivariate.

        Args:
            observation_dict: Dictionary mapping feature names to values
            observation_array: Numpy array of feature values in FEATURE_ORDER

        Returns:
            True if observation was successfully added
        """
        try:
            self._observations_total += 1
            self._last_observation_time = datetime.utcnow()

            # Add to univariate detector
            if self.drift_detector is not None:
                if observation_dict is not None:
                    self.drift_detector.add_observation(observation_dict)
                elif observation_array is not None and self.feature_order:
                    # Convert array to dict
                    obs_dict = {
                        feat: float(observation_array[i])
                        for i, feat in enumerate(self.feature_order)
                        if i < len(observation_array)
                    }
                    self.drift_detector.add_observation(obs_dict)

            # Add to multivariate detector
            if self.multivariate_detector is not None:
                if observation_array is not None:
                    self.multivariate_detector.add_observation(observation_array)
                elif observation_dict is not None and self.feature_order:
                    # Convert dict to array
                    obs_array = np.array([
                        observation_dict.get(feat, 0.0)
                        for feat in self.feature_order
                    ])
                    self.multivariate_detector.add_observation(obs_array)

            return True

        except Exception as e:
            self._errors += 1
            logger.warning(f"Error adding drift observation: {e}")
            return False

    async def observe_async(
        self,
        observation_dict: Optional[Dict[str, float]] = None,
        observation_array: Optional[np.ndarray] = None,
        buffer: bool = False,
    ) -> bool:
        """
        Async version of observe with optional buffering.

        Args:
            observation_dict: Dictionary mapping feature names to values
            observation_array: Numpy array of feature values
            buffer: If True, buffer observation for batch processing

        Returns:
            True if observation was successfully handled
        """
        if buffer:
            async with self._lock:
                self._buffer.append((observation_dict, observation_array))
                self._observations_buffered += 1

                if self.auto_flush and len(self._buffer) >= self.buffer_size:
                    await self._flush_buffer()

            return True
        else:
            return self.observe(observation_dict, observation_array)

    async def _flush_buffer(self) -> int:
        """
        Flush buffered observations to detectors.

        Returns:
            Number of observations flushed
        """
        if not self._buffer:
            return 0

        count = 0
        while self._buffer:
            obs_dict, obs_array = self._buffer.popleft()
            if self.observe(obs_dict, obs_array):
                count += 1

        self._flushes += 1
        logger.debug(f"Flushed {count} observations to drift detectors")
        return count

    async def flush(self) -> int:
        """Public method to flush the buffer."""
        async with self._lock:
            return await self._flush_buffer()

    def observe_batch(
        self,
        observations_dict: Optional[List[Dict[str, float]]] = None,
        observations_array: Optional[np.ndarray] = None,
    ) -> int:
        """
        Add multiple observations at once.

        Args:
            observations_dict: List of observation dictionaries
            observations_array: 2D numpy array of observations

        Returns:
            Number of observations successfully added
        """
        count = 0

        if observations_dict is not None:
            for obs in observations_dict:
                if self.observe(observation_dict=obs):
                    count += 1

        if observations_array is not None:
            for obs in observations_array:
                if self.observe(observation_array=obs):
                    count += 1

        return count

    def get_stats(self) -> Dict:
        """Get service statistics."""
        return {
            "observations_total": self._observations_total,
            "observations_buffered": self._observations_buffered,
            "buffer_current_size": len(self._buffer),
            "flushes": self._flushes,
            "errors": self._errors,
            "last_observation": self._last_observation_time.isoformat() if self._last_observation_time else None,
            "univariate_detector": self.drift_detector is not None,
            "multivariate_detector": self.multivariate_detector is not None,
            "feature_count": len(self.feature_order),
        }

    def get_status(self) -> Dict:
        """Get service status for health checks."""
        univariate_ready = (
            self.drift_detector is not None and
            hasattr(self.drift_detector, 'reference_manager') and
            len(self.drift_detector.reference_manager.feature_names) > 0
        )

        multivariate_ready = (
            self.multivariate_detector is not None and
            self.multivariate_detector._reference_data is not None
        )

        return {
            "service_active": True,
            "univariate_ready": univariate_ready,
            "multivariate_ready": multivariate_ready,
            "fully_ready": univariate_ready and multivariate_ready,
            "observations_processed": self._observations_total,
        }


def create_drift_observation_hook(service: DriftObservationService):
    """
    Create a hook function for the inference engine.

    Returns a callable that can be passed to inference engine
    to automatically observe features after each prediction.

    Usage:
        hook = create_drift_observation_hook(drift_service)
        inference_engine.set_post_predict_hook(hook)

    Or call directly after predictions:
        signal, action, confidence = engine.predict_signal(obs)
        hook(obs)
    """
    def hook(observation: np.ndarray) -> None:
        """Post-prediction hook to feed observation to drift detectors."""
        try:
            service.observe(observation_array=observation.flatten())
        except Exception as e:
            logger.debug(f"Drift observation hook error: {e}")

    return hook


__all__ = [
    "DriftObservationService",
    "create_drift_observation_hook",
]
