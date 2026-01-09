"""
Inference Service
==================
Real-time inference service for PPO-LSTM model predictions.

Author: Pedro @ Lean Tech Solutions
Created: 2025-12-17
"""

import logging
import time
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd

from ..models.model_loader import ONNXModelLoader, DummyModelLoader
from ..models.signal_schema import SignalAction
from ..config import get_config

logger = logging.getLogger(__name__)


class InferenceService:
    """Real-time inference service"""

    def __init__(self, model_loader: Optional[ONNXModelLoader] = None):
        """
        Initialize inference service.

        Args:
            model_loader: Optional model loader instance. If None, creates default loader.
        """
        self.config = get_config()

        if model_loader is None:
            self.model_loader = ONNXModelLoader(
                model_path=self.config.model_path,
                model_version=self.config.model_version,
                use_gpu=False  # Use CPU for stability
            )
        else:
            self.model_loader = model_loader

        self.inference_count = 0
        self.total_latency_ms = 0.0

    def initialize(self) -> bool:
        """
        Initialize the inference service by loading the model.

        Returns:
            True if initialization successful, False otherwise
        """
        logger.info("Initializing inference service...")
        success = self.model_loader.load()

        if success:
            logger.info("Inference service initialized successfully")
        else:
            logger.warning("Inference service running in PLACEHOLDER mode")

        return success

    def prepare_features(self, market_data: Dict[str, Any]) -> np.ndarray:
        """
        Prepare features from market data for model input.

        Args:
            market_data: Dictionary containing OHLCV and technical indicators

        Returns:
            Feature array ready for model input
        """
        # Extract features in expected order
        # Note: This is a placeholder - adjust based on actual model input requirements
        features = []

        # Price features (normalized)
        close = market_data.get('close', 0)
        open_price = market_data.get('open', close)
        high = market_data.get('high', close)
        low = market_data.get('low', close)

        # Normalize prices relative to close
        if close > 0:
            features.extend([
                open_price / close - 1.0,  # open_return
                high / close - 1.0,        # high_return
                low / close - 1.0,         # low_return
            ])
        else:
            features.extend([0.0, 0.0, 0.0])

        # Technical indicators
        features.append(market_data.get('rsi', 50.0) / 100.0)  # RSI normalized to [0, 1]
        features.append(market_data.get('macd', 0.0) / 100.0)  # MACD normalized
        features.append(market_data.get('macd_signal', 0.0) / 100.0)

        # Bollinger Bands position
        bb_upper = market_data.get('bb_upper', 0)
        bb_lower = market_data.get('bb_lower', 0)
        if bb_upper > bb_lower:
            bb_position = (close - bb_lower) / (bb_upper - bb_lower)
        else:
            bb_position = 0.5
        features.append(bb_position)

        # Volume (normalized by rolling average)
        volume = market_data.get('volume', 0)
        avg_volume = market_data.get('avg_volume', volume)
        if avg_volume > 0:
            volume_ratio = volume / avg_volume
        else:
            volume_ratio = 1.0
        features.append(np.log1p(volume_ratio))

        # EMAs
        ema_20 = market_data.get('ema_20', close)
        ema_50 = market_data.get('ema_50', close)
        if close > 0:
            features.append(ema_20 / close - 1.0)
            features.append(ema_50 / close - 1.0)
        else:
            features.extend([0.0, 0.0])

        # ATR (normalized)
        atr = market_data.get('atr', 0)
        if close > 0:
            features.append(atr / close)
        else:
            features.append(0.0)

        return np.array(features, dtype=np.float32)

    def predict_action(
        self,
        market_data: Dict[str, Any]
    ) -> Tuple[SignalAction, float, float]:
        """
        Predict trading action from market data.

        Args:
            market_data: Dictionary containing market data and indicators

        Returns:
            Tuple of (action, confidence, latency_ms)
        """
        try:
            # Prepare features
            features = self.prepare_features(market_data)

            # Run inference
            predictions, latency_ms = self.model_loader.predict(features)

            # Get action from predictions (assuming 3 actions: HOLD, BUY, SELL)
            action_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][action_idx])

            # Map index to action
            action_map = {
                0: SignalAction.HOLD,
                1: SignalAction.BUY,
                2: SignalAction.SELL
            }
            action = action_map.get(action_idx, SignalAction.HOLD)

            # Update stats
            self.inference_count += 1
            self.total_latency_ms += latency_ms

            logger.debug(
                f"Inference #{self.inference_count}: "
                f"Action={action.value}, Confidence={confidence:.3f}, "
                f"Latency={latency_ms:.2f}ms"
            )

            return action, confidence, latency_ms

        except Exception as e:
            logger.error(f"Error during inference: {e}")
            # Return conservative default
            return SignalAction.HOLD, 0.0, 0.0

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        info = self.model_loader.get_model_info()
        info.update({
            'inference_count': self.inference_count,
            'avg_latency_ms': (
                self.total_latency_ms / self.inference_count
                if self.inference_count > 0 else 0.0
            )
        })
        return info

    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics"""
        return {
            'total_inferences': self.inference_count,
            'total_latency_ms': self.total_latency_ms,
            'avg_latency_ms': (
                self.total_latency_ms / self.inference_count
                if self.inference_count > 0 else 0.0
            ),
            'model_loaded': self.model_loader.is_loaded
        }

    def reset_stats(self):
        """Reset inference statistics"""
        self.inference_count = 0
        self.total_latency_ms = 0.0
        logger.info("Inference statistics reset")

    def shutdown(self):
        """Shutdown inference service"""
        logger.info("Shutting down inference service...")
        self.model_loader.unload()
        logger.info("Inference service shutdown complete")
