"""
Inference Service - DI Example
==============================

Demonstrates how services should use ApplicationContext for dependency injection.

This service handles model inference operations with proper DI:
- All dependencies received through constructor (explicit)
- No global state or service locator calls
- Easy to test with mock dependencies

Design Principles:
- Dependency Injection: Dependencies passed via constructor
- Single Responsibility: Only handles inference orchestration
- Interface Segregation: Depends on focused interfaces (IPredictor, IFeatureBuilder)
- Open/Closed: New features can be added without modifying core logic

Author: Trading Team
Version: 1.0.0
Date: 2026-01-16
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from src.core.container import (
    ApplicationContext,
    Event,
    SignalType,
    PredictionResult,
    ILogger,
)

if TYPE_CHECKING:
    from src.core.container import (
        IFeatureBuilder,
        IPredictor,
        IEventBus,
        IRiskManager,
        ITradeRepository,
    )


# =============================================================================
# DOMAIN EVENTS
# =============================================================================

@dataclass
class InferenceRequestedEvent(Event):
    """Event published when inference is requested."""
    pass


@dataclass
class InferenceCompletedEvent(Event):
    """Event published when inference completes."""
    pass


@dataclass
class TradeSignalGeneratedEvent(Event):
    """Event published when a trade signal is generated."""
    pass


# =============================================================================
# RESULT TYPES
# =============================================================================

@dataclass
class InferenceServiceResult:
    """Result of inference service call."""
    signal: SignalType
    confidence: float
    action_probs: Dict[str, float]
    position_size: float
    trade_allowed: bool
    rejection_reason: Optional[str]
    latency_ms: float
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal": self.signal.value,
            "confidence": self.confidence,
            "action_probs": self.action_probs,
            "position_size": self.position_size,
            "trade_allowed": self.trade_allowed,
            "rejection_reason": self.rejection_reason,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
        }


# =============================================================================
# INFERENCE SERVICE
# =============================================================================

class InferenceService:
    """
    Service for model inference with proper dependency injection.

    This class demonstrates the recommended pattern for using ApplicationContext:
    1. Receive context through constructor
    2. Extract needed dependencies
    3. Use explicit interfaces, not concrete types
    4. Publish domain events for side effects

    Example:
        # Production
        context = ApplicationContext.create_production(config)
        service = InferenceService(context)
        result = service.run_inference(market_data, position=0.0, bar_idx=30)

        # Testing
        context = ApplicationContext.create_for_testing(
            predictor=mock_predictor,
            risk_manager=mock_risk_manager,
        )
        service = InferenceService(context)
        result = service.run_inference(market_data, position=0.0, bar_idx=30)
    """

    def __init__(self, context: ApplicationContext):
        """
        Initialize service with application context.

        Args:
            context: ApplicationContext with all dependencies
        """
        # Extract dependencies from context
        self._builder = context.feature_builder
        self._predictor = context.predictor
        self._risk_manager = context.risk_manager
        self._event_bus = context.event_bus
        self._trade_repository = context.trade_repository
        self._logger = context.logger_factory.get_logger(__name__)

        # Service state
        self._inference_count = 0
        self._total_latency_ms = 0.0

        self._logger.info("InferenceService initialized")

    def run_inference(
        self,
        market_data: Dict[str, Any],
        position: float,
        bar_idx: int,
        macro_data: Optional[Dict[str, Any]] = None,
    ) -> InferenceServiceResult:
        """
        Run complete inference pipeline.

        Args:
            market_data: OHLCV data (open, high, low, close, volume)
            position: Current position (-1 to 1)
            bar_idx: Current bar index in episode (1-based)
            macro_data: Optional macro data (dxy, vix, embi, etc.)

        Returns:
            InferenceServiceResult with signal, confidence, and risk check
        """
        start_time = time.perf_counter()
        timestamp = datetime.utcnow().isoformat()

        # Publish request event
        self._publish_event("inference.requested", {
            "bar_idx": bar_idx,
            "position": position,
        })

        try:
            # 1. Build observation
            observation = self._builder.build_observation(
                ohlcv=market_data,
                macro=macro_data,
                position=position,
                bar_idx=bar_idx,
            )

            # 2. Run prediction
            action, confidence = self._predictor.predict(observation)

            # 3. Convert action to signal
            signal = self._action_to_signal(action)

            # 4. Get action probabilities if available
            action_probs = self._get_action_probs(observation)

            # 5. Check risk rules
            trade_allowed, rejection_reason = self._risk_manager.check_trade_allowed(
                signal=signal,
                confidence=confidence,
                position=position,
            )

            # 6. Calculate position size if trade allowed
            if trade_allowed and signal != SignalType.HOLD:
                position_size = self._risk_manager.get_position_size(signal, confidence)
            else:
                position_size = 0.0

            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Update stats
            self._inference_count += 1
            self._total_latency_ms += latency_ms

            # Build result
            result = InferenceServiceResult(
                signal=signal,
                confidence=confidence,
                action_probs=action_probs,
                position_size=position_size,
                trade_allowed=trade_allowed,
                rejection_reason=rejection_reason,
                latency_ms=latency_ms,
                timestamp=timestamp,
            )

            # Publish completion event
            self._publish_event("inference.completed", {
                "signal": signal.value,
                "confidence": confidence,
                "latency_ms": latency_ms,
            })

            # Publish trade signal event if trade is allowed
            if trade_allowed and signal != SignalType.HOLD:
                self._publish_event("trade.signal_generated", {
                    "signal": signal.value,
                    "confidence": confidence,
                    "position_size": position_size,
                })

            self._logger.debug(
                f"Inference complete: signal={signal.value}, "
                f"confidence={confidence:.3f}, latency={latency_ms:.2f}ms"
            )

            return result

        except Exception as e:
            self._logger.error(f"Inference failed: {e}")
            # Return HOLD signal on error
            latency_ms = (time.perf_counter() - start_time) * 1000
            return InferenceServiceResult(
                signal=SignalType.HOLD,
                confidence=0.0,
                action_probs={"HOLD": 1.0, "BUY": 0.0, "SELL": 0.0},
                position_size=0.0,
                trade_allowed=False,
                rejection_reason=f"Inference error: {e}",
                latency_ms=latency_ms,
                timestamp=timestamp,
            )

    def run_batch_inference(
        self,
        observations: np.ndarray,
    ) -> List[Tuple[int, float]]:
        """
        Run inference on batch of observations.

        Args:
            observations: Array of shape (batch_size, obs_dim)

        Returns:
            List of (action, confidence) tuples
        """
        results = []
        for obs in observations:
            action, confidence = self._predictor.predict(obs)
            results.append((action, confidence))
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        return {
            "inference_count": self._inference_count,
            "total_latency_ms": self._total_latency_ms,
            "avg_latency_ms": (
                self._total_latency_ms / self._inference_count
                if self._inference_count > 0 else 0.0
            ),
            "model_name": self._predictor.model_name,
        }

    def _action_to_signal(self, action: int) -> SignalType:
        """Convert action index to signal type."""
        mapping = {
            0: SignalType.HOLD,
            1: SignalType.BUY,
            2: SignalType.SELL,
        }
        return mapping.get(action, SignalType.HOLD)

    def _get_action_probs(self, observation: np.ndarray) -> Dict[str, float]:
        """Get action probabilities if predictor supports it."""
        try:
            if hasattr(self._predictor, 'predict_with_probs'):
                result = self._predictor.predict_with_probs(observation)
                return result.action_probs
        except Exception:
            pass
        return {"HOLD": 0.33, "BUY": 0.33, "SELL": 0.34}

    def _publish_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Publish domain event."""
        event = Event(
            event_type=event_type,
            timestamp=datetime.utcnow().isoformat(),
            payload=payload,
        )
        self._event_bus.publish(event)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_inference_service(context: ApplicationContext) -> InferenceService:
    """
    Factory function for creating InferenceService.

    Args:
        context: ApplicationContext with dependencies

    Returns:
        Configured InferenceService
    """
    return InferenceService(context)


def create_inference_service_from_config(config: Any) -> InferenceService:
    """
    Create InferenceService from config (convenience function).

    Args:
        config: TradingConfig or similar

    Returns:
        Configured InferenceService
    """
    context = ApplicationContext.create_production(config)
    return InferenceService(context)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'InferenceService',
    'InferenceServiceResult',
    'InferenceRequestedEvent',
    'InferenceCompletedEvent',
    'TradeSignalGeneratedEvent',
    'create_inference_service',
    'create_inference_service_from_config',
]
