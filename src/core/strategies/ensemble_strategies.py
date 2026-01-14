"""
Ensemble Strategy Implementations
=================================

Concrete implementations of IEnsembleStrategy following Strategy Pattern.
Open/Closed: Add new strategies without modifying existing code.

Usage:
    # Register custom strategy
    EnsembleStrategyRegistry.register(MyCustomStrategy())

    # Get strategy by name
    strategy = EnsembleStrategyRegistry.get("weighted_average")
    probs, signal, confidence = strategy.combine(results, weights)

Author: Trading Team
Version: 1.0.0
Date: 2025-01-14
"""

from typing import Dict, List, Tuple, Optional, Type
import logging

from src.core.interfaces.inference import (
    IEnsembleStrategy,
    InferenceResult,
    SignalType,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Strategy Implementations
# =============================================================================

class WeightedAverageStrategy(IEnsembleStrategy):
    """
    Combine results using weighted average of action probabilities.

    Each model's contribution is weighted by its configured weight
    multiplied by the confidence of its prediction.
    """

    @property
    def name(self) -> str:
        return "weighted_average"

    def combine(
        self,
        results: List[InferenceResult],
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[Dict[str, float], SignalType, float]:
        """Weighted average of action probabilities."""
        if not results:
            return {"HOLD": 1.0, "BUY": 0.0, "SELL": 0.0}, SignalType.HOLD, 0.0

        weights = weights or {}
        total_weight = 0.0
        combined = {"HOLD": 0.0, "BUY": 0.0, "SELL": 0.0}

        for result in results:
            # Use model weight from config, default to 1.0
            model_weight = weights.get(result.model_name, 1.0)
            # Weight contribution by confidence
            effective_weight = model_weight * result.confidence

            for action in combined:
                combined[action] += result.action_probs.get(action, 0.0) * effective_weight

            total_weight += effective_weight

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


class MajorityVoteStrategy(IEnsembleStrategy):
    """
    Combine results using majority vote.

    Each model gets one vote for its predicted signal.
    Confidence is the proportion of votes for the winning signal.
    """

    @property
    def name(self) -> str:
        return "majority_vote"

    def combine(
        self,
        results: List[InferenceResult],
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[Dict[str, float], SignalType, float]:
        """Simple majority vote."""
        if not results:
            return {"HOLD": 1.0, "BUY": 0.0, "SELL": 0.0}, SignalType.HOLD, 0.0

        votes = {"HOLD": 0, "BUY": 0, "SELL": 0}

        for result in results:
            votes[result.signal.value] += 1

        # Get majority
        total_votes = sum(votes.values())
        best_action = max(votes, key=votes.get)

        signal_map = {"HOLD": SignalType.HOLD, "BUY": SignalType.BUY, "SELL": SignalType.SELL}
        signal = signal_map[best_action]

        # Confidence is proportion of votes
        confidence = votes[best_action] / total_votes if total_votes > 0 else 0.0

        # Normalized votes as probs
        probs = {k: v / total_votes for k, v in votes.items()} if total_votes > 0 else {"HOLD": 1.0, "BUY": 0.0, "SELL": 0.0}

        return probs, signal, confidence


class SoftVoteStrategy(IEnsembleStrategy):
    """
    Combine results using soft voting (average of probabilities).

    Similar to weighted average but without confidence weighting.
    Each model contributes equally to the final probability distribution.
    """

    @property
    def name(self) -> str:
        return "soft_vote"

    def combine(
        self,
        results: List[InferenceResult],
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[Dict[str, float], SignalType, float]:
        """Simple average of probabilities."""
        if not results:
            return {"HOLD": 1.0, "BUY": 0.0, "SELL": 0.0}, SignalType.HOLD, 0.0

        combined = {"HOLD": 0.0, "BUY": 0.0, "SELL": 0.0}
        n_models = len(results)

        for result in results:
            for action in combined:
                combined[action] += result.action_probs.get(action, 0.0)

        # Average
        for action in combined:
            combined[action] /= n_models

        # Get final signal
        signal_map = {"HOLD": SignalType.HOLD, "BUY": SignalType.BUY, "SELL": SignalType.SELL}
        best_action = max(combined, key=combined.get)
        signal = signal_map[best_action]
        confidence = combined[best_action]

        return combined, signal, confidence


class ConfidenceWeightedStrategy(IEnsembleStrategy):
    """
    Weight each model's contribution by its confidence only.

    Models with higher confidence have more influence.
    This ignores configured model weights.
    """

    @property
    def name(self) -> str:
        return "confidence_weighted"

    def combine(
        self,
        results: List[InferenceResult],
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[Dict[str, float], SignalType, float]:
        """Weight by confidence."""
        if not results:
            return {"HOLD": 1.0, "BUY": 0.0, "SELL": 0.0}, SignalType.HOLD, 0.0

        combined = {"HOLD": 0.0, "BUY": 0.0, "SELL": 0.0}
        total_confidence = sum(r.confidence for r in results)

        if total_confidence == 0:
            # Fallback to equal weights
            return SoftVoteStrategy().combine(results, weights)

        for result in results:
            weight = result.confidence / total_confidence
            for action in combined:
                combined[action] += result.action_probs.get(action, 0.0) * weight

        # Get final signal
        signal_map = {"HOLD": SignalType.HOLD, "BUY": SignalType.BUY, "SELL": SignalType.SELL}
        best_action = max(combined, key=combined.get)
        signal = signal_map[best_action]
        confidence = combined[best_action]

        return combined, signal, confidence


class BestModelStrategy(IEnsembleStrategy):
    """
    Use the result from the model with highest confidence.

    No averaging - just pick the most confident model's prediction.
    """

    @property
    def name(self) -> str:
        return "best_model"

    def combine(
        self,
        results: List[InferenceResult],
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[Dict[str, float], SignalType, float]:
        """Use most confident model."""
        if not results:
            return {"HOLD": 1.0, "BUY": 0.0, "SELL": 0.0}, SignalType.HOLD, 0.0

        # Find model with highest confidence
        best_result = max(results, key=lambda r: r.confidence)

        return (
            best_result.action_probs.copy(),
            best_result.signal,
            best_result.confidence,
        )


# =============================================================================
# Strategy Registry (Open/Closed Principle)
# =============================================================================

class EnsembleStrategyRegistry:
    """
    Registry for ensemble strategies.

    Registry Pattern: Allows dynamic registration and lookup of strategies.
    Open/Closed: Add new strategies without modifying existing code.

    Usage:
        # Register custom strategy
        EnsembleStrategyRegistry.register(MyCustomStrategy())

        # Get strategy by name
        strategy = EnsembleStrategyRegistry.get("weighted_average")

        # List available strategies
        names = EnsembleStrategyRegistry.list_strategies()
    """

    _strategies: Dict[str, IEnsembleStrategy] = {}
    _initialized: bool = False

    @classmethod
    def _ensure_initialized(cls):
        """Initialize with default strategies if not done."""
        if not cls._initialized:
            cls._register_defaults()
            cls._initialized = True

    @classmethod
    def _register_defaults(cls):
        """Register built-in strategies."""
        defaults = [
            WeightedAverageStrategy(),
            MajorityVoteStrategy(),
            SoftVoteStrategy(),
            ConfidenceWeightedStrategy(),
            BestModelStrategy(),
        ]
        for strategy in defaults:
            cls._strategies[strategy.name] = strategy

    @classmethod
    def register(cls, strategy: IEnsembleStrategy) -> None:
        """
        Register a strategy.

        Args:
            strategy: Strategy instance implementing IEnsembleStrategy

        Raises:
            ValueError: If strategy with same name already exists
        """
        cls._ensure_initialized()

        if strategy.name in cls._strategies:
            logger.warning(f"Overwriting existing strategy: {strategy.name}")

        cls._strategies[strategy.name] = strategy
        logger.info(f"Registered ensemble strategy: {strategy.name}")

    @classmethod
    def get(cls, name: str) -> IEnsembleStrategy:
        """
        Get strategy by name.

        Args:
            name: Strategy name

        Returns:
            Strategy instance

        Raises:
            KeyError: If strategy not found
        """
        cls._ensure_initialized()

        if name not in cls._strategies:
            available = list(cls._strategies.keys())
            raise KeyError(
                f"Unknown ensemble strategy: '{name}'. "
                f"Available: {available}"
            )

        return cls._strategies[name]

    @classmethod
    def list_strategies(cls) -> List[str]:
        """Get list of available strategy names."""
        cls._ensure_initialized()
        return list(cls._strategies.keys())

    @classmethod
    def get_default(cls) -> IEnsembleStrategy:
        """Get the default strategy (weighted_average)."""
        cls._ensure_initialized()
        return cls._strategies.get("weighted_average", WeightedAverageStrategy())
