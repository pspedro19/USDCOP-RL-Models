"""
Ensemble Predictor
==================

Single Responsibility: Coordinate multiple predictors using strategy pattern.

Author: Trading Team
Version: 1.0.0
Date: 2025-01-14
"""

import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

import numpy as np

from src.core.interfaces.inference import (
    IEnsembleStrategy,
    InferenceResult,
    EnsembleResult,
)
from src.core.strategies.ensemble_strategies import (
    EnsembleStrategyRegistry,
    WeightedAverageStrategy,
)
from .predictor import ONNXPredictor

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Coordinates multiple predictors for ensemble inference.

    Strategy Pattern: Delegates combination logic to IEnsembleStrategy.
    Single Responsibility: Only coordinates predictors, doesn't combine.
    """

    def __init__(
        self,
        predictors: Optional[Dict[str, ONNXPredictor]] = None,
        strategy: Optional[IEnsembleStrategy] = None,
        strategy_name: Optional[str] = None,
    ):
        """
        Args:
            predictors: Dict of name -> predictor
            strategy: Ensemble strategy instance (takes precedence)
            strategy_name: Strategy name to look up in registry
        """
        self._predictors: Dict[str, ONNXPredictor] = predictors or {}

        # Resolve strategy
        if strategy:
            self._strategy = strategy
        elif strategy_name:
            self._strategy = EnsembleStrategyRegistry.get(strategy_name)
        else:
            self._strategy = EnsembleStrategyRegistry.get_default()

        logger.info(f"EnsemblePredictor using strategy: {self._strategy.name}")

    def add_predictor(self, predictor: ONNXPredictor) -> 'EnsemblePredictor':
        """
        Add a predictor to the ensemble.

        Args:
            predictor: Predictor to add

        Returns:
            Self for chaining
        """
        self._predictors[predictor.model_name] = predictor
        logger.info(f"Added predictor: {predictor.model_name}")
        return self

    def remove_predictor(self, name: str) -> 'EnsemblePredictor':
        """
        Remove a predictor from the ensemble.

        Args:
            name: Predictor name to remove

        Returns:
            Self for chaining
        """
        if name in self._predictors:
            del self._predictors[name]
            logger.info(f"Removed predictor: {name}")
        return self

    def set_strategy(self, strategy: IEnsembleStrategy) -> 'EnsemblePredictor':
        """
        Change ensemble strategy.

        Open/Closed: Change behavior without modifying this class.

        Args:
            strategy: New strategy

        Returns:
            Self for chaining
        """
        self._strategy = strategy
        logger.info(f"Changed strategy to: {strategy.name}")
        return self

    def set_strategy_by_name(self, name: str) -> 'EnsemblePredictor':
        """
        Change strategy by name.

        Args:
            name: Strategy name from registry

        Returns:
            Self for chaining
        """
        self._strategy = EnsembleStrategyRegistry.get(name)
        logger.info(f"Changed strategy to: {name}")
        return self

    def predict(self, observation: np.ndarray) -> EnsembleResult:
        """
        Run ensemble inference.

        Args:
            observation: Feature vector

        Returns:
            EnsembleResult with combined signal
        """
        if not self._predictors:
            raise RuntimeError("No predictors in ensemble")

        start_time = time.perf_counter()
        individual_results: List[InferenceResult] = []
        errors: List[str] = []

        # Get predictions from all models
        for name, predictor in self._predictors.items():
            try:
                result = predictor.predict(observation)
                individual_results.append(result)
            except Exception as e:
                logger.error(f"Inference failed for {name}: {e}")
                errors.append(f"{name}: {str(e)}")

        if not individual_results:
            raise RuntimeError(f"All model inferences failed: {errors}")

        # Get weights for strategy
        weights = {
            name: predictor.weight
            for name, predictor in self._predictors.items()
        }

        # Combine results using strategy
        combined_probs, signal, confidence = self._strategy.combine(
            individual_results,
            weights
        )

        total_latency = (time.perf_counter() - start_time) * 1000

        return EnsembleResult(
            signal=signal,
            confidence=confidence,
            action_probs=combined_probs,
            individual_results=individual_results,
            ensemble_strategy=self._strategy.name,
            total_latency_ms=total_latency,
            timestamp=datetime.now().isoformat(),
        )

    @property
    def predictor_names(self) -> List[str]:
        """Get list of predictor names."""
        return list(self._predictors.keys())

    @property
    def predictor_count(self) -> int:
        """Get number of predictors."""
        return len(self._predictors)

    @property
    def strategy_name(self) -> str:
        """Get current strategy name."""
        return self._strategy.name

    def get_predictor(self, name: str) -> Optional[ONNXPredictor]:
        """Get predictor by name."""
        return self._predictors.get(name)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all predictors."""
        return {
            "strategy": self._strategy.name,
            "predictor_count": self.predictor_count,
            "predictors": {
                name: pred.stats
                for name, pred in self._predictors.items()
            },
        }
