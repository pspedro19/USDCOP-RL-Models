"""
Validated Predictor
===================

Wraps a model with input/output validation using contracts.

Contract enforcement:
- Validates input using validate_model_input (CTR-MODEL-INPUT-001)
- Validates output using validate_model_output (CTR-ACTION-001)

Author: Trading Team
Version: 1.0.0
Date: 2026-01-17
"""

import logging
from typing import Tuple, List, Any, Dict, Optional, Protocol, runtime_checkable
from dataclasses import dataclass, field

import numpy as np

from src.core.contracts import (
    Action,
    OBSERVATION_DIM,
    ACTION_COUNT,
    validate_model_input,
    validate_model_output,
    ModelInputError,
    InvalidActionError,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class PredictableModel(Protocol):
    """Protocol for models that can be wrapped by ValidatedPredictor."""

    def predict(self, observation: np.ndarray) -> int:
        """Return action index from observation."""
        ...


@dataclass
class PredictionStats:
    """Statistics for validated predictions."""
    total_predictions: int = 0
    successful_predictions: int = 0
    input_validation_failures: int = 0
    output_validation_failures: int = 0

    @property
    def failure_count(self) -> int:
        """Total number of failures."""
        return self.input_validation_failures + self.output_validation_failures

    @property
    def success_rate(self) -> float:
        """Success rate as a percentage."""
        if self.total_predictions == 0:
            return 0.0
        return self.successful_predictions / self.total_predictions

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_predictions": self.total_predictions,
            "successful_predictions": self.successful_predictions,
            "input_validation_failures": self.input_validation_failures,
            "output_validation_failures": self.output_validation_failures,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate,
        }


class ValidatedPredictor:
    """
    Wrapper that validates model inputs and outputs.

    This class wraps any model with a predict() method and validates:
    - Input observations against CTR-MODEL-INPUT-001
    - Output actions against CTR-ACTION-001

    Supports two modes:
    - strict_mode=True: Raises exceptions on validation failures
    - strict_mode=False: Logs warnings and returns HOLD action on failures

    Example:
        >>> model = load_model("model.onnx")
        >>> validated = ValidatedPredictor(model, strict_mode=True)
        >>> action, probs = validated.predict(observation)
        >>> print(f"Action: {action.name}, Probabilities: {probs}")
    """

    def __init__(
        self,
        model: Any,
        strict_mode: bool = True,
        name: Optional[str] = None,
    ):
        """
        Initialize ValidatedPredictor.

        Args:
            model: Model with predict() method or callable
            strict_mode: If True, raise exceptions on validation failures;
                        if False, log warnings and return HOLD action
            name: Optional name for logging purposes
        """
        self._model = model
        self._strict_mode = strict_mode
        self._name = name or getattr(model, "name", "unknown_model")
        self._stats = PredictionStats()

    @property
    def model(self) -> Any:
        """Get the wrapped model."""
        return self._model

    @property
    def name(self) -> str:
        """Get predictor name."""
        return self._name

    @property
    def strict_mode(self) -> bool:
        """Check if strict mode is enabled."""
        return self._strict_mode

    @property
    def stats(self) -> PredictionStats:
        """Get prediction statistics."""
        return self._stats

    def _get_action_probabilities(self, observation: np.ndarray) -> np.ndarray:
        """
        Get action probabilities from model.

        Tries to use model.policy if available, otherwise falls back to
        one-hot encoding based on predicted action.

        Args:
            observation: Input observation (2D array with shape (batch, features))

        Returns:
            Action probabilities array with shape (ACTION_COUNT,)
        """
        # Try to get probabilities from policy
        if hasattr(self._model, "policy"):
            try:
                policy = self._model.policy
                if hasattr(policy, "get_distribution"):
                    # Stable-baselines3 style
                    dist = policy.get_distribution(observation)
                    if hasattr(dist, "distribution"):
                        probs = dist.distribution.probs
                        if hasattr(probs, "detach"):
                            # PyTorch tensor
                            return probs.detach().cpu().numpy()[0]
                        return np.array(probs)[0]
                elif hasattr(policy, "action_probability"):
                    # Custom policy interface
                    return np.array(policy.action_probability(observation))[0]
            except Exception as e:
                logger.debug(f"Could not get probabilities from policy: {e}")

        # Try direct probability output
        if hasattr(self._model, "predict_proba"):
            try:
                probs = self._model.predict_proba(observation)
                return np.array(probs)[0]
            except Exception as e:
                logger.debug(f"Could not get probabilities from predict_proba: {e}")

        # Fallback: one-hot based on predicted action
        action_idx = self._get_raw_prediction(observation)
        probs = np.zeros(ACTION_COUNT, dtype=np.float32)
        probs[action_idx] = 1.0
        return probs

    def _get_raw_prediction(self, observation: np.ndarray) -> int:
        """
        Get raw action index from model.

        Args:
            observation: Input observation (2D array)

        Returns:
            Action index (0, 1, or 2)
        """
        if hasattr(self._model, "predict"):
            result = self._model.predict(observation)
            # Handle tuple returns (action, state) from RL models
            if isinstance(result, tuple):
                result = result[0]
            # Handle batch dimension
            if hasattr(result, "__len__") and not isinstance(result, (int, np.integer)):
                result = result[0]
            return int(result)
        elif callable(self._model):
            result = self._model(observation)
            if isinstance(result, tuple):
                result = result[0]
            if hasattr(result, "__len__") and not isinstance(result, (int, np.integer)):
                result = result[0]
            return int(result)
        else:
            raise TypeError(f"Model must have predict() method or be callable, got {type(self._model)}")

    def _validate_input(self, observation: np.ndarray) -> np.ndarray:
        """
        Validate and prepare input observation.

        Args:
            observation: Raw input observation

        Returns:
            Validated observation as 2D array (batch_size, OBSERVATION_DIM)

        Raises:
            ModelInputError: If validation fails in strict mode
        """
        # Ensure numpy array
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation, dtype=np.float32)

        # Handle 1D input by reshaping to (1, features)
        if observation.ndim == 1:
            if observation.shape[0] != OBSERVATION_DIM:
                error_msg = f"Invalid observation dimension: {observation.shape[0]}, expected {OBSERVATION_DIM}"
                if self._strict_mode:
                    raise ModelInputError(error_msg)
                logger.warning(error_msg)
            observation = observation.reshape(1, -1)
        elif observation.ndim == 2:
            if observation.shape[1] != OBSERVATION_DIM:
                error_msg = f"Invalid feature dimension: {observation.shape[1]}, expected {OBSERVATION_DIM}"
                if self._strict_mode:
                    raise ModelInputError(error_msg)
                logger.warning(error_msg)
        else:
            error_msg = f"Invalid observation ndim: {observation.ndim}, expected 1 or 2"
            if self._strict_mode:
                raise ModelInputError(error_msg)
            logger.warning(error_msg)

        # Validate using contract (validates single observation from batch)
        try:
            validate_model_input(observation[0])
        except ModelInputError as e:
            if self._strict_mode:
                raise
            logger.warning(f"Input validation warning for {self._name}: {e}")

        return observation.astype(np.float32)

    def _validate_output(
        self,
        action_idx: int,
        probabilities: np.ndarray,
    ) -> Tuple[Action, np.ndarray]:
        """
        Validate model output and convert to Action enum.

        Args:
            action_idx: Raw action index from model
            probabilities: Action probability distribution

        Returns:
            Tuple of (Action enum, validated probabilities)

        Raises:
            InvalidActionError: If validation fails in strict mode
        """
        confidence = float(probabilities[action_idx]) if len(probabilities) > action_idx else 1.0

        try:
            # Validate using contract
            validate_model_output(
                action=action_idx,
                confidence=confidence,
                action_probs=probabilities.tolist(),
                raise_on_error=self._strict_mode,
            )
            # Convert to Action enum
            action = Action.from_int(action_idx)
        except InvalidActionError as e:
            if self._strict_mode:
                raise
            logger.warning(f"Output validation warning for {self._name}: {e}")
            # Default to HOLD on validation failure in non-strict mode
            action = Action.HOLD
            probabilities = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        return action, probabilities

    def predict(self, observation: np.ndarray) -> Tuple[Action, np.ndarray]:
        """
        Run validated prediction on a single observation.

        Args:
            observation: Feature vector with shape (OBSERVATION_DIM,) or (1, OBSERVATION_DIM)

        Returns:
            Tuple of (Action enum, probability array with shape (ACTION_COUNT,))

        Raises:
            ModelInputError: If input validation fails (strict mode only)
            InvalidActionError: If output validation fails (strict mode only)

        Example:
            >>> action, probs = predictor.predict(observation)
            >>> print(f"Action: {action.name}")
            >>> print(f"Probabilities: SELL={probs[0]:.3f}, HOLD={probs[1]:.3f}, BUY={probs[2]:.3f}")
        """
        self._stats.total_predictions += 1

        # Validate input
        try:
            validated_obs = self._validate_input(observation)
        except ModelInputError:
            self._stats.input_validation_failures += 1
            raise

        # Get prediction and probabilities
        try:
            action_idx = self._get_raw_prediction(validated_obs)
            probabilities = self._get_action_probabilities(validated_obs)
        except Exception as e:
            self._stats.output_validation_failures += 1
            if self._strict_mode:
                raise InvalidActionError(f"Prediction failed: {e}") from e
            logger.warning(f"Prediction failed for {self._name}: {e}")
            return Action.HOLD, np.array([0.0, 1.0, 0.0], dtype=np.float32)

        # Validate output
        try:
            action, validated_probs = self._validate_output(action_idx, probabilities)
        except InvalidActionError:
            self._stats.output_validation_failures += 1
            raise

        self._stats.successful_predictions += 1
        return action, validated_probs

    def predict_batch(
        self,
        observations: np.ndarray,
    ) -> Tuple[List[Action], np.ndarray]:
        """
        Run validated predictions on a batch of observations.

        Args:
            observations: Feature matrix with shape (batch_size, OBSERVATION_DIM)

        Returns:
            Tuple of:
                - List of Action enums for each observation
                - Probability matrix with shape (batch_size, ACTION_COUNT)

        Raises:
            ModelInputError: If input validation fails (strict mode only)
            InvalidActionError: If output validation fails (strict mode only)

        Example:
            >>> actions, probs = predictor.predict_batch(observations)
            >>> for i, (action, prob) in enumerate(zip(actions, probs)):
            ...     print(f"Obs {i}: {action.name}, confidence={prob[action.value]:.3f}")
        """
        # Ensure 2D array
        if observations.ndim == 1:
            observations = observations.reshape(1, -1)

        batch_size = observations.shape[0]
        actions: List[Action] = []
        all_probabilities: List[np.ndarray] = []

        for i in range(batch_size):
            obs = observations[i:i+1]  # Keep 2D shape
            try:
                action, probs = self.predict(obs.squeeze(0))
                actions.append(action)
                all_probabilities.append(probs)
            except (ModelInputError, InvalidActionError) as e:
                if self._strict_mode:
                    raise
                logger.warning(f"Batch prediction {i} failed: {e}")
                actions.append(Action.HOLD)
                all_probabilities.append(np.array([0.0, 1.0, 0.0], dtype=np.float32))

        return actions, np.array(all_probabilities, dtype=np.float32)

    def reset_stats(self) -> None:
        """Reset prediction statistics."""
        self._stats = PredictionStats()

    def __repr__(self) -> str:
        return (
            f"ValidatedPredictor(name={self._name!r}, "
            f"strict_mode={self._strict_mode}, "
            f"total_predictions={self._stats.total_predictions})"
        )
