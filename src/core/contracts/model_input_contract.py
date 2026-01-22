"""
MODEL INPUT CONTRACT - Single Source of Truth
==============================================
Define el contrato formal del input que espera el modelo.

Contract ID: CTR-MODEL-INPUT-001
Version: 1.0.0
"""
from dataclasses import dataclass
from typing import Tuple, Final, Optional, List
import numpy as np

from .feature_contract import (
    FEATURE_CONTRACT,
    FEATURE_ORDER,
    OBSERVATION_DIM,
    FeatureContractError,
)


@dataclass(frozen=True)
class ModelInputContract:
    """Contrato formal del input del modelo."""
    observation_dim: int = OBSERVATION_DIM
    supports_batch: bool = True
    dtype: np.dtype = np.float32
    requires_normalized: bool = True
    clip_range: Tuple[float, float] = (-5.0, 5.0)
    feature_order: Tuple[str, ...] = FEATURE_ORDER

    def validate(
        self,
        observation: np.ndarray,
        norm_stats_hash: Optional[str] = None,
        expected_hash: Optional[str] = None,
    ) -> None:
        """Valida que la observación cumple el contrato."""
        # Normalize shape to 2D
        if observation.ndim == 1:
            obs_2d = observation.reshape(1, -1)
        elif observation.ndim == 2:
            obs_2d = observation
        else:
            raise ModelInputError(f"Invalid ndim: {observation.ndim}. Expected 1 or 2.")

        # Check feature dimension
        if obs_2d.shape[1] != self.observation_dim:
            raise ModelInputError(f"Invalid feature dim: {obs_2d.shape[1]}. Expected {self.observation_dim}.")

        # Check dtype
        if observation.dtype != self.dtype:
            raise ModelInputError(f"Invalid dtype: {observation.dtype}. Expected {self.dtype}.")

        # Check for NaN/Inf
        if np.any(np.isnan(obs_2d)):
            nan_locs = np.argwhere(np.isnan(obs_2d))
            raise ModelInputError(f"Observation contains NaN at: {nan_locs.tolist()}")

        if np.any(np.isinf(obs_2d)):
            inf_locs = np.argwhere(np.isinf(obs_2d))
            raise ModelInputError(f"Observation contains Inf at: {inf_locs.tolist()}")

        # Check clip range
        min_val, max_val = self.clip_range
        if np.any(obs_2d < min_val) or np.any(obs_2d > max_val):
            out_of_range = np.argwhere((obs_2d < min_val) | (obs_2d > max_val))
            raise ModelInputError(f"Values outside [{min_val}, {max_val}] at: {out_of_range.tolist()}")

        # Check norm_stats_hash
        if norm_stats_hash and expected_hash and norm_stats_hash != expected_hash:
            raise ModelInputError(f"norm_stats_hash mismatch: {norm_stats_hash[:16]}... != {expected_hash[:16]}...")


class ModelInputError(ValueError):
    """Error de input inválido al modelo."""
    pass


MODEL_INPUT_CONTRACT: Final[ModelInputContract] = ModelInputContract()


class ObservationValidator:
    """Wrapper de validación de observaciones para TODOS los paths de inferencia."""

    def __init__(
        self,
        expected_norm_stats_hash: Optional[str] = None,
        strict_mode: bool = True,
    ):
        self._contract = MODEL_INPUT_CONTRACT
        self._expected_hash = expected_norm_stats_hash
        self._strict = strict_mode
        self._validation_count = 0
        self._error_count = 0

    def validate_and_prepare(
        self,
        observation: np.ndarray,
        norm_stats_hash: Optional[str] = None,
    ) -> np.ndarray:
        """Valida y prepara una observación para el modelo."""
        self._validation_count += 1

        try:
            # Ensure numpy array
            if not isinstance(observation, np.ndarray):
                observation = np.array(observation)

            # Cast to float32
            if observation.dtype != np.float32:
                observation = observation.astype(np.float32)

            # Ensure 1D for single inference
            if observation.ndim == 2 and observation.shape[0] == 1:
                observation = observation.squeeze(0)

            # Validate
            self._contract.validate(
                observation,
                norm_stats_hash=norm_stats_hash,
                expected_hash=self._expected_hash,
            )

            return observation

        except ModelInputError as e:
            self._error_count += 1
            if self._strict:
                raise
            else:
                import logging
                logging.warning(f"Observation validation warning: {e}")
                return observation.astype(np.float32)

    @property
    def stats(self) -> dict:
        return {
            "validation_count": self._validation_count,
            "error_count": self._error_count,
            "error_rate": self._error_count / max(1, self._validation_count),
        }


def validate_model_input(
    observation: np.ndarray,
    norm_stats_hash: Optional[str] = None,
    expected_hash: Optional[str] = None,
) -> np.ndarray:
    """Función de conveniencia para validar input del modelo."""
    if not isinstance(observation, np.ndarray):
        observation = np.array(observation)

    if observation.dtype != np.float32:
        observation = observation.astype(np.float32)

    MODEL_INPUT_CONTRACT.validate(
        observation,
        norm_stats_hash=norm_stats_hash,
        expected_hash=expected_hash,
    )

    return observation
