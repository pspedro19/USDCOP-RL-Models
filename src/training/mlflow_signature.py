"""
MLflow Model Signature - Proper Implementation
==============================================
Crea y attach la signature correctamente al modelo.
"""
import numpy as np
from typing import TYPE_CHECKING

from src.core.contracts.feature_contract import OBSERVATION_DIM
from src.core.contracts.action_contract import ACTION_COUNT

if TYPE_CHECKING:
    from mlflow.models.signature import ModelSignature


def create_model_signature() -> "ModelSignature":
    """Crea la signature MLflow correcta para el modelo."""
    from mlflow.models.signature import ModelSignature
    from mlflow.types.schema import Schema, TensorSpec

    input_schema = Schema([
        TensorSpec(np.dtype("float32"), shape=(-1, OBSERVATION_DIM), name="observation")
    ])
    output_schema = Schema([
        TensorSpec(np.dtype("float32"), shape=(-1, ACTION_COUNT), name="action_probabilities")
    ])
    return ModelSignature(inputs=input_schema, outputs=output_schema)


def create_input_example() -> np.ndarray:
    """Crea un input example real para el modelo."""
    example = np.array([
        0.001, 0.005, 0.01, 0.0, 0.5, 0.0, -0.5, 0.001,
        0.2, 0.3, 0.002, 0.0, -0.001, 0.0, 0.5,
    ], dtype=np.float32).reshape(1, -1)
    assert example.shape == (1, OBSERVATION_DIM)
    return example
