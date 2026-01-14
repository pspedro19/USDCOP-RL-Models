"""
Feature Validation - Runtime validation of feature order.
CLAUDE-T14 | Plan Item: P1-5

Validates at runtime that feature order matches between
FeatureBuilder and ONNX model to prevent inference errors.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class FeatureOrderMismatchError(Exception):
    """
    Error critico: orden de features no coincide.

    Esto causara predicciones incorrectas ya que el modelo
    recibira features en orden diferente al esperado.
    """
    pass


class FeatureCountMismatchError(Exception):
    """
    Error critico: numero de features no coincide.
    """
    pass


def validate_feature_order_at_startup(
    builder,
    model_path: Path,
    strict: bool = True
) -> bool:
    """
    Valida orden de features al iniciar el servicio.

    Args:
        builder: FeatureBuilder instance
        model_path: Path to ONNX model
        strict: If True, raise exception on mismatch

    Returns:
        True if validation passes

    Raises:
        FeatureOrderMismatchError: If feature order differs
        FileNotFoundError: If model file doesn't exist
    """
    if not model_path.exists():
        if strict:
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        logger.warning(f"Modelo no encontrado: {model_path}")
        return False

    # Obtener orden del builder
    builder_order = list(builder.get_feature_names())

    # Intentar cargar metadata del modelo ONNX
    model_order = _extract_feature_order_from_onnx(model_path)

    if model_order is None:
        # Modelo sin metadata - log warning pero continuar
        logger.warning(
            f"Modelo {model_path} sin feature_order metadata - "
            "no se puede validar orden. Asumiendo correcto."
        )
        return True

    # Validar coincidencia
    if len(builder_order) != len(model_order):
        msg = (
            f"Feature count mismatch!\n"
            f"Builder: {len(builder_order)} features\n"
            f"Model: {len(model_order)} features"
        )
        if strict:
            raise FeatureCountMismatchError(msg)
        logger.error(msg)
        return False

    if builder_order != model_order:
        # Encontrar diferencias
        diffs = []
        for i, (b, m) in enumerate(zip(builder_order, model_order)):
            if b != m:
                diffs.append(f"  [{i}] builder='{b}' vs model='{m}'")

        msg = (
            f"Feature order mismatch!\n"
            f"Builder: {builder_order}\n"
            f"Model: {model_order}\n"
            f"Diferencias:\n" + "\n".join(diffs) + "\n"
            f"Esto causara predicciones incorrectas."
        )
        if strict:
            raise FeatureOrderMismatchError(msg)
        logger.error(msg)
        return False

    logger.info(f"Feature order validado: {len(builder_order)} features")
    return True


def _extract_feature_order_from_onnx(model_path: Path) -> Optional[List[str]]:
    """
    Extrae feature_order del metadata del modelo ONNX.

    Args:
        model_path: Path to ONNX model

    Returns:
        List of feature names or None if not found
    """
    try:
        import onnx
    except ImportError:
        logger.warning("onnx no instalado - no se puede validar feature order")
        return None

    try:
        model = onnx.load(str(model_path))

        for prop in model.metadata_props:
            if prop.key == "feature_order":
                return json.loads(prop.value)

        return None

    except Exception as e:
        logger.warning(f"Error leyendo metadata de ONNX: {e}")
        return None


def validate_observation_shape(
    observation,
    expected_dim: int,
    strict: bool = True
) -> bool:
    """
    Valida que el observation tiene la dimension correcta.

    Args:
        observation: numpy array
        expected_dim: Expected dimension
        strict: If True, raise exception on mismatch

    Returns:
        True if valid

    Raises:
        ValueError: If shape doesn't match
    """
    import numpy as np

    if not isinstance(observation, np.ndarray):
        msg = f"observation debe ser np.ndarray, recibido {type(observation)}"
        if strict:
            raise TypeError(msg)
        logger.error(msg)
        return False

    if observation.shape != (expected_dim,):
        msg = f"Shape incorrecto: {observation.shape} vs ({expected_dim},)"
        if strict:
            raise ValueError(msg)
        logger.error(msg)
        return False

    return True


def validate_no_nan_inf(observation, strict: bool = True) -> bool:
    """
    Valida que observation no contenga NaN o Inf.

    Args:
        observation: numpy array
        strict: If True, raise exception on invalid values

    Returns:
        True if valid
    """
    import numpy as np

    if np.isnan(observation).any():
        nan_indices = np.where(np.isnan(observation))[0].tolist()
        msg = f"NaN detectado en indices: {nan_indices}"
        if strict:
            raise ValueError(msg)
        logger.error(msg)
        return False

    if np.isinf(observation).any():
        inf_indices = np.where(np.isinf(observation))[0].tolist()
        msg = f"Inf detectado en indices: {inf_indices}"
        if strict:
            raise ValueError(msg)
        logger.error(msg)
        return False

    return True


def create_startup_validator(builder, model_path: Path):
    """
    Crea un validator que se ejecuta una vez al startup.

    Usage:
        validator = create_startup_validator(builder, model_path)
        validator()  # Raises on error
    """
    def validate():
        validate_feature_order_at_startup(builder, model_path, strict=True)
        logger.info("Startup validation completada exitosamente")

    return validate
