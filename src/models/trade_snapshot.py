"""
Trade Snapshot Schema - Features al momento del trade.
Contrato ID: CTR-004
CLAUDE-T4 | Plan Item: P0-8

Define el schema para features_snapshot que se almacena en BD.
"""

from pydantic import BaseModel, Field
from typing import Dict, Optional
from datetime import datetime
import hashlib


class FeaturesSnapshot(BaseModel):
    """
    Schema para features_snapshot en BD.
    Contrato ID: CTR-004

    Este schema define la estructura exacta del JSONB que se almacena
    en la columna features_snapshot de trades_history.

    Invariantes:
    - version debe ser "v20" (o version actual del contrato)
    - raw_features tiene 13 features (sin position, time_normalized)
    - normalized_features tiene 15 features (todas)
    """
    version: str = Field(
        default="v20",
        description="Version del Feature Contract usado"
    )
    timestamp: datetime = Field(
        description="Timestamp UTC del momento del trade"
    )
    bar_idx: int = Field(
        ge=0,
        description="Indice de la barra en el dataset"
    )
    raw_features: Dict[str, float] = Field(
        description="Features sin normalizar (13 features tecnicas/macro)"
    )
    normalized_features: Dict[str, float] = Field(
        description="Features normalizadas (15 features incluyendo position, time)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "version": "v20",
                "timestamp": "2026-01-11T15:30:00Z",
                "bar_idx": 288,
                "raw_features": {
                    "log_ret_5m": -0.0012,
                    "log_ret_1h": 0.0035,
                    "log_ret_4h": 0.0078,
                    "rsi_9": 45.2,
                    "atr_pct": 0.08,
                    "adx_14": 28.5,
                    "dxy_z": 0.5,
                    "dxy_change_1d": 0.002,
                    "vix_z": -0.3,
                    "embi_z": 0.1,
                    "brent_change_1d": -0.01,
                    "rate_spread": 5.5,
                    "usdmxn_change_1d": 0.003
                },
                "normalized_features": {
                    "log_ret_5m": -0.5,
                    "log_ret_1h": 0.8,
                    "log_ret_4h": 1.2,
                    "rsi_9": -0.3,
                    "atr_pct": 0.4,
                    "adx_14": -0.2,
                    "dxy_z": 0.5,
                    "dxy_change_1d": 0.2,
                    "vix_z": -0.3,
                    "embi_z": 0.1,
                    "brent_change_1d": -0.2,
                    "rate_spread": 0.6,
                    "usdmxn_change_1d": 0.15,
                    "position": 0.0,
                    "time_normalized": 0.5
                }
            }
        }
    }

    def validate_feature_count(self) -> bool:
        """Valida que el snapshot tenga el numero correcto de features."""
        raw_count = len(self.raw_features)
        norm_count = len(self.normalized_features)

        if raw_count != 13:
            raise ValueError(f"raw_features debe tener 13 features, tiene {raw_count}")
        if norm_count != 15:
            raise ValueError(f"normalized_features debe tener 15 features, tiene {norm_count}")

        return True

    def to_db_json(self) -> dict:
        """Serializa para almacenar en BD como JSONB."""
        return {
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "bar_idx": self.bar_idx,
            "raw_features": self.raw_features,
            "normalized_features": self.normalized_features
        }


class ModelMetadata(BaseModel):
    """
    Metadata del modelo al momento de la prediccion.
    Se almacena en la columna model_metadata de trades_history.
    """
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confianza de la prediccion (0-1)"
    )
    action_probs: list[float] = Field(
        min_length=3, max_length=3,
        description="Probabilidades [HOLD, LONG, SHORT]"
    )
    critic_value: float = Field(
        description="Valor estimado por el critic"
    )
    entropy: float = Field(
        ge=0.0,
        description="Entropia de la distribucion de acciones"
    )
    advantage: Optional[float] = Field(
        default=None,
        description="Advantage estimado (opcional)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "confidence": 0.85,
                "action_probs": [0.1, 0.85, 0.05],
                "critic_value": 0.023,
                "entropy": 0.32,
                "advantage": 0.015
            }
        }
    }


def compute_model_hash(model_bytes: bytes) -> str:
    """
    Computa SHA256 hash de los bytes del modelo ONNX.

    Args:
        model_bytes: Bytes del archivo modelo ONNX

    Returns:
        Hash SHA256 como string hexadecimal (64 chars)
    """
    return hashlib.sha256(model_bytes).hexdigest()


def compute_model_hash_from_file(filepath: str) -> str:
    """
    Computa SHA256 hash del archivo modelo.

    Args:
        filepath: Ruta al archivo modelo ONNX

    Returns:
        Hash SHA256 como string hexadecimal
    """
    with open(filepath, 'rb') as f:
        return compute_model_hash(f.read())
