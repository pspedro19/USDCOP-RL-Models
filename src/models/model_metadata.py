"""
Model Metadata - Metadata capturada con cada prediccion.
CLAUDE-T13 | Plan Item: P1-2
Contrato: CTR-009

Captura informacion completa del estado del modelo y mercado
al momento de cada prediccion para auditoria y analisis.
"""

import numpy as np
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional, List, Dict, Any


@dataclass
class PredictionMetadata:
    """
    Metadata capturada con cada prediccion del modelo.
    Contrato: CTR-009

    Esta informacion se almacena en BD para:
    - Auditoria de decisiones
    - Analisis de drift
    - Debugging de comportamiento del modelo
    - Compliance y trazabilidad

    Ejemplo:
        metadata = capture_prediction_metadata(
            model=onnx_model,
            observation=obs,
            raw_features=raw,
            bid_ask_spread=0.0012,
            market_volatility=0.05
        )
        db.insert_trade_metadata(trade_id, metadata.to_dict())
    """
    # Model identification
    model_id: str
    model_version: str
    model_hash: str
    norm_stats_hash: str
    config_hash: Optional[str] = None

    # Features al momento de prediccion
    observation: List[float] = field(default_factory=list)  # 15 floats normalized
    raw_features: Dict[str, float] = field(default_factory=dict)  # 13 floats raw

    # Market state
    bid_ask_spread: float = 0.0
    market_volatility: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Prediccion
    action: int = 0  # 0=HOLD, 1=LONG, 2=SHORT
    action_probabilities: List[float] = field(default_factory=lambda: [0.33, 0.34, 0.33])
    value_estimate: float = 0.0
    entropy: float = 0.0
    confidence: float = 0.0

    # Additional context
    bar_idx: Optional[int] = None
    feature_contract_version: str = "v20"

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializa para almacenar en BD como JSONB.

        Returns:
            Dict JSON-serializable con todos los campos
        """
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PredictionMetadata':
        """
        Deserializa desde BD.

        Args:
            data: Dict con datos del metadata

        Returns:
            PredictionMetadata instance
        """
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
        return cls(**data)

    def validate(self) -> bool:
        """
        Valida que el metadata esta completo.

        Returns:
            True si todos los campos requeridos estan presentes

        Raises:
            ValueError: Si falta informacion critica
        """
        if not self.model_id:
            raise ValueError("model_id es requerido")
        if not self.model_hash:
            raise ValueError("model_hash es requerido")
        if len(self.observation) != 15:
            raise ValueError(f"observation debe tener 15 elementos, tiene {len(self.observation)}")
        if len(self.action_probabilities) != 3:
            raise ValueError(f"action_probabilities debe tener 3 elementos")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"confidence debe estar en [0, 1], es {self.confidence}")
        if not 0 <= self.action <= 2:
            raise ValueError(f"action debe ser 0, 1 o 2, es {self.action}")

        return True


def capture_prediction_metadata(
    model_id: str,
    model_version: str,
    model_hash: str,
    norm_stats_hash: str,
    observation: np.ndarray,
    raw_features: Dict[str, float],
    action: int,
    action_probabilities: List[float],
    value_estimate: float = 0.0,
    bid_ask_spread: float = 0.0,
    market_volatility: float = 0.0,
    bar_idx: Optional[int] = None,
    config_hash: Optional[str] = None
) -> PredictionMetadata:
    """
    Captura metadata completa de prediccion.

    Esta funcion debe ser llamada inmediatamente despues de
    cada prediccion del modelo para capturar el estado completo.

    Args:
        model_id: ID unico del modelo
        model_version: Version del modelo (e.g., "v20")
        model_hash: SHA256 del archivo ONNX
        norm_stats_hash: SHA256 del archivo norm_stats
        observation: Vector de features normalizadas (15 elementos)
        raw_features: Features sin normalizar (13 elementos)
        action: Accion predicha (0=HOLD, 1=LONG, 2=SHORT)
        action_probabilities: Probabilidades de cada accion
        value_estimate: Valor estimado por el critic
        bid_ask_spread: Spread actual del mercado
        market_volatility: Volatilidad actual
        bar_idx: Indice de la barra actual
        config_hash: Hash del config (opcional)

    Returns:
        PredictionMetadata con toda la informacion

    Example:
        metadata = capture_prediction_metadata(
            model_id="ppo_v20_abc12345",
            model_version="v20",
            model_hash="abcdef...",
            norm_stats_hash="123456...",
            observation=obs,
            raw_features=raw,
            action=1,
            action_probabilities=[0.1, 0.85, 0.05],
            bid_ask_spread=0.0012,
            market_volatility=0.05
        )
    """
    # Calcular entropy y confidence
    probs_array = np.array(action_probabilities)
    entropy = -float(np.sum(probs_array * np.log(probs_array + 1e-10)))
    confidence = float(np.max(probs_array))

    return PredictionMetadata(
        model_id=model_id,
        model_version=model_version,
        model_hash=model_hash,
        norm_stats_hash=norm_stats_hash,
        config_hash=config_hash,
        observation=observation.tolist() if isinstance(observation, np.ndarray) else list(observation),
        raw_features=raw_features,
        bid_ask_spread=float(bid_ask_spread),
        market_volatility=float(market_volatility),
        timestamp=datetime.utcnow(),
        action=int(action),
        action_probabilities=list(action_probabilities),
        value_estimate=float(value_estimate),
        entropy=entropy,
        confidence=confidence,
        bar_idx=bar_idx
    )


def compute_entropy(probabilities: List[float]) -> float:
    """
    Computa la entropia de una distribucion de probabilidades.

    Entropia alta = modelo incierto
    Entropia baja = modelo confiado

    Args:
        probabilities: Lista de probabilidades (deben sumar 1)

    Returns:
        Entropia en nats
    """
    probs = np.array(probabilities)
    # Evitar log(0)
    probs = np.clip(probs, 1e-10, 1.0)
    return -float(np.sum(probs * np.log(probs)))


def compute_confidence(probabilities: List[float]) -> float:
    """
    Computa la confianza como la probabilidad maxima.

    Args:
        probabilities: Lista de probabilidades

    Returns:
        Confianza en [0, 1]
    """
    return float(max(probabilities))


# Alias for backward compatibility
ModelMetadata = PredictionMetadata
capture_model_metadata = capture_prediction_metadata
