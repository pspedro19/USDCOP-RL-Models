"""
FEATURE CONTRACTS REGISTRY - Versioned Feature Configurations
===============================================================

Este módulo resuelve el problema de experimentos con diferentes features:
- Cada experimento referencia un CONTRACT_ID
- El contract define el orden exacto y dimensión
- Inference API valida que modelo y contract coincidan

Arquitectura:
    Experiment YAML          Feature Contract           Model
    ─────────────────        ────────────────────       ─────────────
    contract_id: v1.0.0  ──▶ CONTRACTS["v1.0.0"]   ──▶ observation_dim=15
    contract_id: v1.1.0  ──▶ CONTRACTS["v1.1.0"]   ──▶ observation_dim=12

Reglas SSOT:
    1. CANONICAL_CONTRACT_ID define el contrato de producción
    2. Experimentos PUEDEN usar contratos diferentes
    3. Inference API verifica contract_id del modelo vs request
    4. Un modelo SOLO puede usarse con su contract original

Author: Trading Team
Date: 2026-01-18
Contract: CTR-FEATURE-REGISTRY-001
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Final, Optional
from enum import Enum
import hashlib


class FeatureCategory(Enum):
    """Categorías de features para organización."""
    TECHNICAL = "technical"
    MACRO = "macro"
    STATE = "state"


@dataclass(frozen=True)
class FeatureContract:
    """
    Contrato inmutable de features para un experimento/modelo.

    Attributes:
        contract_id: Identificador único versionado (e.g., "v1.0.0")
        name: Nombre descriptivo del contrato
        description: Descripción del propósito
        feature_order: Tupla ordenada de nombres de features
        observation_dim: Dimensión total (len(feature_order))
        market_features: Features de mercado (técnicos + macro)
        state_features: Features de estado (position, time)
        is_production: Si es el contrato de producción actual
    """
    contract_id: str
    name: str
    description: str
    feature_order: Tuple[str, ...]
    market_features: Tuple[str, ...]
    state_features: Tuple[str, ...] = ("position", "time_normalized")
    is_production: bool = False
    deprecated: bool = False
    superseded_by: Optional[str] = None

    @property
    def observation_dim(self) -> int:
        return len(self.feature_order)

    @property
    def market_dim(self) -> int:
        return len(self.market_features)

    @property
    def state_dim(self) -> int:
        return len(self.state_features)

    @property
    def hash(self) -> str:
        """Hash del feature order para validación."""
        return hashlib.sha256(
            ",".join(self.feature_order).encode()
        ).hexdigest()[:16]

    def validate_observation(self, obs_dim: int) -> bool:
        """Valida que la dimensión de observación coincida."""
        return obs_dim == self.observation_dim

    def to_dict(self) -> Dict:
        """Serializa el contrato para logging/MLflow."""
        return {
            "contract_id": self.contract_id,
            "name": self.name,
            "observation_dim": self.observation_dim,
            "market_dim": self.market_dim,
            "state_dim": self.state_dim,
            "feature_order": list(self.feature_order),
            "hash": self.hash,
            "is_production": self.is_production,
        }


# =============================================================================
# FEATURE CONTRACTS DEFINITIONS
# =============================================================================

# Estado features (común a todos los contratos)
STATE_FEATURES: Final[Tuple[str, ...]] = ("position", "time_normalized")

# -----------------------------------------------------------------------------
# v1.0.0 - PRODUCTION CONTRACT (15 features)
# -----------------------------------------------------------------------------
CONTRACT_V1_0_0 = FeatureContract(
    contract_id="v1.0.0",
    name="Full Macro (Production)",
    description="Contrato de producción con 13 market features + 2 state. "
                "Incluye todas las variables macro disponibles.",
    feature_order=(
        # Technical (6)
        "log_ret_5m", "log_ret_1h", "log_ret_4h",
        "rsi_9", "atr_pct", "adx_14",
        # Macro Full (7)
        "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
        "brent_change_1d", "rate_spread", "usdmxn_change_1d",
        # State (2)
        "position", "time_normalized",
    ),
    market_features=(
        "log_ret_5m", "log_ret_1h", "log_ret_4h",
        "rsi_9", "atr_pct", "adx_14",
        "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
        "brent_change_1d", "rate_spread", "usdmxn_change_1d",
    ),
    is_production=True,
)

# -----------------------------------------------------------------------------
# v1.1.0 - REDUCED MACRO CONTRACT (12 features)
# -----------------------------------------------------------------------------
CONTRACT_V1_1_0 = FeatureContract(
    contract_id="v1.1.0",
    name="Core Macro (Reduced)",
    description="Contrato reducido con 10 market features + 2 state. "
                "Elimina features macro redundantes: dxy_change_1d, rate_spread, usdmxn_change_1d.",
    feature_order=(
        # Technical (6) - Same as v1.0.0
        "log_ret_5m", "log_ret_1h", "log_ret_4h",
        "rsi_9", "atr_pct", "adx_14",
        # Macro Core (4) - Reduced
        "dxy_z", "vix_z", "embi_z", "brent_change_1d",
        # State (2)
        "position", "time_normalized",
    ),
    market_features=(
        "log_ret_5m", "log_ret_1h", "log_ret_4h",
        "rsi_9", "atr_pct", "adx_14",
        "dxy_z", "vix_z", "embi_z", "brent_change_1d",
    ),
    is_production=False,
)

# -----------------------------------------------------------------------------
# v1.2.0 - TECHNICAL ONLY CONTRACT (8 features)
# -----------------------------------------------------------------------------
CONTRACT_V1_2_0 = FeatureContract(
    contract_id="v1.2.0",
    name="Technical Only",
    description="Contrato sin features macro. Solo técnicos + state. "
                "Para ablation studies de impacto macro.",
    feature_order=(
        # Technical (6)
        "log_ret_5m", "log_ret_1h", "log_ret_4h",
        "rsi_9", "atr_pct", "adx_14",
        # State (2)
        "position", "time_normalized",
    ),
    market_features=(
        "log_ret_5m", "log_ret_1h", "log_ret_4h",
        "rsi_9", "atr_pct", "adx_14",
    ),
    is_production=False,
)


# =============================================================================
# CONTRACTS REGISTRY
# =============================================================================

CONTRACTS: Final[Dict[str, FeatureContract]] = {
    "v1.0.0": CONTRACT_V1_0_0,
    "v1.1.0": CONTRACT_V1_1_0,
    "v1.2.0": CONTRACT_V1_2_0,
}

# Canonical contract for production
CANONICAL_CONTRACT_ID: Final[str] = "v1.0.0"
CANONICAL_CONTRACT: Final[FeatureContract] = CONTRACTS[CANONICAL_CONTRACT_ID]

# Backward compatibility exports
FEATURE_ORDER: Final[Tuple[str, ...]] = CANONICAL_CONTRACT.feature_order
OBSERVATION_DIM: Final[int] = CANONICAL_CONTRACT.observation_dim
FEATURE_ORDER_HASH: Final[str] = CANONICAL_CONTRACT.hash


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_contract(contract_id: str) -> FeatureContract:
    """
    Obtiene un contrato por ID.

    Args:
        contract_id: ID del contrato (e.g., "v1.0.0")

    Returns:
        FeatureContract

    Raises:
        KeyError: Si el contrato no existe

    Example:
        contract = get_contract("v1.1.0")
        print(contract.observation_dim)  # 12
    """
    if contract_id not in CONTRACTS:
        available = list(CONTRACTS.keys())
        raise KeyError(
            f"Contract '{contract_id}' not found. "
            f"Available contracts: {available}"
        )
    return CONTRACTS[contract_id]


def get_production_contract() -> FeatureContract:
    """Obtiene el contrato de producción actual."""
    return CANONICAL_CONTRACT


def validate_model_contract(
    model_contract_id: str,
    request_contract_id: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Valida compatibilidad entre modelo y request.

    Args:
        model_contract_id: Contract ID con el que se entrenó el modelo
        request_contract_id: Contract ID del request (None = usa mismo del modelo)

    Returns:
        (is_valid, error_message)

    Example:
        valid, msg = validate_model_contract("v1.0.0", "v1.1.0")
        if not valid:
            raise ValueError(msg)
    """
    if request_contract_id is None:
        return True, ""

    if model_contract_id != request_contract_id:
        return False, (
            f"Contract mismatch: model trained with '{model_contract_id}' "
            f"but request uses '{request_contract_id}'. "
            f"Models are only compatible with their training contract."
        )

    return True, ""


def list_contracts() -> List[Dict]:
    """Lista todos los contratos disponibles."""
    return [c.to_dict() for c in CONTRACTS.values()]


def get_norm_stats_path(contract_id: str) -> str:
    """
    Retorna la ruta del archivo norm_stats para un contrato.

    Cada contrato tiene su propio archivo de estadísticas de normalización
    porque las features son diferentes.
    """
    if contract_id == "v1.0.0":
        return "config/norm_stats.json"
    else:
        return f"config/norm_stats_{contract_id.replace('.', '_')}.json"


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Classes
    "FeatureContract",
    "FeatureCategory",
    # Contracts
    "CONTRACT_V1_0_0",
    "CONTRACT_V1_1_0",
    "CONTRACT_V1_2_0",
    "CONTRACTS",
    # Canonical
    "CANONICAL_CONTRACT_ID",
    "CANONICAL_CONTRACT",
    # Backward compatibility
    "FEATURE_ORDER",
    "OBSERVATION_DIM",
    "FEATURE_ORDER_HASH",
    # Functions
    "get_contract",
    "get_production_contract",
    "validate_model_contract",
    "list_contracts",
    "get_norm_stats_path",
]
