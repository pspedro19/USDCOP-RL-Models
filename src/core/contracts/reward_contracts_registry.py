"""
REWARD CONTRACTS REGISTRY - Versioned Reward Configurations
=============================================================

Este módulo define contratos versionados para configuraciones de reward.
Permite que diferentes experimentos usen diferentes configuraciones mientras
mantiene trazabilidad completa.

Arquitectura:
    Experiment YAML          Reward Contract            Training
    ─────────────────        ────────────────────       ─────────────
    reward_contract: v1.0.0  ──▶ CONTRACTS["v1.0.0"]   ──▶ DSR + Sortino + Regime
    reward_contract: v1.1.0  ──▶ CONTRACTS["v1.1.0"]   ──▶ DSR only + curriculum

Reglas SSOT:
    1. CANONICAL_REWARD_CONTRACT_ID define el contrato de producción
    2. Experimentos PUEDEN usar contratos diferentes
    3. Model registry guarda reward_contract_id + reward_config_hash
    4. Un modelo DEBE usar su reward_contract original para backtesting

Author: Trading Team
Date: 2026-01-19
Contract: CTR-REWARD-REGISTRY-001
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Final, Optional, Any, FrozenSet
from enum import Enum
import hashlib
import json

from src.training.config import (
    RewardConfig,
    DSRConfig,
    SortinoConfig,
    RegimeConfig,
    MarketImpactConfig,
    HoldingDecayConfig,
    AntiGamingConfig,
    NormalizerConfig,
    BanrepDetectorConfig,
    OilCorrelationConfig,
    PnLTransformConfig,
    CurriculumConfig,
)


class RewardContractType(Enum):
    """Types of reward contracts."""
    PRODUCTION = "production"        # Full production configuration
    EXPERIMENT = "experiment"        # Experimental configuration
    ABLATION = "ablation"           # Ablation study configuration
    LEGACY = "legacy"               # Backward compatibility


@dataclass(frozen=True)
class RewardContract:
    """
    Immutable reward configuration contract.

    Attributes:
        contract_id: Unique versioned identifier (e.g., "v1.0.0")
        name: Descriptive name
        description: Purpose description
        config: Complete RewardConfig
        enabled_components: Frozenset of enabled component names
        contract_type: Type classification
        is_production: Whether this is the production contract
        deprecated: Whether this contract is deprecated
        superseded_by: ID of contract that supersedes this one
    """
    contract_id: str
    name: str
    description: str
    config: RewardConfig
    enabled_components: FrozenSet[str]
    contract_type: RewardContractType = RewardContractType.EXPERIMENT
    is_production: bool = False
    deprecated: bool = False
    superseded_by: Optional[str] = None

    @property
    def hash(self) -> str:
        """Deterministic hash of reward configuration."""
        return self.config.to_hash()

    @property
    def component_count(self) -> int:
        """Number of enabled components."""
        return len(self.enabled_components)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize contract for logging/MLflow."""
        return {
            "contract_id": self.contract_id,
            "name": self.name,
            "description": self.description,
            "hash": self.hash,
            "component_count": self.component_count,
            "enabled_components": sorted(self.enabled_components),
            "contract_type": self.contract_type.value,
            "is_production": self.is_production,
            "deprecated": self.deprecated,
            "config": self.config.to_dict(),
        }

    def get_component_weights(self) -> Dict[str, float]:
        """Get weights for all components."""
        return {
            "pnl": self.config.weight_pnl,
            "dsr": self.config.weight_dsr,
            "sortino": self.config.weight_sortino,
            "regime_penalty": self.config.weight_regime_penalty,
            "holding_decay": self.config.weight_holding_decay,
            "anti_gaming": self.config.weight_anti_gaming,
        }


# =============================================================================
# COMPONENT SETS
# =============================================================================

# Full component set
FULL_COMPONENTS: Final[FrozenSet[str]] = frozenset({
    "dsr", "sortino", "regime_detector", "market_impact",
    "holding_decay", "gap_risk", "inactivity", "churn",
    "action_correlation", "bias_detector", "pnl_transform",
    "reward_normalizer", "banrep_detector",
})

# Core components (essential for stable training)
CORE_COMPONENTS: Final[FrozenSet[str]] = frozenset({
    "dsr", "sortino", "regime_detector", "market_impact",
    "holding_decay", "pnl_transform", "reward_normalizer",
})

# Minimal components (for ablation studies)
MINIMAL_COMPONENTS: Final[FrozenSet[str]] = frozenset({
    "dsr", "pnl_transform",
})


# =============================================================================
# REWARD CONTRACTS DEFINITIONS
# =============================================================================

# -----------------------------------------------------------------------------
# v1.0.0 - PRODUCTION CONTRACT (Full Components)
# -----------------------------------------------------------------------------
CONTRACT_V1_0_0 = RewardContract(
    contract_id="v1.0.0",
    name="Full Production",
    description=(
        "Contrato de producción completo con todos los componentes. "
        "Incluye DSR, Sortino, régimen, market impact (Almgren-Chriss), "
        "holding decay, anti-gaming, y Banrep detector."
    ),
    config=RewardConfig(
        # DSR - Differential Sharpe Ratio
        dsr=DSRConfig(eta=0.01, min_samples=10, scale=1.0, weight=0.3),
        # Sortino - Downside risk focus
        sortino=SortinoConfig(window_size=20, target_return=0.0, min_samples=5, scale=1.0, weight=0.2),
        # Regime detection
        regime=RegimeConfig(
            low_vol_percentile=25, high_vol_percentile=75,
            crisis_multiplier=1.5, min_stability=3,
            history_window=500, smoothing_window=5,
        ),
        # Market impact (Almgren-Chriss)
        market_impact=MarketImpactConfig(
            permanent_impact_coef=0.1, temporary_impact_coef=0.3,
            volatility_impact_coef=0.15, adv_base_usd=50_000_000.0,
            typical_order_fraction=0.001, default_spread_bps=100,
        ),
        # Holding decay
        holding_decay=HoldingDecayConfig(
            half_life_bars=48, max_penalty=0.3,
            flat_threshold=0, enable_overnight_boost=True,
            overnight_multiplier=1.5,
        ),
        # Anti-gaming
        anti_gaming=AntiGamingConfig(
            inactivity_grace_period=12, inactivity_max_penalty=0.2,
            inactivity_growth_rate=0.01, churn_window_size=20,
            churn_max_trades=10, churn_base_penalty=0.1,
            churn_excess_penalty=0.02, bias_imbalance_threshold=0.75,
            bias_penalty=0.1, bias_min_samples=50,
        ),
        # Normalizer (FinRL-Meta)
        normalizer=NormalizerConfig(
            decay=0.99, epsilon=1e-8, clip_range=10.0,
            warmup_steps=1000, per_episode_reset=False,
        ),
        # Banrep detector
        banrep=BanrepDetectorConfig(
            volatility_spike_zscore=3.0, volatility_baseline_window=100,
            intervention_penalty=0.5, cooldown_bars=24,
            reversal_threshold=0.02, min_history=50,
        ),
        # Oil correlation (disabled by default)
        oil_correlation=OilCorrelationConfig(),
        # PnL transform
        pnl_transform=PnLTransformConfig(
            transform_type="zscore", clip_min=-0.1, clip_max=0.1,
            zscore_window=100, zscore_clip=3.0,
            asymmetric_win_mult=1.0, asymmetric_loss_mult=1.5,
        ),
        # Curriculum
        curriculum=CurriculumConfig(
            enabled=True, phase_1_steps=100_000,
            phase_2_steps=200_000, phase_3_steps=300_000,
            phase_1_cost_mult=0.5, phase_2_cost_mult=0.75,
            phase_3_cost_mult=1.0,
        ),
        # Weights
        weight_pnl=0.5,
        weight_dsr=0.3,
        weight_sortino=0.2,
        weight_regime_penalty=1.0,
        weight_holding_decay=1.0,
        weight_anti_gaming=1.0,
        # Global flags
        enable_normalization=True,
        enable_curriculum=True,
        enable_banrep_detection=True,
        enable_oil_tracking=False,
    ),
    enabled_components=FULL_COMPONENTS,
    contract_type=RewardContractType.PRODUCTION,
    is_production=True,
)

# -----------------------------------------------------------------------------
# v1.1.0 - CORE ONLY CONTRACT (No Anti-Gaming)
# -----------------------------------------------------------------------------
CONTRACT_V1_1_0 = RewardContract(
    contract_id="v1.1.0",
    name="Core Components",
    description=(
        "Contrato con componentes core solamente. "
        "Sin anti-gaming penalties. Para experimentos de baseline."
    ),
    config=RewardConfig(
        dsr=DSRConfig(eta=0.01, min_samples=10, scale=1.0, weight=0.4),
        sortino=SortinoConfig(window_size=20, target_return=0.0, min_samples=5, scale=1.0, weight=0.3),
        regime=RegimeConfig(),
        market_impact=MarketImpactConfig(),
        holding_decay=HoldingDecayConfig(max_penalty=0.2),  # Lower penalty
        anti_gaming=AntiGamingConfig(),  # Defaults but not heavily used
        normalizer=NormalizerConfig(),
        banrep=BanrepDetectorConfig(),
        oil_correlation=OilCorrelationConfig(),
        pnl_transform=PnLTransformConfig(),
        curriculum=CurriculumConfig(enabled=False),  # No curriculum
        weight_pnl=0.4,
        weight_dsr=0.4,
        weight_sortino=0.2,
        weight_regime_penalty=1.0,
        weight_holding_decay=0.5,  # Reduced
        weight_anti_gaming=0.0,  # Disabled
        enable_normalization=True,
        enable_curriculum=False,
        enable_banrep_detection=True,
        enable_oil_tracking=False,
    ),
    enabled_components=CORE_COMPONENTS,
    contract_type=RewardContractType.EXPERIMENT,
    is_production=False,
)

# -----------------------------------------------------------------------------
# v1.2.0 - DSR ONLY CONTRACT (Ablation)
# -----------------------------------------------------------------------------
CONTRACT_V1_2_0 = RewardContract(
    contract_id="v1.2.0",
    name="DSR Only (Ablation)",
    description=(
        "Contrato mínimo con solo DSR. "
        "Para ablation studies del impacto de componentes adicionales."
    ),
    config=RewardConfig(
        dsr=DSRConfig(eta=0.01, min_samples=10, scale=1.0, weight=1.0),
        sortino=SortinoConfig(),
        regime=RegimeConfig(),
        market_impact=MarketImpactConfig(),
        holding_decay=HoldingDecayConfig(),
        anti_gaming=AntiGamingConfig(),
        normalizer=NormalizerConfig(),
        banrep=BanrepDetectorConfig(),
        oil_correlation=OilCorrelationConfig(),
        pnl_transform=PnLTransformConfig(transform_type="clipped"),
        curriculum=CurriculumConfig(enabled=False),
        weight_pnl=0.0,
        weight_dsr=1.0,
        weight_sortino=0.0,
        weight_regime_penalty=0.0,
        weight_holding_decay=0.0,
        weight_anti_gaming=0.0,
        enable_normalization=True,
        enable_curriculum=False,
        enable_banrep_detection=False,
        enable_oil_tracking=False,
    ),
    enabled_components=MINIMAL_COMPONENTS,
    contract_type=RewardContractType.ABLATION,
    is_production=False,
)

# -----------------------------------------------------------------------------
# v1.3.0 - HIGH PENALTY CONTRACT (Aggressive Anti-Gaming)
# -----------------------------------------------------------------------------
CONTRACT_V1_3_0 = RewardContract(
    contract_id="v1.3.0",
    name="High Penalty (Aggressive)",
    description=(
        "Contrato con penalidades agresivas. "
        "Para entrenar agentes más conservadores."
    ),
    config=RewardConfig(
        dsr=DSRConfig(eta=0.01, min_samples=10, scale=1.0, weight=0.3),
        sortino=SortinoConfig(window_size=20, target_return=0.0, min_samples=5, scale=1.0, weight=0.2),
        regime=RegimeConfig(crisis_multiplier=2.0),  # Higher crisis penalty
        market_impact=MarketImpactConfig(
            permanent_impact_coef=0.15,  # Higher impact
            temporary_impact_coef=0.4,
            default_spread_bps=120,  # Wider spread
        ),
        holding_decay=HoldingDecayConfig(
            half_life_bars=24,  # Faster decay
            max_penalty=0.5,   # Higher max
        ),
        anti_gaming=AntiGamingConfig(
            inactivity_max_penalty=0.3,  # Higher penalties
            churn_base_penalty=0.15,
            bias_penalty=0.15,
        ),
        normalizer=NormalizerConfig(),
        banrep=BanrepDetectorConfig(
            intervention_penalty=0.7,  # Higher banrep penalty
        ),
        oil_correlation=OilCorrelationConfig(),
        pnl_transform=PnLTransformConfig(
            asymmetric_loss_mult=2.0,  # Higher loss aversion
        ),
        curriculum=CurriculumConfig(enabled=True),
        weight_pnl=0.4,
        weight_dsr=0.3,
        weight_sortino=0.2,
        weight_regime_penalty=1.5,  # Increased
        weight_holding_decay=1.5,  # Increased
        weight_anti_gaming=1.5,  # Increased
        enable_normalization=True,
        enable_curriculum=True,
        enable_banrep_detection=True,
        enable_oil_tracking=False,
    ),
    enabled_components=FULL_COMPONENTS,
    contract_type=RewardContractType.EXPERIMENT,
    is_production=False,
)


# =============================================================================
# CONTRACTS REGISTRY
# =============================================================================

REWARD_CONTRACTS: Final[Dict[str, RewardContract]] = {
    "v1.0.0": CONTRACT_V1_0_0,
    "v1.1.0": CONTRACT_V1_1_0,
    "v1.2.0": CONTRACT_V1_2_0,
    "v1.3.0": CONTRACT_V1_3_0,
}

# Canonical contract for production
CANONICAL_REWARD_CONTRACT_ID: Final[str] = "v1.0.0"
CANONICAL_REWARD_CONTRACT: Final[RewardContract] = REWARD_CONTRACTS[CANONICAL_REWARD_CONTRACT_ID]

# Backward compatibility
DEFAULT_REWARD_CONFIG: Final[RewardConfig] = CANONICAL_REWARD_CONTRACT.config
REWARD_CONFIG_HASH: Final[str] = CANONICAL_REWARD_CONTRACT.hash


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_reward_contract(contract_id: str) -> RewardContract:
    """
    Get a reward contract by ID.

    Args:
        contract_id: Contract ID (e.g., "v1.0.0")

    Returns:
        RewardContract

    Raises:
        KeyError: If contract not found

    Example:
        contract = get_reward_contract("v1.1.0")
        print(contract.config.weight_dsr)  # 0.4
    """
    if contract_id not in REWARD_CONTRACTS:
        available = list(REWARD_CONTRACTS.keys())
        raise KeyError(
            f"Reward contract '{contract_id}' not found. "
            f"Available contracts: {available}"
        )
    return REWARD_CONTRACTS[contract_id]


def get_production_reward_contract() -> RewardContract:
    """Get the current production reward contract."""
    return CANONICAL_REWARD_CONTRACT


def validate_reward_contract(
    model_contract_id: str,
    request_contract_id: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Validate compatibility between model and request reward contracts.

    Args:
        model_contract_id: Contract ID used during training
        request_contract_id: Contract ID for request (None = use model's)

    Returns:
        (is_valid, error_message)
    """
    if request_contract_id is None:
        return True, ""

    if model_contract_id != request_contract_id:
        return False, (
            f"Reward contract mismatch: model trained with '{model_contract_id}' "
            f"but request uses '{request_contract_id}'. "
            f"Models must use their training reward contract for consistent behavior."
        )

    return True, ""


def list_reward_contracts() -> List[Dict[str, Any]]:
    """List all available reward contracts."""
    return [c.to_dict() for c in REWARD_CONTRACTS.values()]


def create_custom_reward_contract(
    contract_id: str,
    name: str,
    description: str,
    config: RewardConfig,
    enabled_components: Optional[FrozenSet[str]] = None,
) -> RewardContract:
    """
    Create a custom reward contract for experimentation.

    Args:
        contract_id: Unique identifier (should not conflict with existing)
        name: Descriptive name
        description: Purpose description
        config: RewardConfig instance
        enabled_components: Components to enable (defaults to CORE_COMPONENTS)

    Returns:
        New RewardContract (not added to registry)

    Note:
        Custom contracts are NOT added to the global registry.
        They should be tracked through experiment metadata.
    """
    if contract_id in REWARD_CONTRACTS:
        raise ValueError(
            f"Contract ID '{contract_id}' already exists in registry. "
            f"Use a different ID for custom contracts."
        )

    return RewardContract(
        contract_id=contract_id,
        name=name,
        description=description,
        config=config,
        enabled_components=enabled_components or CORE_COMPONENTS,
        contract_type=RewardContractType.EXPERIMENT,
        is_production=False,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Classes
    "RewardContract",
    "RewardContractType",
    # Component sets
    "FULL_COMPONENTS",
    "CORE_COMPONENTS",
    "MINIMAL_COMPONENTS",
    # Contracts
    "CONTRACT_V1_0_0",
    "CONTRACT_V1_1_0",
    "CONTRACT_V1_2_0",
    "CONTRACT_V1_3_0",
    "REWARD_CONTRACTS",
    # Canonical
    "CANONICAL_REWARD_CONTRACT_ID",
    "CANONICAL_REWARD_CONTRACT",
    # Backward compatibility
    "DEFAULT_REWARD_CONFIG",
    "REWARD_CONFIG_HASH",
    # Functions
    "get_reward_contract",
    "get_production_reward_contract",
    "validate_reward_contract",
    "list_reward_contracts",
    "create_custom_reward_contract",
]
