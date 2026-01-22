"""
FEATURE CONTRACT - Single Source of Truth
==========================================
Este módulo exporta el contrato de features CANÓNICO (producción).

Para experimentos con diferentes features, usar:
    from src.core.contracts.feature_contracts_registry import get_contract

Contract ID: CTR-FEATURE-001
Version: 2.1.0
Updated: 2026-01-18 (Added contract registry support)
"""
from typing import Tuple, Dict, Final, List, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import hashlib

# =============================================================================
# IMPORT FROM REGISTRY (SSOT for multiple contracts)
# =============================================================================
# The feature_contracts_registry.py defines all available contracts.
# This file re-exports the CANONICAL (production) contract for backward compat.

try:
    from .feature_contracts_registry import (
        CANONICAL_CONTRACT,
        CANONICAL_CONTRACT_ID,
        FEATURE_ORDER as REGISTRY_FEATURE_ORDER,
        OBSERVATION_DIM as REGISTRY_OBSERVATION_DIM,
        FEATURE_ORDER_HASH as REGISTRY_FEATURE_ORDER_HASH,
        get_contract,
        get_production_contract,
        FeatureContract,
        CONTRACTS,
    )
    _REGISTRY_AVAILABLE = True
except ImportError:
    _REGISTRY_AVAILABLE = False


# =============================================================================
# CANONICAL FEATURE ORDER (PRODUCTION)
# =============================================================================
# This is the PRODUCTION contract (v1.0.0) with 15 features.
# For experiments with different features, use get_contract(contract_id).

if _REGISTRY_AVAILABLE:
    FEATURE_ORDER: Final[Tuple[str, ...]] = REGISTRY_FEATURE_ORDER
    OBSERVATION_DIM: Final[int] = REGISTRY_OBSERVATION_DIM
    FEATURE_ORDER_HASH: Final[str] = REGISTRY_FEATURE_ORDER_HASH
else:
    # Fallback if registry not available
    FEATURE_ORDER: Final[Tuple[str, ...]] = (
        "log_ret_5m",       # 0: 5-minute log return
        "log_ret_1h",       # 1: 1-hour log return
        "log_ret_4h",       # 2: 4-hour log return
        "rsi_9",            # 3: RSI with 9 periods
        "atr_pct",          # 4: ATR as percentage of price
        "adx_14",           # 5: ADX with 14 periods
        "dxy_z",            # 6: DXY index z-score
        "dxy_change_1d",    # 7: DXY 1-day change
        "vix_z",            # 8: VIX z-score
        "embi_z",           # 9: EMBI Colombia z-score
        "brent_change_1d",  # 10: Brent oil 1-day change
        "rate_spread",      # 11: Interest rate spread
        "usdmxn_change_1d", # 12: USD/MXN 1-day change
        "position",         # 13: Current position (-1, 0, 1)
        "time_normalized",  # 14: Normalized time of day [0, 1]
    )
    OBSERVATION_DIM: Final[int] = 15
    FEATURE_ORDER_HASH: Final[str] = hashlib.sha256(
        ",".join(FEATURE_ORDER).encode("utf-8")
    ).hexdigest()[:16]

FEATURE_CONTRACT_VERSION: Final[str] = "2.1.0"


class FeatureType(Enum):
    TECHNICAL = "technical"
    MACRO = "macro"
    STATE = "state"
    TIME = "time"


class FeatureUnit(Enum):
    ZSCORE = "z-score"
    PERCENTAGE = "percentage"
    NORMALIZED = "normalized"
    RAW = "raw"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    index: int
    type: FeatureType
    unit: FeatureUnit
    description: str
    clip_min: float = -5.0
    clip_max: float = 5.0
    requires_normalization: bool = True
    source: str = ""


FEATURE_SPECS: Final[Dict[str, FeatureSpec]] = {
    "log_ret_5m": FeatureSpec("log_ret_5m", 0, FeatureType.TECHNICAL, FeatureUnit.ZSCORE, "5-minute log return", source="L1_features"),
    "log_ret_1h": FeatureSpec("log_ret_1h", 1, FeatureType.TECHNICAL, FeatureUnit.ZSCORE, "1-hour log return", source="L1_features"),
    "log_ret_4h": FeatureSpec("log_ret_4h", 2, FeatureType.TECHNICAL, FeatureUnit.ZSCORE, "4-hour log return", source="L1_features"),
    "rsi_9": FeatureSpec("rsi_9", 3, FeatureType.TECHNICAL, FeatureUnit.ZSCORE, "RSI(9)", clip_min=-3.0, clip_max=3.0, source="L1_features"),
    "atr_pct": FeatureSpec("atr_pct", 4, FeatureType.TECHNICAL, FeatureUnit.ZSCORE, "ATR as % of price", source="L1_features"),
    "adx_14": FeatureSpec("adx_14", 5, FeatureType.TECHNICAL, FeatureUnit.ZSCORE, "ADX(14)", source="L1_features"),
    "dxy_z": FeatureSpec("dxy_z", 6, FeatureType.MACRO, FeatureUnit.ZSCORE, "DXY index z-score", source="L0_macro"),
    "dxy_change_1d": FeatureSpec("dxy_change_1d", 7, FeatureType.MACRO, FeatureUnit.ZSCORE, "DXY 1-day change", source="L0_macro"),
    "vix_z": FeatureSpec("vix_z", 8, FeatureType.MACRO, FeatureUnit.ZSCORE, "VIX z-score", source="L0_macro"),
    "embi_z": FeatureSpec("embi_z", 9, FeatureType.MACRO, FeatureUnit.ZSCORE, "EMBI Colombia z-score", source="L0_macro"),
    "brent_change_1d": FeatureSpec("brent_change_1d", 10, FeatureType.MACRO, FeatureUnit.ZSCORE, "Brent oil 1-day change", source="L0_macro"),
    "rate_spread": FeatureSpec("rate_spread", 11, FeatureType.MACRO, FeatureUnit.ZSCORE, "Interest rate spread", source="L0_macro"),
    "usdmxn_change_1d": FeatureSpec("usdmxn_change_1d", 12, FeatureType.MACRO, FeatureUnit.ZSCORE, "USD/MXN 1-day change", source="L0_macro"),
    "position": FeatureSpec("position", 13, FeatureType.STATE, FeatureUnit.RAW, "Current position", clip_min=-1.0, clip_max=1.0, requires_normalization=False, source="trading_state"),
    "time_normalized": FeatureSpec("time_normalized", 14, FeatureType.TIME, FeatureUnit.NORMALIZED, "Time of day normalized", clip_min=0.0, clip_max=1.0, requires_normalization=False, source="trading_state"),
}


@dataclass(frozen=True)
class FeatureContract:
    version: str = FEATURE_CONTRACT_VERSION
    observation_dim: int = OBSERVATION_DIM
    feature_order: Tuple[str, ...] = FEATURE_ORDER
    feature_order_hash: str = FEATURE_ORDER_HASH
    clip_range: Tuple[float, float] = (-5.0, 5.0)
    dtype: str = "float32"
    normalization_method: str = "zscore"

    def get_feature_index(self, name: str) -> int:
        if name not in self.feature_order:
            raise FeatureContractError(f"Unknown feature: '{name}'")
        return self.feature_order.index(name)

    def get_feature_spec(self, name: str) -> FeatureSpec:
        if name not in FEATURE_SPECS:
            raise FeatureContractError(f"No spec for feature: {name}")
        return FEATURE_SPECS[name]

    def get_features_by_type(self, ftype: FeatureType) -> List[str]:
        return [name for name, spec in FEATURE_SPECS.items() if spec.type == ftype]

    def get_normalizable_features(self) -> List[str]:
        return [name for name, spec in FEATURE_SPECS.items() if spec.requires_normalization]

    def validate_observation(self, obs: np.ndarray, strict: bool = True) -> Tuple[bool, List[str]]:
        errors = []

        if obs.shape != (self.observation_dim,):
            errors.append(f"Invalid shape: expected ({self.observation_dim},), got {obs.shape}")
            return False, errors

        if obs.dtype != np.float32:
            errors.append(f"Invalid dtype: expected float32, got {obs.dtype}")

        if np.any(np.isnan(obs)):
            nan_idx = np.where(np.isnan(obs))[0].tolist()
            errors.append(f"NaN values at indices {nan_idx}")

        if np.any(np.isinf(obs)):
            inf_idx = np.where(np.isinf(obs))[0].tolist()
            errors.append(f"Inf values at indices {inf_idx}")

        if strict and not np.any(np.isnan(obs)) and not np.any(np.isinf(obs)):
            for i, (name, value) in enumerate(zip(self.feature_order, obs)):
                spec = FEATURE_SPECS[name]
                if value < spec.clip_min or value > spec.clip_max:
                    errors.append(f"Feature '{name}': value {value:.4f} outside range [{spec.clip_min}, {spec.clip_max}]")

        return len(errors) == 0, errors

    def validate_feature_dict(self, features: Dict[str, float]) -> Tuple[bool, List[str]]:
        errors = []
        missing = set(self.feature_order) - set(features.keys())
        if missing:
            errors.append(f"Missing features: {sorted(missing)}")
        extra = set(features.keys()) - set(self.feature_order)
        if extra:
            errors.append(f"Extra features: {sorted(extra)}")
        return len(errors) == 0, errors

    def dict_to_array(self, features: Dict[str, float]) -> np.ndarray:
        is_valid, errors = self.validate_feature_dict(features)
        if not is_valid:
            raise FeatureContractError(f"Cannot convert: {errors}")
        return np.array([features[name] for name in self.feature_order], dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "observation_dim": self.observation_dim,
            "feature_order": list(self.feature_order),
            "feature_order_hash": self.feature_order_hash,
            "clip_range": list(self.clip_range),
            "dtype": self.dtype,
            "normalization_method": self.normalization_method,
        }


class FeatureContractError(ValueError):
    pass


FEATURE_CONTRACT: Final[FeatureContract] = FeatureContract()


def validate_feature_vector(obs: np.ndarray, raise_on_error: bool = True, strict: bool = True) -> bool:
    is_valid, errors = FEATURE_CONTRACT.validate_observation(obs, strict=strict)
    if not is_valid and raise_on_error:
        raise FeatureContractError(f"Feature validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    return is_valid


def get_feature_index(name: str) -> int:
    return FEATURE_CONTRACT.get_feature_index(name)


def get_feature_names() -> List[str]:
    return list(FEATURE_ORDER)


def features_dict_to_array(features: Dict[str, float]) -> np.ndarray:
    return FEATURE_CONTRACT.dict_to_array(features)
