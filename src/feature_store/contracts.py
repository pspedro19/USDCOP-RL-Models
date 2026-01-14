"""
Feature Store Contracts
========================
Pydantic contracts for type-safe feature validation across all pipelines.

Ensures feature parity between:
- Training (L3)
- Backtest (L4)
- Inference (L5)

SOLID Principles:
- Single Responsibility: Each contract represents one concept
- Open/Closed: Extend via inheritance, not modification
- Liskov Substitution: All features implement FeatureValue protocol
- Interface Segregation: Minimal required fields per contract
- Dependency Inversion: Depend on abstractions (FeatureSpec)

Author: Trading Team
Version: 1.0.0
Created: 2025-01-12
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field, field_validator, model_validator
import numpy as np


# =============================================================================
# ENUMS
# =============================================================================

class FeatureCategory(str, Enum):
    """Feature categories for organization"""
    PRICE = "price"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    TREND = "trend"
    VOLUME = "volume"
    MACRO = "macro"
    REGIME = "regime"
    DERIVED = "derived"


class SmoothingMethod(str, Enum):
    """Smoothing methods for indicators - MUST be consistent across pipelines"""
    EMA = "ema"           # Exponential Moving Average (Wilder's)
    SMA = "sma"           # Simple Moving Average
    WILDER = "wilder"     # Wilder's smoothing (alpha = 1/n)
    NONE = "none"         # No smoothing


class NormalizationMethod(str, Enum):
    """Normalization methods for feature scaling"""
    ZSCORE = "zscore"           # (x - mean) / std
    MINMAX = "minmax"           # (x - min) / (max - min)
    ROBUST = "robust"           # (x - median) / IQR
    ROLLING_ZSCORE = "rolling"  # Rolling window z-score
    NONE = "none"               # No normalization


class FeatureVersion(str, Enum):
    """Feature set versions"""
    V1 = "v1"     # Legacy 32-dimensional
    CURRENT = "current"   # Current 15-dimensional


# =============================================================================
# BASE CONTRACTS
# =============================================================================

class BaseContract(BaseModel):
    """Base contract with frozen immutability"""
    model_config = {"frozen": True, "extra": "forbid"}


class TimestampedContract(BaseContract):
    """Contract with automatic timestamp"""
    created_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# FEATURE SPECIFICATION CONTRACTS
# =============================================================================

class CalculationParams(BaseContract):
    """Parameters for feature calculation"""
    window: int = Field(ge=1, description="Lookback window")
    smoothing: SmoothingMethod = Field(default=SmoothingMethod.EMA)
    alpha: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_alpha(self) -> "CalculationParams":
        """Set default alpha based on smoothing method"""
        if self.alpha is None and self.smoothing == SmoothingMethod.WILDER:
            # Create a new instance with computed alpha
            object.__setattr__(self, "alpha", 1.0 / self.window)
        return self


class NormalizationParams(BaseContract):
    """Parameters for feature normalization"""
    method: NormalizationMethod = Field(default=NormalizationMethod.ZSCORE)
    mean: Optional[float] = Field(default=None)
    std: Optional[float] = Field(default=None)
    min_val: Optional[float] = Field(default=None)
    max_val: Optional[float] = Field(default=None)
    rolling_window: Optional[int] = Field(default=None, ge=1)
    clip_range: Tuple[float, float] = Field(default=(-3.0, 3.0))

    @model_validator(mode="after")
    def validate_params(self) -> "NormalizationParams":
        """Validate normalization params match method"""
        if self.method == NormalizationMethod.ZSCORE:
            if self.mean is None or self.std is None:
                raise ValueError("ZSCORE requires mean and std")
        elif self.method == NormalizationMethod.MINMAX:
            if self.min_val is None or self.max_val is None:
                raise ValueError("MINMAX requires min_val and max_val")
        elif self.method == NormalizationMethod.ROLLING_ZSCORE:
            if self.rolling_window is None:
                raise ValueError("ROLLING_ZSCORE requires rolling_window")
        return self


class FeatureSpec(BaseContract):
    """
    Complete specification for a single feature.

    This is the SINGLE SOURCE OF TRUTH for how a feature should be calculated
    and normalized across all pipelines.
    """
    name: str = Field(description="Feature name (e.g., 'rsi_9')")
    category: FeatureCategory
    description: str = Field(default="")

    # Calculation parameters
    calculation: CalculationParams

    # Normalization parameters
    normalization: NormalizationParams

    # Dependencies
    requires: List[str] = Field(default_factory=list, description="Required input columns")

    # Validation
    valid_range: Optional[Tuple[float, float]] = Field(default=None)
    allow_nan: bool = Field(default=False)

    @property
    def full_name(self) -> str:
        """Get full feature name with version"""
        return self.name


class FeatureSetSpec(BaseContract):
    """
    Specification for a complete feature set (version).

    Defines all features in a specific version (v1, current, etc.)
    """
    version: FeatureVersion
    features: List[FeatureSpec]
    dimension: int = Field(ge=1)
    description: str = Field(default="")

    @model_validator(mode="after")
    def validate_dimension(self) -> "FeatureSetSpec":
        """Ensure dimension matches feature count"""
        if len(self.features) != self.dimension:
            raise ValueError(
                f"Feature count ({len(self.features)}) != dimension ({self.dimension})"
            )
        return self

    def get_feature(self, name: str) -> Optional[FeatureSpec]:
        """Get feature by name"""
        for f in self.features:
            if f.name == name:
                return f
        return None

    def get_feature_names(self) -> List[str]:
        """Get ordered list of feature names"""
        return [f.name for f in self.features]


# =============================================================================
# FEATURE VALUE CONTRACTS
# =============================================================================

class FeatureValue(BaseContract):
    """Single feature value with metadata"""
    name: str
    value: float
    timestamp: datetime
    is_normalized: bool = Field(default=False)

    @field_validator("value")
    @classmethod
    def validate_value(cls, v: float) -> float:
        """Validate value is finite"""
        if not np.isfinite(v):
            raise ValueError(f"Feature value must be finite, got {v}")
        return v


class FeatureVector(BaseContract):
    """Complete feature vector for a single observation"""
    version: FeatureVersion
    timestamp: datetime
    values: Dict[str, float]
    is_normalized: bool = Field(default=False)

    @model_validator(mode="after")
    def validate_values(self) -> "FeatureVector":
        """Validate all values are finite"""
        for name, value in self.values.items():
            if not np.isfinite(value):
                raise ValueError(f"Feature {name} has non-finite value: {value}")
        return self

    def to_array(self, feature_order: List[str]) -> np.ndarray:
        """Convert to numpy array in specified order"""
        return np.array([self.values[name] for name in feature_order])

    @property
    def dimension(self) -> int:
        return len(self.values)


class FeatureBatch(BaseContract):
    """Batch of feature vectors for training/backtest"""
    version: FeatureVersion
    vectors: List[FeatureVector]
    is_normalized: bool = Field(default=False)

    def to_dataframe(self):
        """Convert to pandas DataFrame"""
        import pandas as pd

        data = []
        for vec in self.vectors:
            row = {"timestamp": vec.timestamp, **vec.values}
            data.append(row)

        return pd.DataFrame(data).set_index("timestamp")

    @classmethod
    def from_dataframe(
        cls,
        df,
        version: FeatureVersion,
        is_normalized: bool = False
    ) -> "FeatureBatch":
        """Create from pandas DataFrame with timestamp index"""
        vectors = []
        for timestamp, row in df.iterrows():
            vectors.append(FeatureVector(
                version=version,
                timestamp=timestamp,
                values=row.to_dict(),
                is_normalized=is_normalized,
            ))

        return cls(
            version=version,
            vectors=vectors,
            is_normalized=is_normalized,
        )


# =============================================================================
# NORMALIZATION STATS CONTRACT
# =============================================================================

class NormalizationStats(BaseContract):
    """
    Normalization statistics for a feature set.

    This contract ensures consistency between training stats
    and inference normalization.
    """
    version: FeatureVersion
    stats: Dict[str, NormalizationParams]
    computed_at: datetime = Field(default_factory=datetime.utcnow)
    sample_size: int = Field(ge=1)

    def get_params(self, feature_name: str) -> NormalizationParams:
        """Get normalization params for a feature"""
        if feature_name not in self.stats:
            raise KeyError(f"No normalization stats for feature: {feature_name}")
        return self.stats[feature_name]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict"""
        return {
            "version": self.version.value,
            "computed_at": self.computed_at.isoformat(),
            "sample_size": self.sample_size,
            "stats": {
                name: {
                    "method": params.method.value,
                    "mean": params.mean,
                    "std": params.std,
                    "min_val": params.min_val,
                    "max_val": params.max_val,
                    "clip_range": list(params.clip_range),
                }
                for name, params in self.stats.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NormalizationStats":
        """Create from serialized dict"""
        stats = {}
        for name, params in data["stats"].items():
            stats[name] = NormalizationParams(
                method=NormalizationMethod(params["method"]),
                mean=params.get("mean"),
                std=params.get("std"),
                min_val=params.get("min_val"),
                max_val=params.get("max_val"),
                clip_range=tuple(params.get("clip_range", [-3.0, 3.0])),
            )

        return cls(
            version=FeatureVersion(data["version"]),
            stats=stats,
            computed_at=datetime.fromisoformat(data["computed_at"]),
            sample_size=data["sample_size"],
        )


# =============================================================================
# FEATURE REGISTRY CONTRACT
# =============================================================================

class FeatureRegistryEntry(TimestampedContract):
    """Entry in feature registry"""
    feature_set: FeatureSetSpec
    normalization_stats: Optional[NormalizationStats] = None
    is_active: bool = Field(default=True)
    trained_models: List[str] = Field(default_factory=list)


# =============================================================================
# PIPELINE CONTRACTS
# =============================================================================

class RawDataInput(BaseContract):
    """Input contract for raw OHLCV data"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None

    @model_validator(mode="after")
    def validate_ohlc(self) -> "RawDataInput":
        """Validate OHLC relationship"""
        if self.high < self.low:
            raise ValueError(f"High ({self.high}) < Low ({self.low})")
        if self.close < self.low or self.close > self.high:
            raise ValueError(f"Close ({self.close}) outside H/L range")
        if self.open < self.low or self.open > self.high:
            raise ValueError(f"Open ({self.open}) outside H/L range")
        return self


class MacroDataInput(BaseContract):
    """Input contract for macro indicators"""
    timestamp: datetime
    dxy: Optional[float] = None      # Dollar index
    vix: Optional[float] = None      # Volatility index
    wti: Optional[float] = None      # Oil price
    embi: Optional[float] = None     # Emerging market bond index


class CalculationRequest(BaseContract):
    """Request to calculate features"""
    version: FeatureVersion
    raw_data: List[RawDataInput]
    macro_data: Optional[List[MacroDataInput]] = None
    normalize: bool = Field(default=True)
    normalization_stats: Optional[NormalizationStats] = None


class CalculationResult(TimestampedContract):
    """Result of feature calculation"""
    version: FeatureVersion
    features: FeatureBatch
    warnings: List[str] = Field(default_factory=list)
    calculation_time_ms: float = Field(ge=0)
