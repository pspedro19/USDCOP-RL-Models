"""
Feature Schema Definitions
==========================

Feature contracts for model observation space.
Single Source of Truth for feature ordering and normalization.

Contract: CTR-SHARED-FEAT-001
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

from pydantic import Field, field_validator, model_validator

from .core import BaseSchema


# =============================================================================
# CONSTANTS - SSOT for Feature Contract
# =============================================================================


OBSERVATION_DIM: int = 15
"""Model observation dimension (15 features)."""

FEATURE_ORDER: Tuple[str, ...] = (
    # Returns (3)
    "log_ret_5m",
    "log_ret_1h",
    "log_ret_4h",
    # Technical Indicators (3)
    "rsi_9",
    "atr_pct",
    "adx_14",
    # Macro Z-Scores (4)
    "dxy_z",
    "dxy_change_1d",
    "vix_z",
    "embi_z",
    # Macro Changes (3)
    "brent_change_1d",
    "rate_spread",
    "usdmxn_change_1d",
    # State Features (2)
    "position",
    "time_normalized",
)
"""Feature order - canonical ordering for observation vector."""


# Type alias for feature order
FeatureOrderType = Literal[
    "log_ret_5m", "log_ret_1h", "log_ret_4h",
    "rsi_9", "atr_pct", "adx_14",
    "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
    "brent_change_1d", "rate_spread", "usdmxn_change_1d",
    "position", "time_normalized"
]


# =============================================================================
# SCHEMAS
# =============================================================================


class NormalizationStatsSchema(BaseSchema):
    """Normalization statistics for a single feature.

    Used for z-score normalization: (x - mean) / std
    """

    mean: float = Field(..., description="Feature mean from training period")
    std: float = Field(..., gt=0, description="Feature standard deviation")
    clip_min: float = Field(default=-5.0, description="Minimum clip value")
    clip_max: float = Field(default=5.0, description="Maximum clip value")

    @property
    def clip_range(self) -> Tuple[float, float]:
        """Get clip range as tuple."""
        return (self.clip_min, self.clip_max)


class NamedFeatures(BaseSchema):
    """Named features for observation.

    Provides type-safe access to individual features.
    """

    # Returns
    log_ret_5m: float = Field(..., description="5-minute log return")
    log_ret_1h: float = Field(..., description="1-hour log return")
    log_ret_4h: float = Field(..., description="4-hour log return")

    # Technical Indicators
    rsi_9: float = Field(..., ge=0, le=100, description="RSI period 9")
    atr_pct: float = Field(..., ge=0, description="ATR as percentage")
    adx_14: float = Field(..., ge=0, le=100, description="ADX period 14")

    # Macro Z-Scores
    dxy_z: float = Field(..., description="DXY z-score")
    dxy_change_1d: float = Field(..., description="DXY daily change")
    vix_z: float = Field(..., description="VIX z-score")
    embi_z: float = Field(..., description="EMBI z-score")

    # Macro Changes
    brent_change_1d: float = Field(..., description="Brent daily change")
    rate_spread: float = Field(..., description="UST 10Y-2Y spread")
    usdmxn_change_1d: float = Field(..., description="USDMXN daily change")

    # State Features
    position: float = Field(..., ge=-1, le=1, description="Current position")
    time_normalized: float = Field(
        ..., ge=0, le=1, description="Normalized session time"
    )

    def to_observation(self) -> List[float]:
        """Convert to observation vector in canonical order."""
        return [
            self.log_ret_5m, self.log_ret_1h, self.log_ret_4h,
            self.rsi_9, self.atr_pct, self.adx_14,
            self.dxy_z, self.dxy_change_1d, self.vix_z, self.embi_z,
            self.brent_change_1d, self.rate_spread, self.usdmxn_change_1d,
            self.position, self.time_normalized,
        ]

    @classmethod
    def from_observation(cls, observation: List[float]) -> "NamedFeatures":
        """Create from observation vector."""
        if len(observation) != OBSERVATION_DIM:
            raise ValueError(
                f"Expected {OBSERVATION_DIM} features, got {len(observation)}"
            )
        return cls(
            log_ret_5m=observation[0],
            log_ret_1h=observation[1],
            log_ret_4h=observation[2],
            rsi_9=observation[3],
            atr_pct=observation[4],
            adx_14=observation[5],
            dxy_z=observation[6],
            dxy_change_1d=observation[7],
            vix_z=observation[8],
            embi_z=observation[9],
            brent_change_1d=observation[10],
            rate_spread=observation[11],
            usdmxn_change_1d=observation[12],
            position=observation[13],
            time_normalized=observation[14],
        )


class ObservationSchema(BaseSchema):
    """Observation vector for model inference.

    This is the raw 15-dimensional vector passed to the model.
    """

    values: List[float] = Field(
        ...,
        min_length=OBSERVATION_DIM,
        max_length=OBSERVATION_DIM,
        description=f"Observation vector ({OBSERVATION_DIM} dimensions)"
    )
    contract_version: str = Field(default="v1", description="Contract version")

    @field_validator("values")
    @classmethod
    def validate_length(cls, v: List[float]) -> List[float]:
        """Validate observation vector length."""
        if len(v) != OBSERVATION_DIM:
            raise ValueError(
                f"Observation must have {OBSERVATION_DIM} values, got {len(v)}"
            )
        return v

    def to_named_features(self) -> NamedFeatures:
        """Convert to named features."""
        return NamedFeatures.from_observation(self.values)


class MarketContextSchema(BaseSchema):
    """Market context at time of observation.

    Additional market information not in the observation vector.
    """

    bid_ask_spread_bps: float = Field(
        ..., ge=0, description="Bid-ask spread in basis points"
    )
    estimated_slippage_bps: float = Field(
        ..., ge=0, description="Estimated slippage in basis points"
    )
    execution_price: Optional[float] = Field(
        default=None, description="Execution price (if filled)"
    )
    timestamp_utc: str = Field(..., description="Timestamp in UTC (ISO format)")


class FeatureSnapshotSchema(BaseSchema):
    """Complete feature snapshot at a point in time.

    Combines observation vector with named features and market context.
    """

    observation: List[float] = Field(
        ...,
        min_length=OBSERVATION_DIM,
        max_length=OBSERVATION_DIM,
        description=f"Raw observation vector ({OBSERVATION_DIM} dims)"
    )
    features: NamedFeatures = Field(..., description="Named features")
    market_context: Optional[MarketContextSchema] = Field(
        default=None, description="Market context"
    )
    contract_version: str = Field(default="v1", description="Contract version")

    @model_validator(mode="after")
    def validate_consistency(self) -> "FeatureSnapshotSchema":
        """Ensure observation and features are consistent."""
        expected = self.features.to_observation()
        for i, (obs, exp) in enumerate(zip(self.observation, expected)):
            if abs(obs - exp) > 1e-6:
                raise ValueError(
                    f"Observation[{i}]={obs} doesn't match features ({exp})"
                )
        return self


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_observation(
    log_ret_5m: float = 0.0,
    log_ret_1h: float = 0.0,
    log_ret_4h: float = 0.0,
    rsi_9: float = 50.0,
    atr_pct: float = 0.5,
    adx_14: float = 25.0,
    dxy_z: float = 0.0,
    dxy_change_1d: float = 0.0,
    vix_z: float = 0.0,
    embi_z: float = 0.0,
    brent_change_1d: float = 0.0,
    rate_spread: float = 0.0,
    usdmxn_change_1d: float = 0.0,
    position: float = 0.0,
    time_normalized: float = 0.5,
) -> ObservationSchema:
    """Create observation with named parameters."""
    return ObservationSchema(
        values=[
            log_ret_5m, log_ret_1h, log_ret_4h,
            rsi_9, atr_pct, adx_14,
            dxy_z, dxy_change_1d, vix_z, embi_z,
            brent_change_1d, rate_spread, usdmxn_change_1d,
            position, time_normalized,
        ]
    )


def validate_observation(observation: List[float]) -> bool:
    """Validate an observation vector.

    Returns True if valid, False otherwise.
    """
    if len(observation) != OBSERVATION_DIM:
        return False

    # Check for NaN/Inf
    for val in observation:
        if not isinstance(val, (int, float)):
            return False
        if val != val or val == float('inf') or val == float('-inf'):
            return False

    return True


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Constants (new names)
    "OBSERVATION_DIM",
    "FEATURE_ORDER",
    # Type aliases (new names)
    "FeatureOrderType",
    # Schemas
    "NormalizationStatsSchema",
    "NamedFeatures",
    "ObservationSchema",
    "MarketContextSchema",
    "FeatureSnapshotSchema",
    # Factory functions
    "create_observation",
    "validate_observation",
]
