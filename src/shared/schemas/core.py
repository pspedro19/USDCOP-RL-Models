"""
Core Schema Definitions
=======================

Base models and enums shared across all schemas.
These form the foundation for both backend and frontend types.

Contract: CTR-SHARED-CORE-001
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# ENUMS - Shared between Frontend and Backend
# =============================================================================


class SignalType(str, Enum):
    """Trading signal types.

    Frontend uses: BUY, SELL, HOLD
    Backend uses: LONG, SHORT, HOLD

    Mapping is handled by mapBackendAction/mapToBackendAction.
    """
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class BackendAction(str, Enum):
    """Backend action types (internal model output).

    Use SignalType for API communication.
    """
    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"


class TradeSide(str, Enum):
    """Trade direction."""
    LONG = "long"
    SHORT = "short"


class TradeStatus(str, Enum):
    """Trade lifecycle status."""
    OPEN = "open"
    CLOSED = "closed"
    PENDING = "pending"
    CANCELLED = "cancelled"


class OrderSide(str, Enum):
    """Position state."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class MarketStatus(str, Enum):
    """Market status."""
    OPEN = "open"
    CLOSED = "closed"
    PRE_MARKET = "pre_market"
    POST_MARKET = "post_market"


class DataSource(str, Enum):
    """Data source identifier."""
    LIVE = "live"
    POSTGRES = "postgres"
    MINIO = "minio"
    CACHED = "cached"
    MOCK = "mock"
    DEMO = "demo"
    FALLBACK = "fallback"
    NONE = "none"


class ModelStatus(str, Enum):
    """Model lifecycle status."""
    REGISTERED = "registered"
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATING = "validating"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    ACTIVE = "active"
    PRODUCTION = "production"
    BACKTEST = "backtest"
    RETIRED = "retired"
    FAILED = "failed"


# =============================================================================
# BASE MODELS
# =============================================================================


class BaseSchema(BaseModel):
    """Base schema with common configuration.

    All shared schemas should inherit from this.
    """
    model_config = ConfigDict(
        # Allow ORM objects to be converted
        from_attributes=True,
        # Validate on assignment
        validate_assignment=True,
        # Use enum values in JSON
        use_enum_values=True,
        # Extra fields forbidden by default
        extra="forbid",
        # JSON schema extras
        json_schema_extra={
            "x-contract-version": "1.0.0",
        }
    )


class TimestampedSchema(BaseSchema):
    """Base schema with timestamp fields."""

    created_at: Optional[datetime] = Field(
        default=None,
        description="Creation timestamp (UTC)",
        json_schema_extra={"format": "date-time"}
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        description="Last update timestamp (UTC)",
        json_schema_extra={"format": "date-time"}
    )


# =============================================================================
# MAPPING FUNCTIONS
# =============================================================================


def map_backend_to_signal(action: BackendAction) -> SignalType:
    """Map backend action to frontend signal type.

    LONG -> BUY
    SHORT -> SELL
    HOLD -> HOLD
    """
    mapping = {
        BackendAction.LONG: SignalType.BUY,
        BackendAction.SHORT: SignalType.SELL,
        BackendAction.HOLD: SignalType.HOLD,
    }
    return mapping.get(action, SignalType.HOLD)


def map_signal_to_backend(signal: SignalType) -> BackendAction:
    """Map frontend signal to backend action.

    BUY -> LONG
    SELL -> SHORT
    HOLD -> HOLD
    """
    mapping = {
        SignalType.BUY: BackendAction.LONG,
        SignalType.SELL: BackendAction.SHORT,
        SignalType.HOLD: BackendAction.HOLD,
    }
    return mapping.get(signal, BackendAction.HOLD)


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "SignalType",
    "BackendAction",
    "TradeSide",
    "TradeStatus",
    "OrderSide",
    "MarketStatus",
    "DataSource",
    "ModelStatus",
    # Base models
    "BaseSchema",
    "TimestampedSchema",
    # Mapping functions
    "map_backend_to_signal",
    "map_signal_to_backend",
]
