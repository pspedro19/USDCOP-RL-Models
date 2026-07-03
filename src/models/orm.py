"""
SQLAlchemy ORM Models for USD/COP RL Trading System.

This module defines the database models using SQLAlchemy 2.0 ORM patterns
for trades, predictions, risk events, and feature snapshots.

Models:
    - Base: Declarative base for all models
    - Trade: Records of trading actions and their outcomes
    - Prediction: Model inference results with observation context
    - RiskEvent: Risk management events and limit breaches
    - FeatureSnapshot: Point-in-time feature values for reproducibility

Example usage:
    >>> from src.models.orm import Trade, Prediction, RiskEvent, FeatureSnapshot
    >>> from sqlalchemy import create_engine
    >>> from sqlalchemy.orm import Session
    >>>
    >>> engine = create_engine("postgresql://...")
    >>> with Session(engine) as session:
    ...     trade = Trade(
    ...         action=1,
    ...         action_name="LONG",
    ...         price=4250.50,
    ...         quantity=1.0,
    ...         model_version="v11.2",
    ...         confidence=0.85,
    ...     )
    ...     session.add(trade)
    ...     session.commit()
"""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    """
    Declarative base class for all ORM models.

    Uses SQLAlchemy 2.0 style with type annotations.
    """

    type_annotation_map = {
        dict[str, Any]: JSONB,
    }


class Trade(Base):
    """
    Trading action records with execution details and outcomes.

    Stores information about each trade executed by the system,
    including the model's decision context, execution prices,
    and realized P&L when the position is closed.

    Attributes:
        id: Primary key, auto-incremented
        timestamp: UTC timestamp when trade was executed
        action: Discretized action (0=HOLD, 1=LONG, 2=SHORT)
        action_name: Human-readable action name
        price: Execution price in COP
        quantity: Position size as fraction of portfolio
        model_version: Model version that generated the action
        confidence: Model's confidence in the action [0, 1]
        observation: Full observation vector as JSON
        probabilities: Action probabilities as JSON
        pnl: Realized P&L in COP (null until position closed)
        is_paper: Whether this is a paper trade
        created_at: Record creation timestamp
    """

    __tablename__ = "trades"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Timestamps
    timestamp: Mapped[datetime] = mapped_column(
        default=func.now(),
        nullable=False,
        index=True,
    )

    # Action details
    action: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Discretized action: 0=HOLD, 1=LONG, 2=SHORT",
    )
    action_name: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        comment="Human-readable action name",
    )

    # Execution details
    price: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Execution price in COP",
    )
    quantity: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Position size as fraction of portfolio",
    )

    # Model context
    model_version: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Model version that generated the action",
    )
    confidence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Model confidence score [0, 1]",
    )

    # Observation and probabilities as JSON
    observation: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Full observation vector used for inference",
    )
    probabilities: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Action probability distribution",
    )

    # Outcome
    pnl: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Realized P&L in COP when position is closed",
    )

    # Trade type
    is_paper: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
        comment="Whether this is a paper trade (not real money)",
    )

    # Audit
    created_at: Mapped[datetime] = mapped_column(
        default=func.now(),
        nullable=False,
    )

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "action IN (0, 1, 2)",
            name="ck_trades_action_valid",
        ),
        CheckConstraint(
            "action_name IN ('HOLD', 'LONG', 'SHORT')",
            name="ck_trades_action_name_valid",
        ),
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="ck_trades_confidence_range",
        ),
        CheckConstraint(
            "quantity >= 0 AND quantity <= 1",
            name="ck_trades_quantity_range",
        ),
        Index("ix_trades_timestamp_model", "timestamp", "model_version"),
        Index("ix_trades_paper_timestamp", "is_paper", "timestamp"),
        {"comment": "Trading action records with execution details and P&L outcomes"},
    )

    def __repr__(self) -> str:
        return (
            f"Trade(id={self.id}, timestamp={self.timestamp}, "
            f"action={self.action_name}, price={self.price}, pnl={self.pnl})"
        )


class Prediction(Base):
    """
    Model inference results with full observation context.

    Records every prediction made by the model for audit,
    debugging, and performance analysis. Links to trades
    via observation_hash for reproducibility verification.

    Attributes:
        id: Primary key, auto-incremented
        timestamp: UTC timestamp of prediction
        model_version: Model version that made the prediction
        observation_hash: SHA-256 hash of observation for deduplication
        action: Predicted action (0=HOLD, 1=LONG, 2=SHORT)
        probabilities: Full probability distribution as JSON
        confidence: Maximum probability (confidence score)
        latency_ms: Inference latency in milliseconds
        created_at: Record creation timestamp
    """

    __tablename__ = "predictions"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Timestamps
    timestamp: Mapped[datetime] = mapped_column(
        default=func.now(),
        nullable=False,
        index=True,
    )

    # Model identification
    model_version: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Model version identifier",
    )

    # Observation tracking
    observation_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        comment="SHA-256 hash of observation vector for reproducibility",
    )

    # Prediction output
    action: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Predicted action: 0=HOLD, 1=LONG, 2=SHORT",
    )
    probabilities: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Full action probability distribution",
    )
    confidence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Confidence score (max probability)",
    )

    # Performance metrics
    latency_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Inference latency in milliseconds",
    )

    # Audit
    created_at: Mapped[datetime] = mapped_column(
        default=func.now(),
        nullable=False,
    )

    # Constraints and indexes
    __table_args__ = (
        CheckConstraint(
            "action IN (0, 1, 2)",
            name="ck_predictions_action_valid",
        ),
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="ck_predictions_confidence_range",
        ),
        CheckConstraint(
            "latency_ms IS NULL OR latency_ms >= 0",
            name="ck_predictions_latency_positive",
        ),
        Index("ix_predictions_model_timestamp", "model_version", "timestamp"),
        Index("ix_predictions_hash_version", "observation_hash", "model_version"),
        {"comment": "Model inference results with observation context for audit"},
    )

    def __repr__(self) -> str:
        return (
            f"Prediction(id={self.id}, timestamp={self.timestamp}, "
            f"model={self.model_version}, action={self.action}, conf={self.confidence})"
        )


class RiskEvent(Base):
    """
    Risk management events and limit breaches.

    Records events when risk limits are approached or breached,
    including the action taken by the risk management system.
    Used for compliance, auditing, and risk analysis.

    Attributes:
        id: Primary key, auto-incremented
        timestamp: UTC timestamp of event
        event_type: Type of risk event (e.g., 'max_drawdown', 'position_limit')
        current_value: Current value that triggered the event
        limit_value: Configured limit value
        action_taken: Action taken by risk system
        details: Additional context as JSON
        created_at: Record creation timestamp
    """

    __tablename__ = "risk_events"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Timestamps
    timestamp: Mapped[datetime] = mapped_column(
        default=func.now(),
        nullable=False,
        index=True,
    )

    # Event classification
    event_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Type of risk event: max_drawdown, position_limit, daily_loss, etc.",
    )

    # Event values
    current_value: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Current value that triggered the event",
    )
    limit_value: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Configured limit value that was breached or approached",
    )

    # Response
    action_taken: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="Action taken by risk management system",
    )

    # Additional context
    details: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Additional context: position info, market conditions, etc.",
    )

    # Audit
    created_at: Mapped[datetime] = mapped_column(
        default=func.now(),
        nullable=False,
    )

    # Constraints and indexes
    __table_args__ = (
        CheckConstraint(
            "event_type IN ('max_drawdown', 'position_limit', 'daily_loss', "
            "'concentration', 'volatility', 'liquidity', 'margin', 'other')",
            name="ck_risk_events_type_valid",
        ),
        Index("ix_risk_events_type_timestamp", "event_type", "timestamp"),
        {"comment": "Risk management events and limit breaches for compliance"},
    )

    def __repr__(self) -> str:
        return (
            f"RiskEvent(id={self.id}, timestamp={self.timestamp}, "
            f"type={self.event_type}, current={self.current_value}, limit={self.limit_value})"
        )


class FeatureSnapshot(Base):
    """
    Point-in-time feature values for reproducibility.

    Stores complete feature vectors at specific timestamps
    to enable replay, debugging, and verification of model
    inputs. Linked to predictions via feature_hash.

    Attributes:
        id: Primary key, auto-incremented
        timestamp: UTC timestamp when features were computed
        features: Complete feature dictionary as JSON
        feature_hash: SHA-256 hash for deduplication and linking
        source: Source of features (e.g., 'live', 'backtest', 'replay')
        created_at: Record creation timestamp
    """

    __tablename__ = "feature_snapshots"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Timestamps
    timestamp: Mapped[datetime] = mapped_column(
        default=func.now(),
        nullable=False,
        index=True,
    )

    # Feature data
    features: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        comment="Complete feature dictionary",
    )

    # Feature tracking
    feature_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        comment="SHA-256 hash of features for deduplication",
    )

    # Source metadata
    source: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="live",
        index=True,
        comment="Source of features: live, backtest, replay, manual",
    )

    # Audit
    created_at: Mapped[datetime] = mapped_column(
        default=func.now(),
        nullable=False,
    )

    # Constraints and indexes
    __table_args__ = (
        CheckConstraint(
            "source IN ('live', 'backtest', 'replay', 'manual')",
            name="ck_feature_snapshots_source_valid",
        ),
        UniqueConstraint(
            "timestamp", "feature_hash",
            name="uq_feature_snapshots_timestamp_hash",
        ),
        Index("ix_feature_snapshots_hash_timestamp", "feature_hash", "timestamp"),
        Index("ix_feature_snapshots_source_timestamp", "source", "timestamp"),
        {"comment": "Point-in-time feature snapshots for reproducibility"},
    )

    def __repr__(self) -> str:
        return (
            f"FeatureSnapshot(id={self.id}, timestamp={self.timestamp}, "
            f"source={self.source}, hash={self.feature_hash[:12]}...)"
        )


# Export all models
__all__ = [
    "Base",
    "Trade",
    "Prediction",
    "RiskEvent",
    "FeatureSnapshot",
]
