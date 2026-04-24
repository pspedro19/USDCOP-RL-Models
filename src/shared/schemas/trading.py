"""
Trading Schema Definitions
==========================

Trade, signal, and market data schemas.
Shared between backend and frontend.

Contract: CTR-SHARED-TRADE-001
"""

from __future__ import annotations

from pydantic import Field, model_validator

from .core import (
    BaseSchema,
    OrderSide,
    SignalType,
    TimestampedSchema,
    TradeSide,
    TradeStatus,
)
from .features import FeatureSnapshotSchema

# =============================================================================
# MARKET DATA SCHEMAS
# =============================================================================


class CandlestickSchema(BaseSchema):
    """OHLCV candlestick data.

    Compatible with both TradingView and Recharts libraries.
    """

    time: int = Field(..., description="Unix timestamp (seconds)")
    open: float = Field(..., ge=0, description="Open price")
    high: float = Field(..., ge=0, description="High price")
    low: float = Field(..., ge=0, description="Low price")
    close: float = Field(..., ge=0, description="Close price")
    volume: float | None = Field(
        default=None, ge=0, description="Volume (often 0 for FX)"
    )

    @model_validator(mode="after")
    def validate_ohlc(self) -> CandlestickSchema:
        """Validate OHLC consistency."""
        if self.high < self.low:
            raise ValueError(f"high ({self.high}) must be >= low ({self.low})")
        if self.high < max(self.open, self.close):
            raise ValueError("high must be >= open and close")
        if self.low > min(self.open, self.close):
            raise ValueError("low must be <= open and close")
        return self


class MarketContextSchema(BaseSchema):
    """Market context at signal/trade time."""

    bid_ask_spread_bps: float = Field(
        default=0.0, ge=0, description="Bid-ask spread in basis points"
    )
    estimated_slippage_bps: float = Field(
        default=0.0, ge=0, description="Estimated slippage in basis points"
    )
    execution_price: float | None = Field(
        default=None, description="Actual execution price"
    )
    timestamp_utc: str = Field(..., description="UTC timestamp (ISO format)")


# =============================================================================
# SIGNAL SCHEMAS
# =============================================================================


class TechnicalIndicatorsSchema(BaseSchema):
    """Technical indicators at signal time.

    Extra fields allowed for extensibility.
    """
    model_config = BaseSchema.model_config.copy()
    model_config["extra"] = "allow"

    rsi: float | None = Field(default=None, ge=0, le=100)
    macd: float | None = Field(default=None)
    macd_signal: float | None = Field(default=None)
    ema_20: float | None = Field(default=None, ge=0)
    ema_50: float | None = Field(default=None, ge=0)
    atr: float | None = Field(default=None, ge=0)
    bollinger_upper: float | None = Field(default=None)
    bollinger_lower: float | None = Field(default=None)


class SignalSchema(BaseSchema):
    """Trading signal from model inference.

    Represents a trading recommendation at a point in time.
    """

    id: str = Field(..., description="Unique signal ID")
    timestamp: str = Field(..., description="Signal timestamp (ISO format)")
    type: SignalType = Field(..., description="Signal type (BUY/SELL/HOLD)")
    confidence: float = Field(
        ..., ge=0, le=1, description="Model confidence (0-1)"
    )
    price: float = Field(..., gt=0, description="Price at signal time")

    # Risk management
    stop_loss: float | None = Field(default=None, description="Stop loss price")
    take_profit: float | None = Field(default=None, description="Take profit price")

    # Explanation
    reasoning: list[str] = Field(
        default_factory=list, description="Signal reasoning/factors"
    )
    risk_score: float | None = Field(
        default=None, ge=0, le=1, description="Risk score (0-1)"
    )
    expected_return: float | None = Field(
        default=None, description="Expected return percentage"
    )
    time_horizon: str | None = Field(
        default=None, description="Expected time horizon"
    )

    # Model info
    model_source: str = Field(..., description="Model identifier")
    model_id: str | None = Field(default=None, description="Specific model ID")
    latency: float | None = Field(
        default=None, ge=0, description="Inference latency (ms)"
    )

    # Technical indicators
    technical_indicators: TechnicalIndicatorsSchema | None = Field(default=None)

    # Data type
    data_type: str | None = Field(
        default=None,
        description="Data type: backtest, out_of_sample, live"
    )


# =============================================================================
# TRADE SCHEMAS
# =============================================================================


class TradeMetadataSchema(BaseSchema):
    """Model metadata for a trade.

    Contains inference details for traceability.
    """

    confidence: float = Field(..., ge=0, le=1, description="Entry confidence")
    action_probs: list[float] | None = Field(
        default=None,
        min_length=3,
        max_length=3,
        description="Action probabilities [HOLD, LONG, SHORT]"
    )
    critic_value: float | None = Field(default=None, description="Critic value")
    entropy: float | None = Field(default=None, ge=0, description="Policy entropy")
    advantage: float | None = Field(default=None, description="Advantage estimate")
    model_version: str = Field(..., description="Model version used")
    norm_stats_version: str = Field(..., description="Normalization stats version")
    model_hash: str | None = Field(default=None, description="Model file hash")


class TradeSchema(TimestampedSchema):
    """Trade record.

    Represents a completed or open trade.
    """

    # Identity
    trade_id: str = Field(..., description="Unique trade ID")
    model_id: str = Field(..., description="Model that generated the trade")

    # Timestamps
    timestamp: str = Field(..., description="Entry timestamp (ISO format)")
    entry_time: str = Field(..., description="Entry time (ISO format)")
    exit_time: str | None = Field(default=None, description="Exit time")

    # Trade details
    side: TradeSide = Field(..., description="Trade direction")
    entry_price: float = Field(..., gt=0, description="Entry price")
    exit_price: float | None = Field(default=None, description="Exit price")
    size: float = Field(default=1.0, gt=0, description="Position size")

    # P&L
    pnl: float = Field(default=0.0, description="P&L in pips/points")
    pnl_usd: float = Field(default=0.0, description="P&L in USD")
    pnl_percent: float = Field(default=0.0, description="P&L percentage")
    pnl_pct: float = Field(default=0.0, description="P&L percentage (alias)")

    # Status
    status: TradeStatus = Field(default=TradeStatus.CLOSED)
    duration_minutes: int | None = Field(
        default=None, ge=0, description="Trade duration in minutes"
    )
    exit_reason: str | None = Field(default=None, description="Exit reason")

    # Equity tracking
    equity_at_entry: float | None = Field(default=None, description="Equity at entry")
    equity_at_exit: float | None = Field(default=None, description="Equity at exit")

    # Confidence
    entry_confidence: float | None = Field(
        default=None, ge=0, le=1, description="Entry confidence"
    )
    exit_confidence: float | None = Field(
        default=None, ge=0, le=1, description="Exit confidence"
    )

    # Additional metadata
    commission: float | None = Field(default=None, ge=0)
    market_regime: str | None = Field(default=None)
    max_adverse_excursion: float | None = Field(
        default=None, description="Maximum adverse excursion"
    )
    max_favorable_excursion: float | None = Field(
        default=None, description="Maximum favorable excursion"
    )

    # Feature snapshot (optional)
    features_snapshot: FeatureSnapshotSchema | None = Field(
        default=None, description="Features at trade entry"
    )
    model_metadata: TradeMetadataSchema | None = Field(
        default=None, description="Model inference metadata"
    )


class TradeSummarySchema(BaseSchema):
    """Summary statistics for a set of trades."""

    total_trades: int = Field(..., ge=0, description="Total number of trades")
    winning_trades: int = Field(..., ge=0, description="Number of winning trades")
    losing_trades: int = Field(..., ge=0, description="Number of losing trades")
    win_rate: float = Field(..., ge=0, le=1, description="Win rate (0-1)")

    # P&L
    total_pnl: float = Field(..., description="Total P&L")
    total_pnl_usd: float = Field(default=0.0, description="Total P&L in USD")
    total_return_pct: float = Field(..., description="Total return percentage")

    # Risk metrics
    max_drawdown_pct: float = Field(..., ge=0, description="Maximum drawdown %")
    sharpe_ratio: float | None = Field(default=None, description="Sharpe ratio")
    sortino_ratio: float | None = Field(default=None, description="Sortino ratio")
    profit_factor: float | None = Field(default=None, ge=0, description="Profit factor")

    # Trade statistics
    avg_win: float | None = Field(default=None, description="Average winning trade")
    avg_loss: float | None = Field(default=None, description="Average losing trade")
    largest_win: float | None = Field(default=None, description="Largest win")
    largest_loss: float | None = Field(default=None, description="Largest loss")
    avg_trade_duration_minutes: float | None = Field(
        default=None, ge=0, description="Average trade duration"
    )

    @model_validator(mode="after")
    def validate_counts(self) -> TradeSummarySchema:
        """Validate trade counts."""
        if self.winning_trades + self.losing_trades > self.total_trades:
            raise ValueError(
                "winning_trades + losing_trades cannot exceed total_trades"
            )
        return self


# =============================================================================
# EQUITY SCHEMAS
# =============================================================================


class EquityPointSchema(BaseSchema):
    """Single point on equity curve."""

    timestamp: str = Field(..., description="Timestamp (ISO format)")
    value: float = Field(..., description="Equity value")
    drawdown_pct: float | None = Field(default=None, ge=0, le=100)
    position: OrderSide | None = Field(default=None)
    price: float | None = Field(default=None, gt=0)


class EquityCurveSchema(BaseSchema):
    """Equity curve data."""

    model_id: str = Field(..., description="Model identifier")
    points: list[EquityPointSchema] = Field(..., min_length=1)

    # Summary
    start_value: float = Field(..., gt=0)
    end_value: float = Field(..., gt=0)
    total_return: float = Field(...)
    total_return_pct: float = Field(...)
    max_drawdown_pct: float = Field(..., ge=0)


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Market data
    "CandlestickSchema",
    "MarketContextSchema",
    # Signals
    "TechnicalIndicatorsSchema",
    "SignalSchema",
    # Trades
    "TradeMetadataSchema",
    "TradeSchema",
    "TradeSummarySchema",
    # Equity
    "EquityPointSchema",
    "EquityCurveSchema",
]
