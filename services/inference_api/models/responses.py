"""
Response schemas for the inference/backtest API.

Reconstructed from construction sites in `orchestrator/backtest_orchestrator.py`,
`routers/health.py`, and `routers/trades.py`. Fields carry permissive defaults and
`extra="allow"` because several are built from DB-row / summary dicts via `**kwargs`
(e.g. `TradeResponse(**row_dict)`, `BacktestSummary(**summary_dict)`) whose exact
key set can drift with schema changes — construction must never raise on an extra key.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TradeResponse(BaseModel):
    """A single simulated/persisted trade. Built from Trade objects and DB rows."""

    model_config = ConfigDict(extra="allow")

    trade_id: int | str | None = Field(default=None, description="Trade identifier.")
    model_id: str = Field(default="ppo_primary", description="Model that produced it.")
    timestamp: str = Field(default="", description="Entry timestamp (ISO8601).")
    entry_time: str = Field(default="", description="Entry time (ISO8601).")
    exit_time: str | None = Field(default=None, description="Exit time (ISO8601).")
    side: str = Field(default="long", description="Trade side (long/short).")
    entry_price: float = Field(default=0.0, description="Entry price.")
    exit_price: float | None = Field(default=None, description="Exit price.")
    # Both naming conventions kept — callers read pnl/pnl_usd and pnl_percent/pnl_pct.
    pnl: float = Field(default=0.0, description="P&L in USD (legacy alias).")
    pnl_usd: float = Field(default=0.0, description="P&L in USD.")
    pnl_percent: float = Field(default=0.0, description="P&L percent (legacy alias).")
    pnl_pct: float = Field(default=0.0, description="P&L percent.")
    status: str = Field(default="closed", description="closed/open.")
    duration_minutes: float | int | None = Field(default=None, description="Duration.")
    exit_reason: str | None = Field(default=None, description="Exit reason.")
    equity_at_entry: float | None = Field(default=None, description="Equity before.")
    equity_at_exit: float | None = Field(default=None, description="Equity after.")
    entry_confidence: float | None = Field(default=None, description="Entry conf.")
    exit_confidence: float | None = Field(default=None, description="Exit conf.")


class BacktestSummary(BaseModel):
    """Aggregate backtest metrics. Built via `BacktestSummary(**summary_dict)`."""

    model_config = ConfigDict(extra="allow")

    total_trades: int = Field(default=0)
    winning_trades: int = Field(default=0)
    losing_trades: int = Field(default=0)
    win_rate: float = Field(default=0.0, description="Win rate percent.")
    total_pnl: float = Field(default=0.0, description="Total P&L in USD.")
    total_return_pct: float = Field(default=0.0, description="Total return percent.")
    max_drawdown_pct: float = Field(default=0.0, description="Max drawdown percent.")
    avg_trade_duration_minutes: float | None = Field(default=None)


class ProgressUpdate(BaseModel):
    """SSE progress event emitted during a streaming backtest."""

    model_config = ConfigDict(extra="allow")

    progress: float = Field(default=0.0, description="0.0..1.0 completion fraction.")
    current_bar: int = Field(default=0)
    total_bars: int = Field(default=0)
    trades_generated: int = Field(default=0)
    status: str = Field(default="running", description="starting/loading/running/saving/completed.")
    message: str = Field(default="", description="Human-readable status message.")


class BacktestResponse(BaseModel):
    """Result of a backtest run (cached-from-DB or freshly generated)."""

    model_config = ConfigDict(extra="allow")

    success: bool = Field(default=True)
    source: str = Field(default="generated", description="database/generated/error.")
    trade_count: int = Field(default=0)
    trades: list[TradeResponse] = Field(default_factory=list)
    summary: BacktestSummary | None = Field(default=None)
    processing_time_ms: float = Field(default=0.0)
    date_range: dict | None = Field(default=None)


class HealthResponse(BaseModel):
    """Health-check payload for GET /api/v1/health."""

    model_config = ConfigDict(extra="allow")

    status: str = Field(default="healthy", description="healthy/degraded.")
    version: str = Field(default="unknown")
    model_loaded: bool = Field(default=False)
    database_connected: bool = Field(default=False)
    timestamp: str = Field(default="")
