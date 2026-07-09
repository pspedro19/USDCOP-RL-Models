"""
Request schemas for the inference/backtest API.

Reconstructed from usage in `routers/backtest.py` (BacktestRequest.model_id,
start_date, end_date, force_regenerate).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class BacktestRequest(BaseModel):
    """Request body for POST /api/v1/backtest[/stream]."""

    model_config = ConfigDict(extra="allow")

    model_id: str = Field(
        default="ppo_primary",
        description="Model ID to use (investor_demo for demo, ppo_primary for real).",
    )
    start_date: str = Field(description="Start date in YYYY-MM-DD format.")
    end_date: str = Field(description="End date in YYYY-MM-DD format.")
    force_regenerate: bool = Field(
        default=False,
        description="Force regeneration even if trades already exist for the range.",
    )
