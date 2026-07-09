"""
Pydantic request/response schemas for the inference/backtest API.

NOTE: this package was historically swallowed by the broad `models/` .gitignore
rule and lost from version control. It is reconstructed here from its usage across
`routers/` and `orchestrator/` and re-added to git via a `.gitignore` negation
(mirroring the `src/forecasting/models/` fix). Do not delete.
"""

from .requests import BacktestRequest
from .responses import (
    BacktestResponse,
    BacktestSummary,
    HealthResponse,
    ProgressUpdate,
    TradeResponse,
)

__all__ = [
    "BacktestRequest",
    "BacktestResponse",
    "BacktestSummary",
    "HealthResponse",
    "ProgressUpdate",
    "TradeResponse",
]
