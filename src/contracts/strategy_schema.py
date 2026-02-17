"""
Universal Strategy Schema (SDD)
================================
Python dataclasses mirroring lib/contracts/strategy.contract.ts.

Spec: .claude/rules/sdd-strategy-spec.md
TS mirror: lib/contracts/strategy.contract.ts
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class StrategyTrade:
    """Universal trade record."""
    trade_id: int
    timestamp: str              # ISO8601 with timezone
    side: str                   # "LONG" | "SHORT"
    entry_price: float
    exit_price: float
    pnl_usd: float
    pnl_pct: float
    exit_reason: str
    equity_at_entry: float
    equity_at_exit: float
    leverage: float
    exit_timestamp: Optional[str] = None
    metadata: Optional[dict] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.metadata:
            d.update(self.metadata)
            del d["metadata"]
        elif d.get("metadata") is None:
            del d["metadata"]
        return d


@dataclass
class StrategyStats:
    """Universal strategy stats (goes into summary.strategies[strategy_id])."""
    final_equity: float
    total_return_pct: float
    sharpe: Optional[float] = None
    max_dd_pct: Optional[float] = None
    win_rate_pct: Optional[float] = None
    profit_factor: Optional[float] = None   # null if no losses (NEVER Infinity)
    trading_days: Optional[int] = None
    exit_reasons: Optional[dict] = None
    n_long: Optional[int] = None
    n_short: Optional[int] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class StrategySummary:
    """Universal summary (strategy-agnostic)."""
    generated_at: str
    strategy_name: str
    strategy_id: str
    year: int
    initial_capital: float
    n_trading_days: int
    strategies: dict                     # Record<string, StrategyStats-like dict>
    statistical_tests: dict
    direction_accuracy_pct: Optional[float] = None
    monthly: Optional[dict] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class StrategyTradeFile:
    """Universal trade file."""
    strategy_name: str
    strategy_id: str
    initial_capital: float
    date_range: dict                     # {start, end}
    trades: list                         # List of trade dicts
    summary: dict

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GateResult:
    """One of the validation gates."""
    gate: str
    label: str
    passed: bool
    value: float
    threshold: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ApprovalState:
    """Approval state (persisted as JSON)."""
    status: str                          # PENDING_APPROVAL | APPROVED | REJECTED | LIVE
    strategy: str
    backtest_recommendation: str         # PROMOTE | REVIEW | REJECT
    backtest_confidence: float
    gates: list                          # List[GateResult dicts]
    created_at: str
    last_updated: str
    strategy_name: Optional[str] = None
    backtest_year: Optional[int] = None
    backtest_metrics: Optional[dict] = None
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    reviewer_notes: Optional[str] = None
    rejected_by: Optional[str] = None
    rejected_at: Optional[str] = None
    rejection_reason: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


# ---------------------------------------------------------------------------
# Safe JSON serialization (handles Infinity, NaN, datetime)
# ---------------------------------------------------------------------------

def _sanitize_for_json(obj):
    """Recursively replace Infinity/NaN floats with None, datetimes with ISO strings."""
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


def _json_default(obj):
    """JSON default handler: converts Infinity/NaN to None, datetime to ISO."""
    if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
        return None
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


def safe_json_dump(data, fp, **kwargs):
    """JSON dump that converts Infinity/NaN to null. Use for all dashboard exports.

    NOTE: Python's json.dump default callback is NOT called for float types,
    so we must sanitize the data tree BEFORE serialization.
    """
    kwargs.setdefault("indent", 2)
    kwargs["default"] = _json_default
    sanitized = _sanitize_for_json(data)
    json.dump(sanitized, fp, **kwargs)


def safe_json_dumps(data, **kwargs) -> str:
    """JSON dumps that converts Infinity/NaN to null."""
    kwargs.setdefault("indent", 2)
    kwargs["default"] = _json_default
    sanitized = _sanitize_for_json(data)
    return json.dumps(sanitized, **kwargs)


# ---------------------------------------------------------------------------
# Exit Reason Registry
# ---------------------------------------------------------------------------

EXIT_REASONS = {
    "take_profit":     {"color": "emerald", "label": "Take Profit"},
    "trailing_stop":   {"color": "emerald", "label": "Trailing Stop"},
    "hard_stop":       {"color": "red",     "label": "Hard Stop"},
    "week_end":        {"color": "blue",    "label": "Fin de Semana"},
    "session_close":   {"color": "blue",    "label": "Cierre Sesion"},
    "circuit_breaker": {"color": "amber",   "label": "Circuit Breaker"},
    "no_bars":         {"color": "slate",   "label": "No Bars"},
}
