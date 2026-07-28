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
from dataclasses import asdict, dataclass


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
    exit_timestamp: str | None = None
    metadata: dict | None = None

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
    sharpe: float | None = None
    max_dd_pct: float | None = None
    win_rate_pct: float | None = None
    profit_factor: float | None = None   # null if no losses (NEVER Infinity)
    trading_days: int | None = None
    exit_reasons: dict | None = None
    n_long: int | None = None
    n_short: int | None = None

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
    direction_accuracy_pct: float | None = None
    monthly: dict | None = None
    # BL-13/C-005 (optional, additive): "action" = tradeable strategy, "diagnostic" =
    # look-only research surface. A diagnostic surface can never be champion/visible.
    surface: str | None = None           # "action" | "diagnostic"

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
    strategy_name: str | None = None
    backtest_year: int | None = None
    backtest_metrics: dict | None = None
    approved_by: str | None = None
    approved_at: str | None = None
    reviewer_notes: str | None = None
    rejected_by: str | None = None
    rejected_at: str | None = None
    rejection_reason: str | None = None

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


# ---------------------------------------------------------------------------
# Small-sample guard (quant-constitution.md §6)
# ---------------------------------------------------------------------------
MIN_TRADES_FOR_STATS = 20


def suppress_small_sample_stats(summary: dict, n_trades: int | None = None) -> dict:
    """Null out inferential statistics when the trade count is too small to support them.

    The constitution is explicit: "con N < 20 trades se reporta solo conteo y PnL (nada de
    'Sharpe 19, p=0.000, 3 trades')". Three published bundles were breaking it — COP with
    5 trades reporting Sharpe -0.949 and p=0.5944, BTC with **3 trades** reporting Sharpe
    -1.617, p=0.9562 and a PSR, Gold with 7 trades reporting Sharpe 0.618.

    A p-value on 3 trades is not a weak result; it is not a result. Rendering it next to
    a real one invites the reader to compare them, which is the harm.

    Descriptive quantities survive (return, drawdown, win rate, counts) — they describe
    what happened. Inferential ones (Sharpe, p-value, PSR, bootstrap CI) are nulled, and
    ``insufficient_trades`` is set so the UI can say why instead of rendering a blank.

    Mutates and returns ``summary`` for convenience at call sites.
    """
    if n_trades is None:
        n_trades = summary.get("n_trades")
        if n_trades is None:
            for block in (summary.get("strategies") or {}).values():
                if isinstance(block, dict):
                    lo, sh = block.get("n_long"), block.get("n_short")
                    if isinstance(lo, int) and isinstance(sh, int):
                        n_trades = lo + sh
                        break
    if n_trades is None or n_trades >= MIN_TRADES_FOR_STATS:
        return summary

    summary["n_trades"] = n_trades
    summary["insufficient_trades"] = True

    tests = summary.get("statistical_tests")
    if isinstance(tests, dict):
        for key in ("p_value", "bootstrap_95ci_ann", "psr", "sharpe_per_period",
                    "t_stat", "dsr", "deflated_sharpe"):
            if key in tests:
                tests[key] = None
        tests["significant"] = False
        tests["insufficient_trades"] = True
        tests["min_trades_for_stats"] = MIN_TRADES_FOR_STATS

    for sid, block in (summary.get("strategies") or {}).items():
        if sid == "buy_and_hold" or not isinstance(block, dict):
            continue
        for key in ("sharpe", "sortino", "calmar"):
            if key in block:
                block[key] = None
        block["insufficient_trades"] = True

    return summary
