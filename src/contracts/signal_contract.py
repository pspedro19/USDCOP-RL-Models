"""
Universal Signal Contract
=========================
Standard signal format that decouples signal generation from execution.

Any strategy (RL, H1 ML, H5 ML) produces UniversalSignalRecord[],
and a single ReplayBacktestEngine executes them against OHLCV data
to produce StrategyTrade[].

Spec: .claude/rules/sdd-strategy-spec.md
Contract: CTR-SIGNAL-001
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from enum import Enum, IntEnum
from pathlib import Path
from typing import List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SignalDirection(IntEnum):
    SHORT = -1
    HOLD = 0
    LONG = 1


class BarFrequency(str, Enum):
    FIVE_MIN = "5min"
    DAILY = "daily"
    WEEKLY = "weekly"


class EntryType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"


# ---------------------------------------------------------------------------
# Core Dataclass
# ---------------------------------------------------------------------------

@dataclass
class UniversalSignalRecord:
    """
    Universal signal format for all strategy types.

    One record = one trading decision (enter, hold, or skip).
    The ReplayBacktestEngine reads these and simulates execution.
    """

    # --- Identity ---
    signal_id: str              # "h5_2025-W03", "h1_2025-01-15", "rl_2025-01-15T13:00"
    strategy_id: str            # "smart_simple_v11", "forecast_vt_trailing", "rl_v215b"
    signal_date: str            # ISO date of signal generation

    # --- Core Signal (universal) ---
    direction: int              # +1 (LONG), -1 (SHORT), 0 (HOLD/SKIP)
    magnitude: float            # |predicted_return| or |raw_action| (signal strength)
    confidence: float           # 0.0 to 1.0 (normalized)
    skip_trade: bool            # True if strategy says "don't trade this one"

    # --- Position Sizing ---
    leverage: float             # Final leverage after vol-target + multipliers

    # --- Stop/Target Levels (None = not used) ---
    hard_stop_pct: Optional[float] = None       # e.g. 0.0281 for 2.81%
    take_profit_pct: Optional[float] = None
    trailing_activation_pct: Optional[float] = None
    trailing_distance_pct: Optional[float] = None

    # --- Entry ---
    entry_price: float = 0.0           # Price at signal time
    entry_type: str = "limit"          # "limit" | "market"

    # --- Timeframe ---
    horizon_bars: int = 1              # 1 (H1 daily), 5 (H5 weekly), 60 (RL session)
    bar_frequency: str = "daily"       # "daily" | "weekly" | "5min"

    # --- Strategy Metadata (opaque) ---
    metadata: Optional[dict] = None    # model_predictions, confidence_tier, etc.

    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        d = asdict(self)
        # Remove None values for cleaner JSON
        return {k: v for k, v in d.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> UniversalSignalRecord:
        """Create from dict, ignoring unknown keys."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# SignalStore — Read/write UniversalSignalRecord[] as parquet or JSON
# ---------------------------------------------------------------------------

def _sanitize_for_json(obj):
    """Recursively replace Infinity/NaN floats with None."""
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


class SignalStore:
    """Read/write UniversalSignalRecord[] as parquet or JSON."""

    # Columns that map directly to the dataclass (non-metadata)
    _CORE_COLUMNS = [
        "signal_id", "strategy_id", "signal_date",
        "direction", "magnitude", "confidence", "skip_trade",
        "leverage",
        "hard_stop_pct", "take_profit_pct",
        "trailing_activation_pct", "trailing_distance_pct",
        "entry_price", "entry_type",
        "horizon_bars", "bar_frequency",
    ]

    @staticmethod
    def save_parquet(signals: List[UniversalSignalRecord], path: Path) -> None:
        """Save signals to parquet file."""
        if not signals:
            pd.DataFrame().to_parquet(path)
            return

        rows = []
        for s in signals:
            row = {
                "signal_id": s.signal_id,
                "strategy_id": s.strategy_id,
                "signal_date": s.signal_date,
                "direction": s.direction,
                "magnitude": s.magnitude,
                "confidence": s.confidence,
                "skip_trade": s.skip_trade,
                "leverage": s.leverage,
                "hard_stop_pct": s.hard_stop_pct,
                "take_profit_pct": s.take_profit_pct,
                "trailing_activation_pct": s.trailing_activation_pct,
                "trailing_distance_pct": s.trailing_distance_pct,
                "entry_price": s.entry_price,
                "entry_type": s.entry_type,
                "horizon_bars": s.horizon_bars,
                "bar_frequency": s.bar_frequency,
                "metadata_json": json.dumps(s.metadata) if s.metadata else None,
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

    @staticmethod
    def load_parquet(path: Path) -> List[UniversalSignalRecord]:
        """Load signals from parquet file."""
        df = pd.read_parquet(path)
        if df.empty:
            return []

        signals = []
        for _, row in df.iterrows():
            metadata = None
            if "metadata_json" in row and pd.notna(row.get("metadata_json")):
                metadata = json.loads(row["metadata_json"])

            signals.append(UniversalSignalRecord(
                signal_id=str(row["signal_id"]),
                strategy_id=str(row["strategy_id"]),
                signal_date=str(row["signal_date"]),
                direction=int(row["direction"]),
                magnitude=float(row["magnitude"]),
                confidence=float(row["confidence"]),
                skip_trade=bool(row["skip_trade"]),
                leverage=float(row["leverage"]),
                hard_stop_pct=float(row["hard_stop_pct"]) if pd.notna(row.get("hard_stop_pct")) else None,
                take_profit_pct=float(row["take_profit_pct"]) if pd.notna(row.get("take_profit_pct")) else None,
                trailing_activation_pct=float(row["trailing_activation_pct"]) if pd.notna(row.get("trailing_activation_pct")) else None,
                trailing_distance_pct=float(row["trailing_distance_pct"]) if pd.notna(row.get("trailing_distance_pct")) else None,
                entry_price=float(row["entry_price"]),
                entry_type=str(row.get("entry_type", "limit")),
                horizon_bars=int(row.get("horizon_bars", 1)),
                bar_frequency=str(row.get("bar_frequency", "daily")),
                metadata=metadata,
            ))
        return signals

    @staticmethod
    def save_json(signals: List[UniversalSignalRecord], path: Path) -> None:
        """Save signals to JSON file (safe serialization)."""
        data = [s.to_dict() for s in signals]
        data = _sanitize_for_json(data)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @staticmethod
    def load_json(path: Path) -> List[UniversalSignalRecord]:
        """Load signals from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return [UniversalSignalRecord.from_dict(d) for d in data]
