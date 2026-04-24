"""Universal strategy contracts (SDD)."""

from src.contracts.execution_strategies import (
    DailyTrailingStopExecution,
    ExecutionStrategy,
    IntradaySLTPExecution,
    WeeklyTPHSExecution,
)
from src.contracts.replay_engine import (
    ReplayBacktestEngine,
    ReplayResult,
)
from src.contracts.signal_contract import (
    BarFrequency,
    EntryType,
    SignalDirection,
    SignalStore,
    UniversalSignalRecord,
)
from src.contracts.strategy_schema import (
    ApprovalState,
    GateResult,
    StrategyStats,
    StrategySummary,
    StrategyTrade,
    StrategyTradeFile,
    safe_json_dump,
)

__all__ = [
    # Strategy schema
    "StrategyTrade",
    "StrategyStats",
    "StrategySummary",
    "StrategyTradeFile",
    "GateResult",
    "ApprovalState",
    "safe_json_dump",
    # Signal contract
    "UniversalSignalRecord",
    "SignalDirection",
    "BarFrequency",
    "EntryType",
    "SignalStore",
    # Replay engine
    "ReplayBacktestEngine",
    "ReplayResult",
    # Execution strategies
    "ExecutionStrategy",
    "WeeklyTPHSExecution",
    "DailyTrailingStopExecution",
    "IntradaySLTPExecution",
]
