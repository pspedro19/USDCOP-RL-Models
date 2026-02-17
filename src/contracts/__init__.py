"""Universal strategy contracts (SDD)."""

from src.contracts.strategy_schema import (
    StrategyTrade,
    StrategyStats,
    StrategySummary,
    StrategyTradeFile,
    GateResult,
    ApprovalState,
    safe_json_dump,
)
from src.contracts.signal_contract import (
    UniversalSignalRecord,
    SignalDirection,
    BarFrequency,
    EntryType,
    SignalStore,
)
from src.contracts.replay_engine import (
    ReplayBacktestEngine,
    ReplayResult,
)
from src.contracts.execution_strategies import (
    ExecutionStrategy,
    WeeklyTPHSExecution,
    DailyTrailingStopExecution,
    IntradaySLTPExecution,
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
