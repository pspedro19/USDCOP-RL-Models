"""
Replay Backtest (Universal)
============================

CLI tool that replays UniversalSignalRecord[] through OHLCV data,
producing StrategyTrade[] and optionally exporting to the dashboard.

Usage:
    # Replay H5 signals through OHLCV (generates signals on-the-fly if no parquet)
    python scripts/replay_backtest_universal.py --strategy smart_simple_v11 --year 2025

    # Replay from pre-generated signal file
    python scripts/replay_backtest_universal.py --signals data/signals/smart_simple_v11_2025.parquet --year 2025

    # Export to dashboard (summary.json + trades + approval_state.json)
    python scripts/replay_backtest_universal.py --strategy smart_simple_v11 --year 2025 --export-dashboard

    # Compare multiple strategies side by side
    python scripts/replay_backtest_universal.py --compare smart_simple_v11,forecast_vt_trailing --year 2025

Contract: CTR-REPLAY-CLI-001
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.contracts.signal_contract import SignalStore, UniversalSignalRecord
from src.contracts.signal_adapters import (
    ADAPTER_REGISTRY,
    H5SmartSimpleAdapter,
    H1ForecastVTAdapter,
    load_forecasting_data,
)
from src.contracts.execution_strategies import (
    WeeklyTPHSExecution,
    DailyTrailingStopExecution,
    IntradaySLTPExecution,
)
from src.contracts.replay_engine import ReplayBacktestEngine, ReplayResult
from src.contracts.strategy_schema import (
    StrategySummary,
    StrategyTradeFile,
    GateResult,
    ApprovalState,
    safe_json_dump,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Strategy -> Execution mapping
EXECUTION_MAP = {
    "smart_simple_v11": lambda: WeeklyTPHSExecution(),
    "forecast_vt_trailing": lambda: DailyTrailingStopExecution(),
    "rl_v215b": lambda: IntradaySLTPExecution(),
}

# Strategy display names
STRATEGY_NAMES = {
    "smart_simple_v11": "Smart Simple v1.1.0",
    "forecast_vt_trailing": "Forecast + VT + Trail",
    "rl_v215b": "RL V21.5b",
}


def generate_or_load_signals(strategy_id, year, signals_path=None):
    """Generate signals on-the-fly or load from file."""
    if signals_path:
        path = Path(signals_path)
        if path.suffix == ".json":
            return SignalStore.load_json(path)
        return SignalStore.load_parquet(path)

    # Try loading from default path
    default_path = PROJECT_ROOT / "data" / "signals" / f"{strategy_id}_{year}.parquet"
    if default_path.exists():
        logger.info("Loading signals from %s", default_path)
        return SignalStore.load_parquet(default_path)

    # Generate on-the-fly
    logger.info("Generating signals for %s %d on-the-fly...", strategy_id, year)

    if strategy_id == "smart_simple_v11":
        adapter = H5SmartSimpleAdapter()
        df, feature_cols = load_forecasting_data()
        return adapter.generate_signals(df, feature_cols, year)

    elif strategy_id == "forecast_vt_trailing":
        adapter = H1ForecastVTAdapter()
        df, feature_cols = load_forecasting_data()
        return adapter.generate_signals(df, feature_cols, year)

    else:
        raise ValueError(
            f"Cannot auto-generate signals for {strategy_id}. "
            f"Provide --signals path or run generate_universal_signals.py first."
        )


def load_ohlcv(strategy_id):
    """Load the appropriate OHLCV data for the strategy."""
    if strategy_id in ("smart_simple_v11", "forecast_vt_trailing"):
        path = PROJECT_ROOT / "seeds" / "latest" / "usdcop_daily_ohlcv.parquet"
        df = pd.read_parquet(path).reset_index()
        df.rename(columns={"time": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        return df[["date", "open", "high", "low", "close"]].sort_values("date")
    else:  # RL uses 5-min bars
        path = PROJECT_ROOT / "seeds" / "latest" / "usdcop_m5_ohlcv.parquet"
        df = pd.read_parquet(path).reset_index()
        if "time" in df.columns:
            df.rename(columns={"time": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        return df[["date", "open", "high", "low", "close"]].sort_values("date")


def run_replay(strategy_id, year, signals_path=None):
    """Run replay for a single strategy."""
    signals = generate_or_load_signals(strategy_id, year, signals_path)
    ohlcv = load_ohlcv(strategy_id)

    execution = EXECUTION_MAP[strategy_id]()
    engine = ReplayBacktestEngine(execution, initial_capital=10000.0)
    result = engine.replay(signals, ohlcv)

    return result, signals


def print_result(strategy_id, year, result: ReplayResult):
    """Print replay results in standard format."""
    s = result.stats
    t = result.statistical_tests

    print(f"\n  {STRATEGY_NAMES.get(strategy_id, strategy_id)} ({year})")
    print(f"  {'-' * 55}")
    print(f"  $10K -> ${s.final_equity:,.2f}  ({s.total_return_pct:+.2f}%)")
    print(f"  Trades: {len(result.trades)} ({s.n_long or 0}L / {s.n_short or 0}S)")
    print(f"  WR: {s.win_rate_pct:.1f}%  |  Sharpe: {s.sharpe:.3f}" if s.sharpe else f"  WR: {s.win_rate_pct:.1f}%")
    print(f"  MaxDD: {s.max_dd_pct:.2f}%  |  PF: {s.profit_factor:.2f}" if s.profit_factor else f"  MaxDD: {s.max_dd_pct:.2f}%")
    print(f"  p-value: {t['p_value']:.4f}  {'*** SIGNIFICANT ***' if t['significant'] else '(not significant)'}")

    if s.exit_reasons:
        exits = ", ".join(f"{k}={v}" for k, v in sorted(s.exit_reasons.items()))
        print(f"  Exits: {exits}")

    print(f"  Signals: {result.signals_total} total, {result.signals_traded} traded, {result.signals_skipped} skipped")


def export_dashboard(strategy_id, year, result: ReplayResult):
    """Export results to dashboard format (SDD pipeline)."""
    dashboard_dir = PROJECT_ROOT / "usdcop-trading-dashboard" / "public" / "data" / "production"
    trades_dir = dashboard_dir / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)

    s = result.stats
    t = result.statistical_tests
    name = STRATEGY_NAMES.get(strategy_id, strategy_id)
    now = datetime.now().isoformat()

    # --- summary_{year}.json ---
    bh_df = load_ohlcv(strategy_id)
    bh_start = bh_df[bh_df["date"] >= pd.Timestamp(f"{year}-01-01")].iloc[0]["close"]
    bh_end = bh_df[bh_df["date"] <= pd.Timestamp(f"{year}-12-31")].iloc[-1]["close"]
    bh_return = (bh_end / bh_start - 1) * 100

    summary = StrategySummary(
        generated_at=now,
        strategy_name=name,
        strategy_id=strategy_id,
        year=year,
        initial_capital=10000.0,
        n_trading_days=s.trading_days or 0,
        strategies={
            "buy_and_hold": {
                "final_equity": round(10000 * (1 + bh_return / 100), 2),
                "total_return_pct": round(bh_return, 2),
            },
            strategy_id: s.to_dict(),
        },
        statistical_tests=t,
    )

    summary_file = dashboard_dir / f"summary_{year}.json"
    with open(summary_file, "w") as f:
        safe_json_dump(summary.to_dict(), f)
    logger.info("  Wrote: %s", summary_file)

    # --- trades/{strategy_id}_{year}.json ---
    trade_dicts = [tr.to_dict() for tr in result.trades]
    trade_file = StrategyTradeFile(
        strategy_name=name,
        strategy_id=strategy_id,
        initial_capital=10000.0,
        date_range={
            "start": result.trades[0].timestamp[:10] if result.trades else "",
            "end": result.trades[-1].exit_timestamp[:10] if result.trades and result.trades[-1].exit_timestamp else "",
        },
        trades=trade_dicts,
        summary={
            "total_trades": len(result.trades),
            "winning_trades": sum(1 for tr in result.trades if tr.pnl_pct > 0),
            "losing_trades": sum(1 for tr in result.trades if tr.pnl_pct <= 0),
            "win_rate": s.win_rate_pct,
            "total_pnl": round(s.final_equity - 10000, 2),
            "total_return_pct": s.total_return_pct,
            "max_drawdown_pct": s.max_dd_pct,
            "sharpe_ratio": s.sharpe,
            "profit_factor": s.profit_factor,
            "p_value": t["p_value"],
            "n_long": s.n_long,
            "n_short": s.n_short,
        },
    )

    trades_path = trades_dir / f"{strategy_id}_{year}.json"
    with open(trades_path, "w") as f:
        safe_json_dump(trade_file.to_dict(), f)
    logger.info("  Wrote: %s", trades_path)

    # --- approval_state.json ---
    gates = [
        GateResult("min_return_pct", "Retorno Minimo", s.total_return_pct > -15, s.total_return_pct, -15.0),
        GateResult("min_sharpe_ratio", "Sharpe Minimo", (s.sharpe or 0) > 0, s.sharpe or 0, 0.0),
        GateResult("max_drawdown_pct", "Max Drawdown", (s.max_dd_pct or 0) < 20, s.max_dd_pct or 0, 20.0),
        GateResult("min_trades", "Trades Minimos", len(result.trades) >= 10, len(result.trades), 10.0),
        GateResult("statistical_significance", "Significancia (p<0.05)", t["p_value"] < 0.05, t["p_value"], 0.05),
    ]

    all_passed = all(g.passed for g in gates)
    any_critical_fail = not gates[0].passed or not gates[2].passed  # return or drawdown

    if all_passed:
        recommendation = "PROMOTE"
    elif any_critical_fail:
        recommendation = "REJECT"
    else:
        recommendation = "REVIEW"

    approval = ApprovalState(
        status="PENDING_APPROVAL",
        strategy=strategy_id,
        strategy_name=name,
        backtest_year=year,
        backtest_recommendation=recommendation,
        backtest_confidence=sum(1 for g in gates if g.passed) / len(gates),
        gates=[g.to_dict() for g in gates],
        backtest_metrics={
            "return_pct": s.total_return_pct,
            "sharpe": s.sharpe,
            "max_dd_pct": s.max_dd_pct,
            "p_value": t["p_value"],
            "trades": len(result.trades),
            "win_rate_pct": s.win_rate_pct,
        },
        created_at=now,
        last_updated=now,
    )

    approval_path = dashboard_dir / "approval_state.json"
    with open(approval_path, "w") as f:
        safe_json_dump(approval.to_dict(), f)
    logger.info("  Wrote: %s", approval_path)


def run_comparison(strategy_ids, year):
    """Run and compare multiple strategies side by side."""
    results = {}
    for sid in strategy_ids:
        if sid not in EXECUTION_MAP:
            logger.warning("  Skipping unknown strategy: %s", sid)
            continue
        result, _ = run_replay(sid, year)
        results[sid] = result

    # Print comparison table
    print(f"\n{'=' * 90}")
    print(f"  STRATEGY COMPARISON — {year}")
    print(f"{'=' * 90}")
    print(f"\n  {'Strategy':<28s} {'Return':>8s} {'Sharpe':>7s} {'MaxDD':>7s} {'WR':>6s} {'p-val':>7s} {'Trades':>7s} {'$10K':>10s}")
    print(f"  {'-' * 88}")

    for sid, result in results.items():
        s = result.stats
        t = result.statistical_tests
        name = STRATEGY_NAMES.get(sid, sid)[:27]
        sharpe_str = f"{s.sharpe:7.3f}" if s.sharpe is not None else "    N/A"
        print(
            f"  {name:<28s} {s.total_return_pct:>+7.2f}% {sharpe_str} "
            f"{s.max_dd_pct:>6.2f}% {s.win_rate_pct:>5.1f}% {t['p_value']:>7.4f} "
            f"{len(result.trades):>7d} ${s.final_equity:>9,.2f}"
        )

    print()
    return results


def main():
    parser = argparse.ArgumentParser(description="Replay backtest with universal signals")
    parser.add_argument("--strategy", type=str, default=None,
                        help="Strategy ID to replay")
    parser.add_argument("--signals", type=str, default=None,
                        help="Path to pre-generated signals file")
    parser.add_argument("--year", type=int, default=2025,
                        help="Year for backtest (default: 2025)")
    parser.add_argument("--export-dashboard", action="store_true",
                        help="Export results to dashboard format")
    parser.add_argument("--compare", type=str, default=None,
                        help="Comma-separated list of strategy IDs to compare")
    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print(f"  Universal Replay Backtest Engine")
    print(f"{'=' * 70}")

    # --- Comparison mode ---
    if args.compare:
        strategy_ids = [s.strip() for s in args.compare.split(",")]
        run_comparison(strategy_ids, args.year)
        return

    # --- Single strategy mode ---
    if not args.strategy and not args.signals:
        parser.error("Either --strategy or --signals is required (or use --compare)")

    strategy_id = args.strategy
    if not strategy_id and args.signals:
        # Infer from signal file
        signals = SignalStore.load_parquet(Path(args.signals)) if args.signals.endswith(".parquet") else SignalStore.load_json(Path(args.signals))
        if signals:
            strategy_id = signals[0].strategy_id

    if strategy_id not in EXECUTION_MAP:
        logger.error("Unknown strategy: %s. Known: %s", strategy_id, list(EXECUTION_MAP.keys()))
        sys.exit(1)

    result, signals = run_replay(strategy_id, args.year, args.signals)
    print_result(strategy_id, args.year, result)

    # --- Trade-by-trade output ---
    if len(result.trades) <= 50:
        print(f"\n  --- Trade Details ---")
        for tr in result.trades:
            meta = tr.metadata or {}
            conf = meta.get("confidence_tier", "-")
            hs = meta.get("hard_stop_pct", "-")
            tp = meta.get("take_profit_pct", "-")
            print(
                f"    {tr.timestamp[:10]}  {tr.side:5s}  conf={conf:>6s}  "
                f"lev={tr.leverage:.3f}  entry={tr.entry_price:.0f}  "
                f"exit={tr.exit_price:.0f}  pnl={tr.pnl_pct:+.3f}%  "
                f"[{tr.exit_reason}]"
            )

    # --- Export ---
    if args.export_dashboard:
        print(f"\n  Exporting to dashboard...")
        export_dashboard(strategy_id, args.year, result)
        print(f"  Dashboard export complete.")

    print()


if __name__ == "__main__":
    main()
