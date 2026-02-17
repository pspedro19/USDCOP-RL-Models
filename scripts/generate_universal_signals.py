"""
Generate Universal Signals
==========================

CLI tool that generates UniversalSignalRecord[] for any registered strategy.

Usage:
    # H5 Smart Simple for 2025 backtest
    python scripts/generate_universal_signals.py --strategy smart_simple_v11 --year 2025

    # H1 Forecast+VT+Trail for 2025
    python scripts/generate_universal_signals.py --strategy forecast_vt_trailing --year 2025

    # RL PPO for 2025 (requires trained model)
    python scripts/generate_universal_signals.py --strategy rl_v215b --year 2025 \\
        --model-path models/ppo_v215b/best_model.zip

    # Output as JSON instead of parquet
    python scripts/generate_universal_signals.py --strategy smart_simple_v11 --year 2025 --format json

Output:
    data/signals/{strategy_id}_{year}.parquet  (default)
    data/signals/{strategy_id}_{year}.json     (with --format json)

Contract: CTR-SIGNAL-GEN-001
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.contracts.signal_contract import SignalStore
from src.contracts.signal_adapters import (
    ADAPTER_REGISTRY,
    H5SmartSimpleAdapter,
    H1ForecastVTAdapter,
    RLPPOAdapter,
    load_forecasting_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate universal signals for any strategy")
    parser.add_argument("--strategy", required=True, choices=list(ADAPTER_REGISTRY.keys()),
                        help="Strategy ID")
    parser.add_argument("--year", type=int, default=2025,
                        help="Year to generate signals for (default: 2025)")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional config path override")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Model path (required for RL)")
    parser.add_argument("--norm-stats", type=str, default=None,
                        help="Norm stats path (for RL)")
    parser.add_argument("--format", choices=["parquet", "json"], default="parquet",
                        help="Output format (default: parquet)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: data/signals/)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else (PROJECT_ROOT / "data" / "signals")
    output_dir.mkdir(parents=True, exist_ok=True)

    strategy = args.strategy
    year = args.year

    logger.info("=" * 70)
    logger.info("  Universal Signal Generator")
    logger.info("  Strategy: %s  |  Year: %d", strategy, year)
    logger.info("=" * 70)

    # --- Generate signals ---
    if strategy == "smart_simple_v11":
        config_path = Path(args.config) if args.config else None
        adapter = H5SmartSimpleAdapter(config_path=config_path)
        df, feature_cols = load_forecasting_data()
        signals = adapter.generate_signals(df, feature_cols, year)

    elif strategy == "forecast_vt_trailing":
        config_path = Path(args.config) if args.config else None
        adapter = H1ForecastVTAdapter(config_path=config_path)
        df, feature_cols = load_forecasting_data()
        signals = adapter.generate_signals(df, feature_cols, year)

    elif strategy == "rl_v215b":
        import pandas as pd
        model_path = Path(args.model_path) if args.model_path else None
        norm_stats = Path(args.norm_stats) if args.norm_stats else None
        config_path = Path(args.config) if args.config else None

        adapter = RLPPOAdapter(
            model_path=model_path,
            norm_stats_path=norm_stats,
            config_path=config_path,
        )

        # Load 5-min OHLCV
        ohlcv_path = PROJECT_ROOT / "seeds" / "latest" / "usdcop_m5_ohlcv.parquet"
        ohlcv = pd.read_parquet(ohlcv_path).reset_index()
        if "time" in ohlcv.columns:
            ohlcv.rename(columns={"time": "date"}, inplace=True)
        ohlcv["date"] = pd.to_datetime(ohlcv["date"]).dt.tz_localize(None)
        signals = adapter.generate_signals(ohlcv, year)

    else:
        logger.error("Unknown strategy: %s", strategy)
        sys.exit(1)

    # --- Summary ---
    total = len(signals)
    skipped = sum(1 for s in signals if s.skip_trade)
    traded = total - skipped
    longs = sum(1 for s in signals if s.direction == 1 and not s.skip_trade)
    shorts = sum(1 for s in signals if s.direction == -1 and not s.skip_trade)

    logger.info("")
    logger.info("  Signals generated: %d total (%d traded, %d skipped)", total, traded, skipped)
    logger.info("  Direction: %d LONG, %d SHORT", longs, shorts)

    if signals:
        magnitudes = [s.magnitude for s in signals if not s.skip_trade]
        leverages = [s.leverage for s in signals if not s.skip_trade]
        if magnitudes:
            logger.info("  Magnitude: mean=%.4f, min=%.4f, max=%.4f",
                        sum(magnitudes) / len(magnitudes), min(magnitudes), max(magnitudes))
        if leverages:
            logger.info("  Leverage: mean=%.3f, min=%.3f, max=%.3f",
                        sum(leverages) / len(leverages), min(leverages), max(leverages))

    # --- Save ---
    if args.format == "parquet":
        out_path = output_dir / f"{strategy}_{year}.parquet"
        SignalStore.save_parquet(signals, out_path)
    else:
        out_path = output_dir / f"{strategy}_{year}.json"
        SignalStore.save_json(signals, out_path)

    logger.info("  Saved to: %s", out_path)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
