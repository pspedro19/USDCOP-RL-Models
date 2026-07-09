#!/usr/bin/env python
"""
Export per-asset daily OHLCV → dashboard chart JSON (file-driven, SSOT).
=======================================================================

The dashboard backtest chart (`TradingChartWithSignals` → `/api/market/candlesticks-filtered`)
serves USD/COP intraday bars from Postgres. Multi-asset strategies (BTC, Gold) are
**daily** and have no live DB feed in the file-driven web app, so their price series
must be published as static JSON alongside the strategy bundles.

This exporter reads the canonical daily OHLCV seed parquets and writes one JSON per
`chart_symbol` into `public/data/market/`, matching the shape the chart endpoint returns:

    [{ "time": <epoch_ms>, "open", "high", "low", "close", "volume" }, ...]

Keyed by `chart_symbol` (BTCUSDT, XAUUSD) so the endpoint can resolve symbol → file.
Additive & idempotent: re-run any time; safe to wire into the asset pipeline DAG.

Usage:
    python -m scripts.data.export_chart_ohlcv           # all configured assets
    python -m scripts.data.export_chart_ohlcv --asset btcusdt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
SEEDS_DIR = REPO_ROOT / "seeds" / "latest"
OUT_DIR = REPO_ROOT / "usdcop-trading-dashboard" / "public" / "data" / "market"

from src.contracts.asset_profile import load_asset_profile  # noqa: E402

# Which assets get a **file-driven** daily chart (USD/COP is served from Postgres, so it's excluded).
# This is a deployment choice; chart_symbol + seed path are read from the AssetProfile SSOT (no dup).
DEFAULT_ASSETS: tuple[str, ...] = ("btcusdt", "xauusd")


def _seed_for(profile) -> Path:
    """Resolve the daily OHLCV seed from the AssetProfile (falls back to <safe_name>_daily)."""
    rel = profile.data_source.daily_seed_file or f"seeds/latest/{profile.safe_name}_daily_ohlcv.parquet"
    return REPO_ROOT / rel


def export_asset(asset_id: str) -> Path:
    profile = load_asset_profile(asset_id)
    chart_symbol = profile.chart_symbol
    seed = _seed_for(profile)
    if not seed.exists():
        raise FileNotFoundError(f"Missing daily OHLCV seed for {asset_id}: {seed}")

    df = pd.read_parquet(seed)
    df = df[["time", "open", "high", "low", "close", "volume"]].dropna(
        subset=["open", "high", "low", "close"]
    )
    # time is tz-aware (UTC); emit epoch-ms to match the DB endpoint's `.getTime()` output.
    ts = pd.to_datetime(df["time"], utc=True)
    df = df.assign(time_ms=(ts.astype("int64") // 1_000_000)).sort_values("time_ms")

    records = [
        {
            "time": int(r.time_ms),
            "open": float(r.open),
            "high": float(r.high),
            "low": float(r.low),
            "close": float(r.close),
            "volume": float(r.volume) if pd.notna(r.volume) else 0.0,
        }
        for r in df.itertuples(index=False)
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{chart_symbol}_daily.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(records, separators=(",", ":")), encoding="utf-8")
    tmp.replace(out)  # atomic
    print(f"[export_chart_ohlcv] {asset_id}: {len(records)} bars -> {out.relative_to(REPO_ROOT)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asset", help="Export a single asset (default: all file-driven daily assets)")
    args = ap.parse_args()
    for asset_id in ([args.asset] if args.asset else DEFAULT_ASSETS):
        export_asset(asset_id)


if __name__ == "__main__":
    main()
