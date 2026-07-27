"""Acquire reproducible public daily snapshots with explicit availability policy.

This is an acquisition/lineage step, not a production PIT claim. Yahoo Finance data has
no historical revision vintages in this adapter, so manifests are marked
``reconstructed_availability`` and cannot pass the production OOS gate until a vintaged
provider is attached.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
SPECS = {
    "usdcop": {"ticker": "COP=X", "lag_days": 1},
    "xauusd": {"ticker": "GC=F", "lag_days": 1},
    "btcusdt": {"ticker": "BTC-USD", "lag_days": 0},
    # spx500 RETIRADO (directiva operador 2026-07-27): la SSOT del S&P 500 es el
    # indice OFICIAL de Investing (seeds/latest/spx500_daily_ohlcv.parquet, 1995->),
    # ingerido por asset_spx500_pipeline_weekly. El snapshot SPY quedo eliminado.
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def acquire(asset: str, start: str, end: str, output: Path) -> dict:
    spec = SPECS[asset]
    retrieved = datetime.now(timezone.utc)
    frame = yf.download(spec["ticker"], start=start, end=end, auto_adjust=False,
                        progress=False, group_by="column")
    if frame.empty:
        raise RuntimeError(f"No public data returned for {asset} ({spec['ticker']})")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame = frame.rename(columns={c: c.lower().replace(" ", "_") for c in frame.columns})
    frame = frame.reset_index()
    date_col = next((c for c in frame.columns if str(c).lower() in {"date", "datetime", "index"}), None)
    if date_col is None:
        raise RuntimeError(f"No timestamp column returned for {asset}: {list(frame.columns)}")
    frame = frame.rename(columns={date_col: "timestamp"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["available_at"] = frame["timestamp"] + pd.to_timedelta(spec["lag_days"], unit="D")
    frame["asset_id"] = asset
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / f"{asset}_daily.parquet"
    frame.to_parquet(data_path, index=False)
    manifest = {
        "schema_version": 1, "asset_id": asset, "ticker": spec["ticker"],
        "source": "yahoo_finance_public_adapter", "retrieved_at": retrieved.isoformat(),
        "start": start, "end": end, "rows": len(frame),
        "data_path": str(data_path.relative_to(ROOT)), "sha256": sha256(data_path),
        "availability_policy": "reconstructed_close_plus_conservative_lag",
        "pit_vintage": False, "promotion_eligible": False,
        "reason": "Public adapter lacks historical provider revision vintages.",
    }
    (output / f"{asset}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=datetime.now(timezone.utc).date().isoformat())
    ap.add_argument("--asset", choices=[*SPECS, "all"], default="all")
    ap.add_argument("--output", type=Path, default=ROOT / "data" / "snapshots" / "public_daily")
    ns = ap.parse_args()
    assets = SPECS if ns.asset == "all" else {ns.asset: SPECS[ns.asset]}
    manifests = [acquire(a, ns.start, ns.end, ns.output) for a in assets]
    print(json.dumps({"snapshots": manifests, "promotion_eligible": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
