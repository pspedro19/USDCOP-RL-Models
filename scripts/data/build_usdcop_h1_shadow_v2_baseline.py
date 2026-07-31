"""Build the immutable pre-W31 baseline for USD/COP H1 shadow v2."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "data/experiments/usdcop_intraday_latam_lead_v1"
OUTPUT_DIR = ROOT / "data/experiments/usdcop_h1_regime_shadow_v2"
CUTOFF = pd.Timestamp("2026-07-21", tz="UTC")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    deep = pd.read_parquet(SOURCE_DIR / "asset_daily_ohlcv.parquet")
    deep_time = pd.to_datetime(deep["time"], utc=True)
    deep = deep[
        deep["symbol"].eq("USD/COP")
        & deep["source"].eq("twelvedata_daily_deep")
        & deep_time.le(CUTOFF)
    ].copy()
    current = pd.read_parquet(SOURCE_DIR / "usdcop_daily_ohlcv.parquet")
    current_time = pd.to_datetime(current["time"], utc=True)
    current = current[current_time.le(CUTOFF)].copy()
    if deep.empty or current.empty:
        raise RuntimeError("Baseline sources produced an empty USD/COP slice")
    if pd.to_datetime(current["time"], utc=True).max() != CUTOFF:
        raise RuntimeError("Frozen current baseline does not end exactly on cutoff")
    deep_path = OUTPUT_DIR / "asset_daily_ohlcv_through_2026-07-21.parquet"
    current_path = OUTPUT_DIR / "usdcop_daily_ohlcv_through_2026-07-21.parquet"
    deep.to_parquet(deep_path, index=False)
    current.to_parquet(current_path, index=False)
    manifest = {
        "schema_version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cutoff": CUTOFF.date().isoformat(),
        "historical_availability_class": "reconstructed_not_true_pit",
        "files": {
            deep_path.name: {
                "rows": int(len(deep)),
                "start": pd.to_datetime(deep["time"], utc=True).min().isoformat(),
                "end": pd.to_datetime(deep["time"], utc=True).max().isoformat(),
                "sha256": sha256(deep_path),
            },
            current_path.name: {
                "rows": int(len(current)),
                "start": pd.to_datetime(current["time"], utc=True).min().isoformat(),
                "end": pd.to_datetime(current["time"], utc=True).max().isoformat(),
                "sha256": sha256(current_path),
            },
        },
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
