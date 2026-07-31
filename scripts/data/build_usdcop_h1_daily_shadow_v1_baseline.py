"""Freeze the pre-launch baseline for the prospective daily H1 shadow."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DEEP = (
    ROOT
    / "data/experiments/usdcop_h1_regime_shadow_v2/asset_daily_ohlcv_through_2026-07-21.parquet"
)
SOURCE_CURRENT = ROOT / "seeds/latest/usdcop_daily_ohlcv.parquet"
OUTPUT_DIR = ROOT / "data/experiments/usdcop_h1_daily_shadow_v1"
CUTOFF = pd.Timestamp("2026-07-22", tz="UTC")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate(frame: pd.DataFrame, label: str) -> None:
    required = {"time", "open", "high", "low", "close"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"{label} missing columns: {missing}")
    if frame[list(required - {"time"})].isna().any().any():
        raise RuntimeError(f"{label} contains null OHLC")
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame[["open", "high", "low", "close"]].le(0).any().any():
        raise RuntimeError(f"{label} contains non-positive OHLC")
    if not (
        frame["high"].ge(frame[["open", "close"]].max(axis=1)).all()
        and frame["low"].le(frame[["open", "close"]].min(axis=1)).all()
    ):
        raise RuntimeError(f"{label} violates OHLC ordering")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    deep = pd.read_parquet(SOURCE_DEEP).copy()
    current = pd.read_parquet(SOURCE_CURRENT).copy()
    deep_time = pd.to_datetime(deep["time"], utc=True)
    current_time = pd.to_datetime(current["time"], utc=True)
    deep = deep[deep_time.le(CUTOFF)].copy()
    current = current[current_time.le(CUTOFF)].copy()
    _validate(deep, "deep baseline")
    _validate(current, "current baseline")
    current_dates = pd.to_datetime(current["time"], utc=True)
    if current_dates.max() != CUTOFF:
        raise RuntimeError("Current baseline must end exactly on 2026-07-22")
    if current_dates.duplicated().any():
        raise RuntimeError("Current baseline contains duplicate dates")

    deep_path = OUTPUT_DIR / "asset_daily_ohlcv_through_2026-07-22.parquet"
    current_path = OUTPUT_DIR / "usdcop_daily_ohlcv_through_2026-07-22.parquet"
    deep.to_parquet(deep_path, index=False)
    current.to_parquet(current_path, index=False)
    manifest = {
        "schema_version": "1.0.0",
        "experiment_id": "usdcop_h1_daily_shadow_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cutoff": CUTOFF.date().isoformat(),
        "historical_availability_class": "reconstructed_not_true_pit",
        "launch_rule": "NO_HISTORICAL_DAILY_PERFORMANCE_EVALUATION",
        "source_files": {
            str(SOURCE_DEEP.relative_to(ROOT)).replace("\\", "/"): sha256(SOURCE_DEEP),
            str(SOURCE_CURRENT.relative_to(ROOT)).replace("\\", "/"): sha256(SOURCE_CURRENT),
        },
        "source_current_mtime_utc": datetime.fromtimestamp(
            SOURCE_CURRENT.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
        "files": {},
    }
    for path, frame in ((deep_path, deep), (current_path, current)):
        dates = pd.to_datetime(frame["time"], utc=True)
        manifest["files"][path.name] = {
            "rows": int(len(frame)),
            "start": dates.min().isoformat(),
            "end": dates.max().isoformat(),
            "sha256": sha256(path),
        }
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "output_directory": str(OUTPUT_DIR.relative_to(ROOT)).replace("\\", "/"),
        "cutoff": manifest["cutoff"],
        "files": manifest["files"],
        "manifest_sha256": sha256(manifest_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
