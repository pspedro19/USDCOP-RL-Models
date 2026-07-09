#!/usr/bin/env python3
"""
Build Unified FX Seed — Multi-Pair OHLCV aligned to Bogota Time (8:00-12:55 COT).

Loads raw OHLCV seeds for USD/COP, USD/MXN, USD/BRL, normalizes timezones to
America/Bogota, filters to trading session, standardizes schema, validates quality,
and saves unified outputs.

Outputs:
    seeds/latest/usdcop_m5_ohlcv.parquet  ← Updated: fixed tz, standardized schema
    seeds/latest/usdmxn_m5_ohlcv.parquet  ← Updated: COT timestamps, added symbol col
    seeds/latest/usdbrl_m5_ohlcv.parquet   ← Updated: COT timestamps, added symbol col
    seeds/latest/fx_multi_m5_ohlcv.parquet ← NEW: All 3 pairs unified (for DB restore)

Usage:
    python scripts/build_unified_fx_seed.py
    python scripts/build_unified_fx_seed.py --dry-run   # Validate without saving

Author: Pedro @ Lean Tech Solutions
Created: 2026-02-12
"""

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEEDS_DIR = PROJECT_ROOT / "seeds" / "latest"

PAIR_CONFIG = {
    "USD/COP": {
        "seed": SEEDS_DIR / "usdcop_m5_ohlcv.parquet",
        "raw_tz": "UTC_mislabeled_as_bogota",
        "price_range": (3000, 6000),
    },
    "USD/MXN": {
        "seed": SEEDS_DIR / "usdmxn_m5_ohlcv.parquet",
        "raw_tz": "UTC_naive",
        "price_range": (10, 30),
    },
    "USD/BRL": {
        "seed": SEEDS_DIR / "usdbrl_m5_ohlcv.parquet",
        "raw_tz": "UTC_naive",
        "price_range": (3, 8),
    },
}

# Standardized output schema
OUTPUT_COLUMNS = ["time", "symbol", "open", "high", "low", "close", "volume"]

# Session window (America/Bogota)
SESSION_START_HOUR = 8
SESSION_END_HOUR = 12
SESSION_END_MINUTE = 55
BOGOTA_TZ = "America/Bogota"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1a. Load raw data
# ---------------------------------------------------------------------------

def load_raw(symbol: str, cfg: dict) -> pd.DataFrame:
    """Load raw parquet for a single pair."""
    path = cfg["seed"]
    if not path.exists():
        raise FileNotFoundError(f"Seed not found for {symbol}: {path}")

    df = pd.read_parquet(path)
    logger.info(f"[{symbol}] Raw load: {len(df):,} rows, columns={df.columns.tolist()}")

    # Normalize column names to lowercase
    df.columns = df.columns.str.lower()

    # Ensure 'time' column exists (some files use 'datetime')
    if "time" not in df.columns:
        for alt in ("datetime", "timestamp", "date"):
            if alt in df.columns:
                df = df.rename(columns={alt: "time"})
                break
        else:
            raise KeyError(f"[{symbol}] No time/datetime column found in {df.columns.tolist()}")

    # Parse time
    df["time"] = pd.to_datetime(df["time"])

    return df


# ---------------------------------------------------------------------------
# 1b. Fix timezones -> ALL to America/Bogota
# ---------------------------------------------------------------------------

def fix_timezone(df: pd.DataFrame, symbol: str, raw_tz: str) -> pd.DataFrame:
    """
    Normalize timestamps to America/Bogota.

    Cases:
      - UTC_mislabeled_as_bogota (USDCOP): tz_localize was used instead of tz_convert.
        Timestamps say "America/Bogota" but the NUMBERS are UTC (13-17).
        Fix: strip tz -> localize as UTC -> convert to Bogota.
      - UTC_naive (MXN/BRL): timestamps are UTC without tz info.
        Fix: localize as UTC -> convert to Bogota.
    """
    ts = df["time"]

    if raw_tz == "UTC_mislabeled_as_bogota":
        # Strip the wrong tz label, then treat as UTC
        if ts.dt.tz is not None:
            ts = ts.dt.tz_localize(None)
        ts = ts.dt.tz_localize("UTC").dt.tz_convert(BOGOTA_TZ)
        logger.info(f"[{symbol}] Fixed mislabeled tz: stripped -> UTC -> Bogota")

    elif raw_tz == "UTC_naive":
        if ts.dt.tz is not None:
            ts = ts.dt.tz_localize(None)
        ts = ts.dt.tz_localize("UTC").dt.tz_convert(BOGOTA_TZ)
        logger.info(f"[{symbol}] Fixed naive UTC: localized -> Bogota")

    else:
        raise ValueError(f"Unknown raw_tz={raw_tz} for {symbol}")

    df["time"] = ts
    return df


# ---------------------------------------------------------------------------
# 1c. Filter to exact session: Mon-Fri 8:00-12:55 COT
# ---------------------------------------------------------------------------

def filter_session(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Remove weekends and bars outside 8:00-12:55 COT."""
    n_before = len(df)

    # Remove weekends (dayofweek 5=Sat, 6=Sun)
    mask_weekday = df["time"].dt.dayofweek < 5

    # Session hours: 8:00 to 12:55 inclusive
    hour = df["time"].dt.hour
    minute = df["time"].dt.minute
    mask_session = (
        (hour >= SESSION_START_HOUR)
        & (
            (hour < SESSION_END_HOUR)
            | ((hour == SESSION_END_HOUR) & (minute <= SESSION_END_MINUTE))
        )
    )

    df = df[mask_weekday & mask_session].copy()
    n_removed = n_before - len(df)
    if n_removed > 0:
        logger.info(f"[{symbol}] Session filter: removed {n_removed:,} bars -> {len(df):,} remaining")
    else:
        logger.info(f"[{symbol}] Session filter: all {len(df):,} bars within session")

    return df


# ---------------------------------------------------------------------------
# 1d. Standardize schema
# ---------------------------------------------------------------------------

def standardize_schema(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Ensure all parquets have [time, symbol, open, high, low, close, volume]."""
    # Add symbol column
    df["symbol"] = symbol

    # Ensure volume exists (0 where missing)
    if "volume" not in df.columns:
        df["volume"] = 0
    df["volume"] = df["volume"].fillna(0)

    # Select only output columns, ensure OHLC exist
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise KeyError(f"[{symbol}] Missing required column: {col}")

    df = df[OUTPUT_COLUMNS].copy()

    # Sort by time and drop duplicates
    df = df.sort_values("time").drop_duplicates(subset=["time", "symbol"], keep="last")
    df = df.reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# 1e. Validate data quality
# ---------------------------------------------------------------------------

def validate_pair(df: pd.DataFrame, symbol: str, price_range: tuple) -> dict:
    """Per-pair quality checks. Returns dict with results + any issues."""
    issues = []
    lo, hi = price_range

    # Price range
    out_of_range = ((df["close"] < lo) | (df["close"] > hi)).sum()
    if out_of_range > 0:
        issues.append(f"Price out of range ({lo}-{hi}): {out_of_range} bars")

    # Duplicates on (time, symbol)
    dupes = df.duplicated(subset=["time", "symbol"]).sum()
    if dupes > 0:
        issues.append(f"Duplicate (time, symbol): {dupes}")

    # NaN in OHLC
    for col in ("open", "high", "low", "close"):
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            issues.append(f"NaN in {col}: {n_nan}")

    # OHLC integrity
    bad_hl = (df["high"] < df["low"]).sum()
    if bad_hl > 0:
        issues.append(f"high < low: {bad_hl} bars")

    bad_ho = (df["high"] < df["open"]).sum()
    if bad_ho > 0:
        issues.append(f"high < open: {bad_ho} bars")

    bad_hc = (df["high"] < df["close"]).sum()
    if bad_hc > 0:
        issues.append(f"high < close: {bad_hc} bars")

    bad_lo = (df["low"] > df["open"]).sum()
    if bad_lo > 0:
        issues.append(f"low > open: {bad_lo} bars")

    bad_lc = (df["low"] > df["close"]).sum()
    if bad_lc > 0:
        issues.append(f"low > close: {bad_lc} bars")

    # Bars per day stats
    dates = df["time"].dt.date
    bars_per_day = dates.value_counts()
    n_days = bars_per_day.nunique() if len(bars_per_day) > 0 else 0

    # Hour distribution (sanity check)
    hours = df["time"].dt.hour.unique()

    result = {
        "symbol": symbol,
        "rows": len(df),
        "days": len(bars_per_day),
        "date_min": str(df["time"].min()),
        "date_max": str(df["time"].max()),
        "bars_per_day_median": float(bars_per_day.median()) if len(bars_per_day) > 0 else 0,
        "hours": sorted(hours.tolist()),
        "issues": issues,
        "price_min": float(df["close"].min()),
        "price_max": float(df["close"].max()),
    }

    # Gap detection: days with <10 bars
    gap_days = (bars_per_day < 10).sum()
    result["gap_days"] = int(gap_days)

    return result


def cross_pair_alignment(dfs: dict) -> dict:
    """Check alignment across pairs on common date range."""
    # Get trading dates per pair
    dates_per_pair = {}
    for symbol, df in dfs.items():
        dates_per_pair[symbol] = set(df["time"].dt.date.unique())

    # Common date range
    all_dates = set()
    for d in dates_per_pair.values():
        all_dates |= d

    min_date = min(all_dates)
    max_date = max(all_dates)

    # Find common range start/end
    common_start = max(min(d) for d in dates_per_pair.values())
    common_end = min(max(d) for d in dates_per_pair.values())

    # Alignment percentages (based on COP as reference)
    cop_dates = dates_per_pair.get("USD/COP", set())
    alignments = {}
    for symbol, dates in dates_per_pair.items():
        if symbol == "USD/COP":
            continue
        if len(cop_dates) > 0:
            # Only compare within common range
            cop_in_range = {d for d in cop_dates if common_start <= d <= common_end}
            other_in_range = {d for d in dates if common_start <= d <= common_end}
            overlap = len(cop_in_range & other_in_range)
            alignments[f"COP-{symbol.split('/')[-1]}"] = (
                round(overlap / max(len(cop_in_range), 1) * 100, 1)
            )

    return {
        "common_start": str(common_start),
        "common_end": str(common_end),
        "alignments": alignments,
    }


# ---------------------------------------------------------------------------
# 1f. Save outputs
# ---------------------------------------------------------------------------

def compute_hash(df: pd.DataFrame) -> str:
    """Quick content hash for lineage tracking."""
    buf = df.to_csv(index=False).encode()
    return hashlib.md5(buf).hexdigest()[:12]


def save_outputs(dfs: dict, dry_run: bool = False):
    """Save individual per-pair parquets + unified multi-pair parquet."""
    output_paths = {}

    for symbol, df in dfs.items():
        # Per-pair file
        pair_slug = symbol.replace("/", "").lower()
        path = SEEDS_DIR / f"{pair_slug}_m5_ohlcv.parquet"
        output_paths[symbol] = path

        if not dry_run:
            df.to_parquet(path, index=False, engine="pyarrow")
            logger.info(f"[{symbol}] Saved {len(df):,} rows -> {path}")
        else:
            logger.info(f"[{symbol}] DRY RUN: would save {len(df):,} rows -> {path}")

    # Unified multi-pair file
    unified = pd.concat(list(dfs.values()), ignore_index=True)
    unified = unified.sort_values(["time", "symbol"]).reset_index(drop=True)
    unified_path = SEEDS_DIR / "fx_multi_m5_ohlcv.parquet"

    if not dry_run:
        unified.to_parquet(unified_path, index=False, engine="pyarrow")
        logger.info(f"[UNIFIED] Saved {len(unified):,} rows -> {unified_path}")
    else:
        logger.info(f"[UNIFIED] DRY RUN: would save {len(unified):,} rows -> {unified_path}")

    return output_paths, unified_path


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(validations: dict, alignment: dict):
    """Print summary report in tabular format."""
    print()
    print("=" * 78)
    print("  UNIFIED FX SEED REPORT")
    print("=" * 78)
    print()
    print(f"{'Par':<12} | {'Rows':>8} | {'Dias':>5} | {'Rango fechas':<35} | {'Bars/dia':>9} | {'Gaps':>4}")
    print("-" * 78)

    for symbol, v in validations.items():
        date_range = f"{v['date_min'][:10]} -> {v['date_max'][:10]}"
        print(
            f"{symbol:<12} | {v['rows']:>8,} | {v['days']:>5} | "
            f"{date_range:<35} | {v['bars_per_day_median']:>9.1f} | {v['gap_days']:>4}"
        )

    print("-" * 78)
    print(f"Rango comun: {alignment['common_start']} -> {alignment['common_end']}")
    for key, pct in alignment["alignments"].items():
        print(f"Alineamiento {key}: {pct}%")
    print(f"Timezone: {BOGOTA_TZ} | Sesion: {SESSION_START_HOUR:02d}:00-{SESSION_END_HOUR}:{SESSION_END_MINUTE} COT")
    print("=" * 78)
    print()

    # Print issues
    any_issues = False
    for symbol, v in validations.items():
        if v["issues"]:
            any_issues = True
            for issue in v["issues"]:
                print(f"  WARNING [{symbol}]: {issue}")

    if not any_issues:
        print("  All quality checks PASSED.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build unified FX seed files")
    parser.add_argument("--dry-run", action="store_true", help="Validate without saving")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("BUILD UNIFIED FX SEED")
    logger.info("=" * 60)

    processed = {}
    validations = {}

    for symbol, cfg in PAIR_CONFIG.items():
        logger.info(f"\n{'='*40} {symbol} {'='*40}")

        # 1a. Load raw
        try:
            df = load_raw(symbol, cfg)
        except FileNotFoundError as e:
            logger.warning(f"[{symbol}] Skipping: {e}")
            continue

        # 1b. Fix timezone
        df = fix_timezone(df, symbol, cfg["raw_tz"])

        # Verify hours are now 8-12 COT
        hours_after = sorted(df["time"].dt.hour.unique())
        logger.info(f"[{symbol}] Hours after tz fix: {hours_after}")

        # 1c. Filter session
        df = filter_session(df, symbol)

        # 1d. Standardize schema
        df = standardize_schema(df, symbol)

        # 1e. Validate
        val = validate_pair(df, symbol, cfg["price_range"])
        validations[symbol] = val

        # Verify final hours
        final_hours = sorted(df["time"].dt.hour.unique())
        logger.info(f"[{symbol}] Final hours: {final_hours}")
        assert all(SESSION_START_HOUR <= h <= SESSION_END_HOUR for h in final_hours), (
            f"[{symbol}] Hours outside session: {final_hours}"
        )

        processed[symbol] = df

    if not processed:
        logger.error("No pairs processed. Check seed files in seeds/latest/")
        return 1

    # Cross-pair alignment
    alignment = cross_pair_alignment(processed)

    # Print report
    print_report(validations, alignment)

    # 1f. Save
    if not args.dry_run:
        save_outputs(processed, dry_run=False)
        logger.info("All seed files saved successfully.")
    else:
        save_outputs(processed, dry_run=True)
        logger.info("DRY RUN complete. No files written.")

    # Return 1 if any critical issues
    for v in validations.values():
        critical = [i for i in v["issues"] if "NaN" in i or "high < low" in i]
        if critical:
            logger.error(f"Critical issues found: {critical}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
